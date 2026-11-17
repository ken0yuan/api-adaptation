#!/usr/bin/env python3
"""Workstation bridge: source server -> GitHub/WSL -> source server, using native SSH."""
import argparse
import base64
import io
import hashlib
import json
import os
from pathlib import Path
import queue
import shlex
import subprocess
import sys
import threading
import zipfile

from step1_relay import SourceShell
from step1_wsl_gpu import fetch, submit


class NativeChannel:
    def __init__(self, alias):
        if not alias or alias.startswith('-'):
            raise ValueError('Invalid source SSH alias')
        self.proc = subprocess.Popen(['ssh', '-tt', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', alias],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.pending = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        while True:
            data = os.read(self.proc.stdout.fileno(), 65536)
            if not data:
                break
            self.pending.put(data)

    @property
    def closed(self):
        return self.proc.poll() is not None and self.pending.empty()

    def sendall(self, data):
        self.proc.stdin.write(data.encode() if isinstance(data, str) else data)
        self.proc.stdin.flush()

    def recv_ready(self):
        return not self.pending.empty()

    def recv(self, count):
        return self.pending.get_nowait()

    def close(self):
        self.proc.terminate()
        self.proc.wait(timeout=10)
        self.proc.stdin.close()
        self.proc.stdout.close()


class NativeClient:
    def __init__(self, alias):
        self.channel = NativeChannel(alias)

    def invoke_shell(self, **kwargs):
        return self.channel

    def close(self):
        self.channel.close()


def update_request(shell, config, output, status, **fields):
    if not config.get('request_path'):
        return
    sha = hashlib.sha256((output / 'manifest.json').read_bytes()).hexdigest()
    code = '''from pathlib import Path
import json,sys
sys.path.insert(0,SCRIPTS)
from step1_remote import write_json
p=Path(REQUEST);ticket=json.loads(p.read_text())
if ticket['manifest_sha256']!=SHA:raise ValueError('Request identity changed')
ticket.update(FIELDS);write_json(p,ticket);result=True
'''.replace('SCRIPTS', repr(config['source']['skill_dir'] + '/scripts')).replace(
        'REQUEST', repr(config['request_path'])).replace('SHA', repr(sha)).replace(
        'FIELDS', repr(dict(fields, status=status)))
    shell.call(code)


def assist(alias, request_path, output, wait_seconds):
    client = NativeClient(alias)
    try:
        shell = SourceShell(client, 'python3')
        ticket = shell.call('from pathlib import Path\nimport json\nresult=json.loads(Path(' + repr(request_path) + ').read_text())')
    finally:
        client.close()
    if ticket.get('type') != 'step1_wsl_gpu' or ticket.get('schema_version') != 1:
        raise ValueError('Only Step1 WSL GPU validation requests are supported')
    if ticket['request_path'] != request_path:
        raise ValueError('Request path does not match its identity')
    config = {'source': dict(ticket['source'], ssh_alias=alias), 'branch': ticket['branch'], 'manifest_sha256': ticket['manifest_sha256'],
              'request_path': request_path}
    if output.exists():
        if hashlib.sha256((output / 'manifest.json').read_bytes()).hexdigest() != ticket['manifest_sha256']:
            raise ValueError('Local output belongs to another request')
        state = json.loads((output / 'submission.json').read_text(encoding='utf-8'))
        if state['branch'] != config['branch']:
            raise ValueError('Local submission belongs to another request branch')
    config_file = output.with_name(output.name + '.bridge.json')
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    return execute('fetch' if output.exists() else 'run', config, output, wait_seconds)


def execute(action, config, output, wait_seconds):
    source = config['source']
    if Path(source['output']).name != output.name:
        raise ValueError('Source and workstation output basenames must match')
    if action == 'run' and output.exists():
        raise ValueError('Existing output: use fetch to resume without another dispatch')
    client = NativeClient(source['ssh_alias'])
    try:
        shell = SourceShell(client, shlex.quote(source['python']))
        if action == 'run':
            print('Exporting tests and portable dependencies from source', flush=True)
            code = '''import subprocess,pathlib,base64,json,sys,hashlib,io,zipfile
out=pathlib.Path(OUTPUT)
if not out.exists():
 p=subprocess.run(ARGS,capture_output=True,text=True)
 if p.returncode: raise RuntimeError(p.stdout+p.stderr)
cfg=json.loads(pathlib.Path(CONFIG).read_text())
raw=(out/'manifest.json').read_bytes()
if EXPECTED and hashlib.sha256(raw).hexdigest()!=EXPECTED: raise ValueError('Source export changed after request')
m=json.loads(raw)
if m['device']!='cuda' or m['torch_index']!=cfg['torch_index']: raise ValueError('Export config mismatch')
sys.path.insert(0,SCRIPTS)
from step1_remote import export_inputs
export_inputs(cfg,json.loads(pathlib.Path(cfg['results']).read_text(encoding='utf-8')),m)
packed=(out/'bundle.zip').read_bytes()
with zipfile.ZipFile(io.BytesIO(packed)) as z:
 if z.read('manifest.json')!=raw: raise ValueError('Bundle does not match source export')
result=base64.b64encode(packed).decode()
'''.replace('OUTPUT', repr(source['output'])).replace('CONFIG', repr(source['config'])).replace(
                'EXPECTED', repr(config.get('manifest_sha256'))).replace('SCRIPTS', repr(source['skill_dir'] + '/scripts')).replace(
                'ARGS', repr([source['python'], source['skill_dir'] + '/scripts/step1_remote.py',
                              'prepare', '--config', source['config'], '--output', source['output']]))
            bundle = output.with_name(output.name + '.bundle.zip')
            bundle.parent.mkdir(parents=True, exist_ok=True)
            bundle.write_bytes(base64.b64decode(shell.call(code)))
            state = submit(bundle, output, config['branch'])
            print('Run ID %s | https://github.com/%s/actions/runs/%s' % (state['run_id'], state['repository'], state['run_id']), flush=True)
        state = json.loads((output / 'submission.json').read_text(encoding='utf-8'))
        update_request(shell, config, output, 'running', run_id=state.get('run_id'), head_sha=state.get('head_sha'))
        report = fetch(output, wait_seconds)
        if report is None:
            return 3
        print('Returning GPU evidence to source; preserving NPU fields', flush=True)
        package = io.BytesIO()
        with zipfile.ZipFile(package, 'w', zipfile.ZIP_DEFLATED) as archive:
            for file in [output / 'submission.json', output / 'run.json', *(output / 'evidence').rglob('*')]:
                if file.is_file():
                    archive.write(file, file.relative_to(output).as_posix())
        shell.upload(source['output'] + '/wsl-evidence.zip', package.getvalue())
        code = '''from pathlib import Path
import zipfile,json,sys
out=Path(OUTPUT)
with zipfile.ZipFile(out/'wsl-evidence.zip') as z:
 for n in z.namelist():
  if not (out/n).resolve().is_relative_to(out.resolve()) or '\\\\' in n: raise ValueError('Invalid evidence path')
 z.extractall(out)
sys.path.insert(0,SCRIPTS)
from step1_wsl_gpu import merge
result=merge(json.loads(Path(CONFIG).read_text()),out)
'''.replace('OUTPUT', repr(source['output'])).replace('SCRIPTS', repr(source['skill_dir'] + '/scripts')).replace('CONFIG', repr(source['config']))
        print(json.dumps(shell.call(code), ensure_ascii=False, indent=2), flush=True)
        update_request(shell, config, output, 'result_received', run_id=report['run_id'],
                       execution=report['execution'], report=source['output'] + '/evidence/report.json')
        return 0 if report['execution'] == 'pass' else 2
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'fetch', 'assist'])
    parser.add_argument('--config', type=Path)
    parser.add_argument('--source-ssh', help='Existing workstation SSH alias for the source server')
    parser.add_argument('--request', help='Absolute Step1 request JSON path on the source server')
    parser.add_argument('--local-output', required=True, type=Path)
    parser.add_argument('--wait-seconds', type=int, default=0)
    parser.add_argument('--watch', action='store_true', help='Poll until completion, then return the evidence')
    args = parser.parse_args()
    wait = None if args.watch else max(0, min(args.wait_seconds, 60))
    if args.action == 'assist':
        if not args.source_ssh or not args.request:
            parser.error('assist requires --source-ssh and --request')
        return assist(args.source_ssh, args.request, args.local_output.resolve(), wait)
    if not args.config:
        parser.error('run/fetch require --config')
    return execute(args.action, json.loads(args.config.read_text(encoding='utf-8')), args.local_output.resolve(), wait)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        print('STEP1_WSL_BRIDGE_ERROR: ' + str(error), file=sys.stderr)
        sys.exit(2)
