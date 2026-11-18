#!/usr/bin/env python3
"""Run Step1 from a workstation that can SSH to both isolated servers."""
import argparse
import base64
import getpass
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import uuid
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step1_remote import connect


class SourceShell:
    """Support normal SSH shells and forced interactive container entrypoints."""
    def __init__(self, client, python):
        self.channel = client.invoke_shell(width=200, height=40)
        self.python = python
        self.channel.sendall("stty -echo; unset PROMPT_COMMAND; PS1=''\n")

    def call(self, code, timeout=120):
        token = 'STEP1_' + uuid.uuid4().hex
        wrapper = r'''import json,base64,textwrap,traceback
try:
 exec(CODE)
 response={'ok':True,'result':result}
except Exception:
 response={'ok':False,'error':traceback.format_exc()}
print('\nBEGIN_TOKEN')
print('\n'.join(textwrap.wrap(base64.b64encode(json.dumps(response).encode()).decode(),60)))
print('END_TOKEN',flush=True)
'''.replace('CODE', repr(code)).replace('BEGIN_TOKEN', token + '_BEGIN').replace('END_TOKEN', token + '_END')
        encoded = base64.b64encode(wrapper.encode()).decode()
        command = self.python + ' -c ' + shlex.quote("import base64;exec(base64.b64decode('" + encoded + "'))") + '\n'
        if len(command) > 4000:
            raise ValueError('Source command too large; reduce upload chunk size')
        self.channel.sendall(command)
        deadline, output = time.monotonic() + timeout, ''
        while time.monotonic() < deadline:
            if self.channel.recv_ready():
                output += self.channel.recv(65536).decode(errors='replace')
                clean = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', output).replace('\r', '')
                match = re.search(r'(?:^|\n)' + token + r'_BEGIN\n([\s\S]*?)\n' + token + '_END', clean)
                if match:
                    response = json.loads(base64.b64decode(re.sub(r'\s', '', match.group(1))))
                    if not response['ok']:
                        raise RuntimeError(response['error'])
                    return response['result']
            elif self.channel.closed:
                raise RuntimeError('Source SSH shell closed')
            else:
                time.sleep(0.05)
        raise TimeoutError('Source SSH operation timed out')

    def upload(self, path, data):
        self.call('from pathlib import Path\np=Path(' + repr(path) + ');p.write_bytes(b\'\');result=True')
        encoded = base64.b64encode(data).decode()
        for start in range(0, len(encoded), 1200):
            code = 'import base64\nwith open(' + repr(path) + ",'ab') as f: f.write(base64.b64decode(" + repr(encoded[start:start+1200]) + '))\nresult=True'
            self.call(code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--local-output', required=True, type=Path)
    parser.add_argument('--resume', action='store_true', help='Fetch existing target evidence and return it to source')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    source = cfg['source']
    output = args.local_output.resolve()
    if output.exists() and not args.resume:
        raise ValueError('Use a new local-output directory')
    if args.resume and not (output / 'target-connection.json').exists():
        raise ValueError('Resume requires an existing local target-connection.json')
    # The remote directory is named after local-output, so use the same basename.
    if Path(source['output']).name != output.name:
        raise ValueError('Source output and local output must have the same basename')
    client = connect(source)
    try:
        shell = SourceShell(client, shlex.quote(source['python']))
        code = '''import subprocess,pathlib,base64,json
args=ARGS
p=subprocess.run(args,capture_output=True,text=True)
if p.returncode: raise RuntimeError(p.stdout+p.stderr)
out=pathlib.Path(OUTPUT)
result={'manifest':base64.b64encode((out/'manifest.json').read_bytes()).decode(),'bundle':base64.b64encode((out/'bundle.zip').read_bytes()).decode(),'config':json.loads(pathlib.Path(CONFIG).read_text())}
'''.replace('ARGS', repr([source['python'], source['skill_dir'] + '/scripts/step1_remote.py', 'prepare', '--config', source['config'], '--output', source['output']])).replace('OUTPUT', repr(source['output'])).replace('CONFIG', repr(source['config']))
        if args.resume:
            target = json.loads((output / 'target-connection.json').read_text())
        else:
            print('Preparing tests and dependency versions on source server', flush=True)
            prepared = shell.call(code)
            output.mkdir(parents=True)
            for name in ['manifest', 'bundle']:
                (output / (name + ('.json' if name == 'manifest' else '.zip'))).write_bytes(base64.b64decode(prepared[name]))
            target = prepared['config']
        target.update(cfg.get('target_auth', {}))
        target.pop('source_python', None)
        # No credential is persisted; target_auth accepts key paths/env names.
        if 'password' in target:
            raise ValueError('Use password_env or identity_file, not a plaintext password config')
        env_name = target.get('password_env', 'API_ADAPT_SSH_PASSWORD')
        if not target.get('identity_file') and not os.environ.get(env_name):
            os.environ[env_name] = getpass.getpass('Target SSH password: ')
        target_config = output / 'target-connection.json'
        target_config.write_text(json.dumps(target, indent=2))
        print('Fetching target test evidence' if args.resume else 'Running isolated tests on target and fetching evidence', flush=True)
        command = [sys.executable, str(Path(__file__).with_name('step1_remote.py')), 'fetch' if args.resume else 'run', '--config', str(target_config), '--output', str(output)]
        code = subprocess.call(command)
        if not (output / 'evidence/report.json').exists():
            raise RuntimeError('No fetched target report; retain local output for recovery')
        package = io.BytesIO()
        with zipfile.ZipFile(package, 'w', zipfile.ZIP_DEFLATED) as archive:
            for file in (output / 'evidence').rglob('*'):
                if file.is_file():
                    archive.write(file, file.relative_to(output).as_posix())
        print('Returning evidence to source server and merging per-device results', flush=True)
        shell.upload(source['output'] + '/relay-evidence.zip', package.getvalue())
        merge_code = '''import pathlib,zipfile,json,sys
out=pathlib.Path(OUTPUT)
with zipfile.ZipFile(out/'relay-evidence.zip') as z:
 for n in z.namelist():
  if not (out/n).resolve().is_relative_to(out.resolve()): raise ValueError('Invalid evidence path')
 z.extractall(out)
sys.path.insert(0,SCRIPTS)
from step1_remote import merge
config=json.loads(pathlib.Path(CONFIG).read_text())
report=json.loads((out/'evidence/report.json').read_text())
merge(config,out,report)
result={'execution':report['execution'],'device':report['device'],'report':str(out/'evidence/report.json')}
'''.replace('OUTPUT', repr(source['output'])).replace('SCRIPTS', repr(source['skill_dir'] + '/scripts')).replace('CONFIG', repr(source['config']))
        print(json.dumps(shell.call(merge_code), ensure_ascii=False), flush=True)
        return code
    finally:
        client.close()


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        print('STEP1_RELAY_ERROR: ' + str(error), file=sys.stderr)
        sys.exit(2)
