#!/usr/bin/env python3
"""Prepare, execute and fetch Step1 tests using a remote nightly environment."""
import argparse
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import time
import zipfile


def write_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def normalize(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def inventory(python):
    code = '''import importlib.metadata as m,json,sys
r=[]
for d in m.distributions():
 try: direct=json.loads(d.read_text('direct_url.json') or '{}')
 except Exception: direct={}
 r.append({'name':d.metadata['Name'],'version':d.version,'direct':direct})
print(json.dumps({'python':list(sys.version_info[:3]),'packages':r}))'''
    return json.loads(subprocess.check_output([str(python), '-c', code], text=True))


def portable_packages(packages, exclude):
    kept, omitted, unresolved = {}, [], []
    for item in packages:
        name = normalize(item['name'])
        url = item.get('direct', {}).get('url', '')
        if name in {'torch', 'torch-npu', 'torchvision', 'torchaudio', 'torch-musa', 'triton'} or name.startswith('nvidia-'):
            reason = 'Replaced by target PyTorch nightly runtime'
        elif name in exclude:
            reason = exclude[name]
        elif any(s in url.lower() for s in ['/ascend/', '/selfgz', '/cann/']):
            reason = 'Source Ascend/CANN installation; unavailable on CUDA host'
        elif item.get('direct'):
            unresolved.append(item['name'])
            continue
        else:
            kept[name] = item['version']
            continue
        omitted.append({'name': item['name'], 'version': item['version'], 'reason': reason})
    if unresolved:
        raise ValueError('Local/editable/URL packages need explicit portable replacements or exclude_packages reasons: ' + ', '.join(unresolved))
    return kept, omitted


def contained(root, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or '..' in path.parts or '\\' in relative:
        raise ValueError('Invalid relative path: ' + relative)
    full = (root / relative).resolve()
    if not full.is_relative_to(root) or not full.is_file():
        raise ValueError('Missing/outside-source file: ' + relative)
    return full


def prepare(config, output):
    output.mkdir(parents=True, exist_ok=False)
    root = Path(config['source_root']).resolve()
    results_path = Path(config['results']).resolve()
    results = json.loads(results_path.read_text(encoding='utf-8'))
    inv = inventory(config['source_python'])
    kept, omitted = portable_packages(inv['packages'], {normalize(k): v for k, v in config.get('exclude_packages', {}).items()})
    device = config.get('device', 'cuda')
    index = config['torch_index'].rstrip('/')
    channel = index.rsplit('/', 1)[-1]
    if not re.fullmatch(r'https://download\.pytorch\.org/whl/nightly/(cpu|cu[0-9]+)', index):
        raise ValueError('torch_index must be an official CPU/CUDA nightly channel')
    if device not in {'cpu', 'cuda'} or (device == 'cpu') != (channel == 'cpu'):
        raise ValueError('device and nightly channel disagree')
    entries, files = [], {}
    for entry in results['results']:
        paths = entry.get('test_files', [])
        if not paths:
            raise ValueError('No test files: ' + entry['api'])
        for relative in paths:
            files[relative] = contained(root, relative)
        entries.append({'api': entry['api'], 'test_files': paths})
    for relative in config.get('extra_paths', []):
        full = (root / relative).resolve()
        if not full.is_relative_to(root):
            raise ValueError('extra_paths outside source root')
        paths = list(full.rglob('*.py')) if full.is_dir() else [full]
        for path in paths:
            rel = path.relative_to(root).as_posix()
            files[rel] = contained(root, rel)
    # Preserve package imports without copying either implementation repository.
    for relative in list(files):
        parent = PurePosixPath(relative).parent
        while str(parent) != '.':
            init = str(parent / '__init__.py')
            if (root / init).is_file():
                files[init] = contained(root, init)
            parent = parent.parent
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    manifest = {'schema_version': 1, 'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'device': device,
                'python_minor': inv['python'][:2], 'torch_index': index, 'torch_requirement': config.get('torch_requirement', 'torch'),
                'packages': kept, 'excluded_packages': omitted, 'files': hashes, 'results': entries,
                'timeout_per_file': int(config.get('timeout_per_file', 300)), 'install_timeout': int(config.get('install_timeout', 1800))}
    if not re.fullmatch(r'torch(?:==[A-Za-z0-9.+_-]+)?', manifest['torch_requirement']):
        raise ValueError('torch_requirement must be torch or a pinned torch version')
    for key, command in [('source_commit', ['git', '-C', str(root), 'rev-parse', 'HEAD']), ('source_branch', ['git', '-C', str(root), 'branch', '--show-current'])]:
        try:
            manifest[key] = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            manifest[key] = 'unknown'
    write_json(output / 'manifest.json', manifest)
    (output / 'requirements.txt').write_text(''.join(n + '==' + v + '\n' for n, v in sorted(kept.items())))
    with zipfile.ZipFile(output / 'bundle.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, 'tests/' + name)
        archive.write(output / 'manifest.json', 'manifest.json')
        archive.write(output / 'requirements.txt', 'requirements.txt')
        archive.write(Path(__file__).with_name('step1_remote_worker.py'), 'worker.py')
    return manifest


def connect(config):
    try:
        import paramiko
    except ImportError as error:
        raise RuntimeError('Install paramiko in the controller environment (not the source test environment)') from error
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    pin = config.get('host_key_sha256')
    if pin:
        class PinnedKey(paramiko.MissingHostKeyPolicy):
            def missing_host_key(self, client, hostname, key):
                actual = 'SHA256:' + base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip('=')
                if actual != pin:
                    raise RuntimeError('Remote host key does not match configured fingerprint')
        client.set_missing_host_key_policy(PinnedKey())
    else:
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
    password = os.environ.get(config.get('password_env', 'API_ADAPT_SSH_PASSWORD'))
    client.connect(config['host'], port=int(config.get('port', 22)), username=config['user'], password=password,
                   key_filename=config.get('identity_file'), allow_agent=not bool(password), look_for_keys=not bool(password),
                   timeout=20, auth_timeout=20, banner_timeout=20)
    client.get_transport().set_keepalive(15)
    return client


def remote_command(client, command, log, timeout):
    channel = client.get_transport().open_session(timeout=20)
    channel.set_combine_stderr(True)
    channel.exec_command(command)
    deadline = time.monotonic() + timeout
    with Path(log).open('wb') as stream:
        while True:
            if channel.recv_ready():
                stream.write(channel.recv(65536))
                stream.flush()
            elif channel.exit_status_ready():
                break
            elif time.monotonic() > deadline:
                channel.close()
                raise TimeoutError('Remote command timed out; use fetch to retrieve any completed evidence')
            else:
                time.sleep(0.1)
    return channel.recv_exit_status()


def remote_path(config, output):
    root = PurePosixPath(config['remote_root'])
    if not root.is_absolute() or not re.fullmatch(r'[A-Za-z0-9._-]+', output.name):
        raise ValueError('Use an absolute remote_root and a simple unique output directory name')
    return str(root / output.name)


def fetch(client, config, output):
    remote = remote_path(config, output)
    target = output / 'evidence'
    target.mkdir(exist_ok=True)
    with client.open_sftp() as sftp:
        def download(source, dest):
            import stat
            for item in sftp.listdir_attr(source):
                if item.filename in {'.', '..'} or '/' in item.filename or '\\' in item.filename:
                    raise ValueError('Invalid evidence filename')
                local = dest / item.filename
                if stat.S_ISDIR(item.st_mode):
                    local.mkdir(exist_ok=True)
                    download(source + '/' + item.filename, local)
                elif stat.S_ISREG(item.st_mode):
                    sftp.get(source + '/' + item.filename, str(local))
        download(remote + '/evidence', target)
    report = json.loads((target / 'report.json').read_text())
    manifest = json.loads((output / 'manifest.json').read_text())
    if report.get('manifest_sha256') != hashlib.sha256((output / 'manifest.json').read_bytes()).hexdigest():
        raise ValueError('Fetched report belongs to a different manifest')
    if report.get('device') != manifest['device'] or [x['api'] for x in report['results']] != [x['api'] for x in manifest['results']]:
        raise ValueError('Fetched device/API scope mismatch')
    for entry in report['results']:
        if entry['execution'] in {'pass', 'fail'}:
            if not entry.get('files'):
                raise ValueError('Completed API lacks file reports')
            for file in entry['files']:
                if file.get('sha256') != manifest['files'].get(file['file']):
                    raise ValueError('Fetched source hash mismatch')
    return report


def merge(config, output, report):
    path = Path(config['results']).resolve()
    source = Path(config['source_root']).resolve()
    manifest = json.loads((output / 'manifest.json').read_text())
    for relative, digest in manifest['files'].items():
        if hashlib.sha256(contained(source, relative).read_bytes()).hexdigest() != digest:
            raise ValueError('Source changed after packaging: ' + relative)
    data = json.loads(path.read_text())
    lookup = {e['api']: e for e in data['results']}
    run = Path(config['run_dir']).resolve()
    relative_report = (output / 'evidence/report.json').resolve().relative_to(run).as_posix()
    for result in report['results']:
        entry = lookup[result['api']]
        if entry['test_files'] != next(e['test_files'] for e in manifest['results'] if e['api'] == entry['api']):
            raise ValueError('Current task file scope changed')
        entry.setdefault('device_validation', {})[manifest['device']] = dict(result, report=relative_report, environment=report.get('environment', {}))
        # A remote reference device does not establish NPU support.
        if entry.get('execution', 'not_run') not in {'pass', 'fail'}:
            entry['execution'] = 'partial' if result['execution'] in {'pass', 'fail', 'partial'} else 'blocked'
            entry['next_step'] = 'step1'
    backup = output / 'result.before-merge.json'
    if not backup.exists():
        shutil.copy2(path, backup)
    write_json(path, data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'fetch'])
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--merge', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.output.resolve()
    if args.action == 'prepare':
        manifest = prepare(config, output)
        print(json.dumps({'status': 'PREPARED', 'apis': len(manifest['results']), 'portable_packages': len(manifest['packages']), 'excluded_packages': len(manifest['excluded_packages']), 'bundle': str(output / 'bundle.zip')}))
        return 0
    if args.action == 'run' and not output.exists():
        prepare(config, output)
    client = connect(config)
    try:
        if args.action == 'run':
            remote = remote_path(config, output)
            python = shlex.quote(config['remote_python'])
            command = python + ' -c ' + shlex.quote('import pathlib; pathlib.Path(' + repr(remote) + ').mkdir(parents=True,exist_ok=False)')
            if remote_command(client, command, output / 'connect.log', 30):
                raise RuntimeError('Remote directory already exists or cannot be created; use fetch or a new output name')
            with client.open_sftp() as sftp:
                sftp.put(str(output / 'bundle.zip'), remote + '/bundle.zip')
            unpack = 'import zipfile; zipfile.ZipFile(' + repr(remote + '/bundle.zip') + ').extractall(' + repr(remote) + ')'
            command = python + ' -c ' + shlex.quote(unpack) + ' && cd ' + shlex.quote(remote) + ' && ' + python + ' worker.py'
            code = remote_command(client, command, output / 'remote.log', int(config.get('remote_timeout', 7200)))
            print('Remote exit:', code)
        report = fetch(client, config, output)
        if args.merge:
            merge(config, output, report)
        print(json.dumps({'status': report['execution'], 'device': report['device'], 'results': [{'api': r['api'], 'execution': r['execution']} for r in report['results']]}, ensure_ascii=False))
        return 0 if report['execution'] == 'pass' else 2
    finally:
        client.close()


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        print('REMOTE_STEP1_ERROR: ' + str(error), file=sys.stderr)
        sys.exit(2)
