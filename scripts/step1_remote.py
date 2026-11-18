#!/usr/bin/env python3
"""Prepare, execute and fetch Step1 tests using a remote PyTorch environment."""
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
import tempfile
import time
import zipfile

from adaptation_target import equivalent_requirements, gpu_target, matches_build


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
            reason = 'Replaced by target PyTorch runtime'
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


def collect_files(root, test_paths, extra_paths):
    root = Path(root).resolve()
    files = {name: contained(root, name) for name in test_paths}
    for relative in extra_paths:
        path = PurePosixPath(relative)
        if path.is_absolute() or '..' in path.parts or '\\' in relative:
            raise ValueError('Invalid helper path: ' + relative)
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
    return files


def export_inputs(config, data, manifest=None, *, legacy_subset=False):
    """Select this dispatch and compare the files with each generator NPU run."""
    root = Path(config['source_root']).resolve()
    if manifest and 'torch_index' in config:
        index, req = gpu_target(config)
        if index != manifest.get('torch_index') or not equivalent_requirements(req, manifest.get('torch_requirement', 'torch')):
            raise ValueError('Saved GPU export belongs to a different PyTorch target')
    rows = data['results']
    handoff = bool(manifest and manifest.get('npu_handoff')) or any(
        'handoff_files' in row.get('device_validation', {}).get('npu', {}) for row in rows)
    if handoff:
        rows = [row for row in rows if row.get('strategy', '补充testcase') == '补充testcase']
    apis = config.get('apis')
    if legacy_subset and not handoff and apis is None:
        apis = [entry['api'] for entry in manifest['results']]
    if apis is not None:
        if not isinstance(apis, list) or not apis or len(apis) != len(set(apis)):
            raise ValueError('apis must be a nonempty list of distinct task APIs')
        lookup = {row['api']: row for row in rows}
        if set(apis) - lookup.keys():
            raise ValueError('Selected APIs are outside the validation scope')
        rows = [lookup[api] for api in apis]
    independent_apis = config.get('device_independent_apis')
    if independent_apis is None:
        independent_apis = [entry['api'] for entry in (manifest or {}).get('results', [])
                            if entry.get('device_independent') is True]
    if (not isinstance(independent_apis, list) or any(not isinstance(api, str) for api in independent_apis)
            or len(independent_apis) != len(set(independent_apis))
            or set(independent_apis) - {entry['api'] for entry in data['results']}):
        raise ValueError('device_independent_apis must list distinct APIs from the current task')
    entries, files, hashes = [], {}, {}
    for row in rows:
        paths = row.get('test_files')
        if not paths:
            raise ValueError('No test files: ' + row['api'])
        entry = {'api': row['api'], 'test_files': paths}
        if row['api'] in independent_apis:
            entry['device_independent'] = True
        entries.append(entry)
        extras = config.get('extra_paths', [])
        if handoff:
            from npu_handoff import verify_evidence
            if 'handoff_files' not in row.get('device_validation', {}).get('npu', {}):
                raise ValueError('Missing NPU handoff for selected API: ' + row['api'])
            snapshot = verify_evidence(row, Path(config['run_dir']).resolve())
            extras = snapshot['extra_paths']
        selected = collect_files(root, paths, extras)
        for name, path in selected.items():
            if name not in files:
                files[name] = path
                hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if handoff:
            if snapshot['files'] != {name: hashes[name] for name in selected}:
                raise ValueError('NPU handoff files changed: ' + row['api'])
    if not entries or len(entries) != len({row['api'] for row in entries}):
        raise ValueError('Empty or duplicate API scope')
    if handoff and 'extra_paths' in config:
        tests = [path for entry in entries for path in entry['test_files']]
        if collect_files(root, tests, config['extra_paths']).keys() != files.keys():
            raise ValueError('Configured helper set differs from selected NPU attempts')
    if manifest is not None:
        if entries != manifest['results']:
            raise ValueError('Export API/file scope changed')
        # Archived legacy configurations may omit extra_paths. Verify their original manifest.
        if not handoff:
            hashes = {name: hashlib.sha256(contained(root, name).read_bytes()).hexdigest()
                      for name in manifest['files']}
        if hashes != manifest['files']:
            raise ValueError('Source changed after export')
    return entries, files, hashes, handoff


def prepare(config, output):
    root = Path(config['source_root']).resolve()
    results = json.loads(Path(config['results']).read_text(encoding='utf-8'))
    entries, files, hashes, handoff = export_inputs(config, results)
    inv = inventory(config['source_python'])
    kept, omitted = portable_packages(inv['packages'], {normalize(k): v for k, v in config.get('exclude_packages', {}).items()})
    device = config.get('device', 'cuda')
    index, requirement = gpu_target(config)
    if '/nightly/' not in index:
        source_torch = next((pkg['version'] for pkg in inv['packages'] if normalize(pkg['name']) == 'torch'), None)
        if not matches_build(source_torch, requirement, False):
            raise ValueError('Source torch version disagrees with the fixed release')
    manifest = {'schema_version': 1, 'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'device': device,
                'python_minor': inv['python'][:2], 'torch_index': index, 'torch_requirement': requirement,
                'packages': kept, 'excluded_packages': omitted, 'files': hashes, 'results': entries,
                'timeout_per_file': int(config.get('timeout_per_file', 300)), 'install_timeout': int(config.get('install_timeout', 1800))}
    if handoff:
        manifest['npu_handoff'] = True
    for key, command in [('source_commit', ['git', '-C', str(root), 'rev-parse', 'HEAD']), ('source_branch', ['git', '-C', str(root), 'branch', '--show-current'])]:
        try:
            manifest[key] = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            manifest[key] = 'unknown'
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp) / 'export'
        stage.mkdir()
        write_json(stage / 'manifest.json', manifest)
        (stage / 'requirements.txt').write_text(''.join(n + '==' + v + '\n' for n, v in sorted(kept.items())))
        with zipfile.ZipFile(stage / 'bundle.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for name, path in files.items():
                content = path.read_bytes()
                if hashlib.sha256(content).hexdigest() != hashes[name]:
                    raise ValueError('Source changed during export: ' + name)
                archive.writestr('tests/' + name, content)
            archive.write(stage / 'manifest.json', 'manifest.json')
            archive.write(stage / 'requirements.txt', 'requirements.txt')
            archive.write(Path(__file__).with_name('step1_remote_worker.py'), 'worker.py')
            archive.write(Path(__file__).with_name('adaptation_target.py'), 'adaptation_target.py')
        stage.rename(output)
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
    from npu_handoff import record_lock
    path = Path(config['results']).resolve()
    with record_lock(path):
        return _merge(config, output, report, path)


def _merge(config, output, report, path):
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    data = json.loads(path.read_text(encoding='utf-8'))
    _, _, _, handoff = export_inputs(config, data, manifest, legacy_subset=True)
    if handoff and (report.get('manifest_sha256') != hashlib.sha256((output / 'manifest.json').read_bytes()).hexdigest()
                    or report.get('device') != manifest['device']
                    or [r['api'] for r in report['results']] != [e['api'] for e in manifest['results']]):
        raise ValueError('Returned report does not match the exported API scope and files')
    lookup = {e['api']: e for e in data['results']}
    run = Path(config['run_dir']).resolve()
    relative_report = (output / 'evidence/report.json').resolve().relative_to(run).as_posix()
    for result in report['results']:
        entry = lookup[result['api']]
        if entry['test_files'] != next(e['test_files'] for e in manifest['results'] if e['api'] == entry['api']):
            raise ValueError('Current task file scope changed')
        entry.setdefault('device_validation', {})[manifest['device']] = dict(result, report=relative_report, environment=report.get('environment', {}))
        # A remote reference device does not establish NPU support.
        if not handoff and entry.get('execution', 'not_run') not in {'pass', 'fail'}:
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
    elif args.action == 'run' and config.get('source_python'):
        manifest = json.loads((output / 'manifest.json').read_bytes())
        export_inputs(config, json.loads(Path(config['results']).read_bytes()), manifest)
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
