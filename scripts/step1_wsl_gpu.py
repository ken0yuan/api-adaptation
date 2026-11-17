#!/usr/bin/env python3
"""Submit an exported Step1 bundle from a GitHub-enabled workstation; return GPU evidence."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import xml.etree.ElementTree as ET
import zipfile

from step1_remote import export_inputs, prepare, write_json

REPO = 'the-fall-moon/git_test_demo'
WORKFLOW = 'wsl-gpu-tests.yml'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def request(config_path, output):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    output.relative_to(Path(config['run_dir']).resolve())
    if config.get('device') != 'cuda':
        raise ValueError('Step1 workstation requests require CUDA validation')
    if not output.exists():
        prepare(config, output)
    raw = (output / 'manifest.json').read_bytes()
    manifest, bundled_raw, _ = bundle_files(output / 'bundle.zip')
    if raw != bundled_raw or manifest['torch_index'] != config['torch_index']:
        raise ValueError('Existing export does not match this request')
    current = json.loads(Path(config['results']).read_text(encoding='utf-8'))
    export_inputs(config, current, manifest)
    path = output / 'step1-request.json'
    if path.exists():
        ticket = json.loads(path.read_text(encoding='utf-8'))
        if ticket['manifest_sha256'] != digest(raw):
            raise ValueError('Existing request belongs to another export')
        return ticket
    ticket = {'schema_version': 1, 'type': 'step1_wsl_gpu', 'status': 'needs_workstation',
              'request_path': str(path), 'manifest_sha256': digest(raw),
              'request_id': output.name + '-' + digest(raw)[:12], 'branch': 'check/' + output.name,
              'source': {'python': sys.executable, 'skill_dir': str(Path(__file__).resolve().parents[1]),
                         'config': str(config_path), 'output': str(output)},
              'apis': [e['api'] for e in manifest['results']],
              'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_json(path, ticket)
    return ticket


def gh(*args, payload=None):
    command = ['gh', *args]
    if payload is not None:
        command += ['--input', '-']
    proc = subprocess.run(command, input=json.dumps(payload) if payload is not None else None,
                          capture_output=True, text=True, encoding='utf-8')
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return proc.stdout


def api(path, payload=None):
    return json.loads(gh('api', '--hostname', 'github.com', '-H', 'X-GitHub-Api-Version: 2026-03-10',
                         'repos/' + REPO + ('/' + path if path else ''),
                         *(['--method', 'POST'] if payload is not None else []), payload=payload))


def bundle_files(bundle):
    with zipfile.ZipFile(bundle) as archive:
        raw = archive.read('manifest.json')
        manifest = json.loads(raw)
        if manifest['device'] != 'cuda':
            raise ValueError('WSL requires a CUDA bundle')
        if manifest['python_minor'] not in [[3, 10], [3, 11], [3, 12], [3, 13]]:
            raise ValueError('Unsupported source Python version')
        channel = manifest['torch_index'].rstrip('/').rsplit('/', 1)[-1]
        if channel not in {'cu132', 'cu134'} or manifest.get('torch_requirement', 'torch') != 'torch':
            raise ValueError('WSL workflow supports latest cu132/cu134 nightly only')
        files = {}
        for name, sha in manifest['files'].items():
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or '\\' in name or not name.startswith('test/'):
                raise ValueError('WSL bundle paths must remain under test/: ' + name)
            data = archive.read('tests/' + name)
            if digest(data) != sha:
                raise ValueError('Bundle hash mismatch: ' + name)
            files[name] = data
        for entry in manifest['results']:
            if not entry['test_files'] or any(p not in files for p in entry['test_files']):
                raise ValueError('Missing API test files')
            if any(not (PurePosixPath(p).name.startswith('test_') or PurePosixPath(p).name.endswith('_test.py')) for p in entry['test_files']):
                raise ValueError('Test file cannot be selected by WSL workflow')
        for name in ['step1_gpu_audit.py', 'pytest.ini', 'step1-manifest.json']:
            if name in files:
                raise ValueError('Bundle collides with relay metadata')
        files['requirements.txt'] = archive.read('requirements.txt')
    files['step1-manifest.json'] = raw
    files['step1_gpu_audit.py'] = Path(__file__).with_name('step1_gpu_audit.py').read_bytes()
    files['pytest.ini'] = b'[pytest]\naddopts = -p step1_gpu_audit\n'
    return manifest, raw, files


def submit(bundle, output, branch):
    if output.exists():
        raise ValueError('Output already exists; use fetch, not another dispatch')
    if not re.fullmatch(r'check/[A-Za-z0-9][A-Za-z0-9._/-]*', branch) or '..' in branch:
        raise ValueError('Use a new check/... branch')
    manifest, raw, files = bundle_files(bundle)
    output.mkdir(parents=True)
    (output / 'manifest.json').write_bytes(raw)
    state = {'repository': REPO, 'workflow': WORKFLOW, 'branch': branch,
             'python_version': '.'.join(map(str, manifest['python_minor'])),
             'cuda_build': manifest['torch_index'].rsplit('/', 1)[-1], 'phase': 'preparing'}
    write_json(output / 'submission.json', state)
    base = api('')
    base_sha = api('git/ref/heads/' + base['default_branch'])['object']['sha']
    tree = api('git/commits/' + base_sha)['tree']['sha']
    base_files = api('git/trees/' + tree + '?recursive=1')
    if base_files.get('truncated'):
        raise ValueError('Base tree truncated; cannot verify selected tests')
    lookup = {x['path']: x['sha'] for x in base_files['tree'] if x['type'] == 'blob'}
    changed = set()
    entries = []
    import base64
    for name, data in files.items():
        sha = api('git/blobs', {'content': base64.b64encode(data).decode(), 'encoding': 'base64'})['sha']
        entries.append({'path': name, 'mode': '100644', 'type': 'blob', 'sha': sha})
        if lookup.get(name) != sha:
            changed.add(name)
    expected = {p for e in manifest['results'] for p in e['test_files']}
    if not expected.issubset(changed):
        raise ValueError('Tests unchanged from base would not be selected: ' + ', '.join(sorted(expected - changed)))
    tree_sha = api('git/trees', {'base_tree': tree, 'tree': entries})['sha']
    sha = api('git/commits', {'message': 'Validate exported Step1 tests: ' + branch,
                             'tree': tree_sha, 'parents': [base_sha]})['sha']
    state.update(head_sha=sha, phase='committed')
    write_json(output / 'submission.json', state)
    api('git/refs', {'ref': 'refs/heads/' + branch, 'sha': sha})
    # Persist intent before dispatch: an ambiguous network error must not cause a duplicate run.
    state.update(phase='dispatch_requested', requested_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_json(output / 'submission.json', state)
    response = api('actions/workflows/' + WORKFLOW + '/dispatches', {
        'ref': branch, 'inputs': {'python_version': state['python_version'], 'cuda_build': state['cuda_build']}})
    if not isinstance(response, dict) or not response.get('workflow_run_id'):
        raise RuntimeError('Dispatch returned no Run ID; use fetch to recover, do not resubmit')
    state.update(run_id=response['workflow_run_id'], phase='dispatched')
    write_json(output / 'submission.json', state)
    return state


def analyze(manifest_bytes, state, run, artifact):
    manifest = json.loads(manifest_bytes)
    if run['head_sha'] != state['head_sha'] or run['head_branch'] != state['branch'] or run['event'] != 'workflow_dispatch':
        raise ValueError('Run branch/SHA/event does not match submission')
    artifact = Path(artifact)
    report = {'device': 'cuda', 'manifest_sha256': digest(manifest_bytes),
              'run_id': run['id'], 'run_attempt': run['run_attempt'], 'url': run['html_url'],
              'branch': state['branch'], 'head_sha': state['head_sha'],
              'workflow_conclusion': run.get('conclusion'), 'environment': {}, 'results': []}
    audit, selected, reason = {}, set(), ''
    try:
        if (artifact / 'commit.txt').read_text().strip() != state['head_sha']:
            raise ValueError('Artifact commit does not match submitted SHA')
        selected = set((artifact / 'selected-files.txt').read_text().splitlines())
        with tarfile.open(artifact / 'test-output.tar.gz', 'r:gz') as archive:
            members = [m for m in archive.getmembers() if m.name.removeprefix('./') == 'step1-audit.json' and m.isfile()]
            if len(members) != 1:
                raise ValueError('Missing unique device audit')
            audit = json.loads(archive.extractfile(members[0]).read())
        if audit['manifest_sha256'] != digest(manifest_bytes):
            raise ValueError('Device audit belongs to a different bundle')
        env = audit['environment']
        requires_cuda = any(entry.get('device_independent') is not True for entry in manifest['results'])
        if audit['run_flag'] != '1' or not env['cuda_build'] or '.dev' not in env['torch']:
            raise ValueError('CUDA/nightly/run-flag audit failed')
        if requires_cuda and (not env['cuda_available'] or not env.get('gpu_name')):
            raise ValueError('Required CUDA device is unavailable')
        if 'cu' + env['cuda_build'].replace('.', '') != state['cuda_build']:
            raise ValueError('CUDA channel mismatch')
        report['environment'] = env
        # Retain JUnit as independent raw evidence and require it to parse.
        ET.parse(artifact / 'junit.xml')
    except (OSError, ValueError, KeyError, tarfile.TarError, ET.ParseError) as error:
        reason = str(error)
    for entry in manifest['results']:
        result = {'api': entry['api'], 'execution': 'blocked', 'files': []}
        device_independent = entry.get('device_independent') is True
        if device_independent:
            result['device_independent'] = True
        for file in entry['test_files']:
            item = {'file': file, 'sha256': manifest['files'][file], 'execution': 'blocked',
                    'counts': {'passed': 0, 'failed': 0, 'skipped': 0}}
            info = audit.get('files', {}).get(file, {})
            if (reason or file not in selected or info.get('sha256') != item['sha256']
                    or not device_independent and info.get('selected_device') != 'cuda'
                    or not info.get('collected')):
                item['reason'] = reason or 'Missing selection, collected tests, matching source hash or required CUDA evidence'
            else:
                records = {}
                for record in audit.get('reports', []):
                    if record['nodeid'].split('::')[0] == file:
                        records.setdefault(record['nodeid'], []).append(record)
                incomplete = len(records) != info['collected']
                for phases in records.values():
                    phase_names = {r['when'] for r in phases}
                    if any(r['outcome'] == 'failed' for r in phases):
                        item['counts']['failed'] += 1
                    elif any(r['outcome'] == 'skipped' or r['wasxfail'] for r in phases):
                        item['counts']['skipped'] += 1
                    elif phase_names == {'setup', 'call', 'teardown'} and all(r['outcome'] == 'passed' for r in phases):
                        item['counts']['passed'] += 1
                    else:
                        incomplete = True
                counts = item['counts']
                item['execution'] = ('fail' if counts['failed'] else 'partial' if incomplete or counts['skipped'] else
                                     'pass' if counts['passed'] else 'blocked')
                item['selected_device'] = info.get('selected_device')
            result['files'].append(item)
        states = [f['execution'] for f in result['files']]
        result['execution'] = 'fail' if 'fail' in states else 'blocked' if 'blocked' in states else 'partial' if 'partial' in states else 'pass'
        result['counts'] = {k: sum(f['counts'][k] for f in result['files']) for k in ['passed', 'failed', 'skipped']}
        report['results'].append(result)
    states = [e['execution'] for e in report['results']]
    report['execution'] = 'fail' if 'fail' in states else 'blocked' if 'blocked' in states else 'partial' if 'partial' in states else 'pass'
    if reason:
        report['reason'] = reason
    return report


def fetch(output, wait_seconds):
    state = json.loads((output / 'submission.json').read_text(encoding='utf-8'))
    if not state.get('run_id'):
        runs = api('actions/workflows/' + WORKFLOW + '/runs?event=workflow_dispatch&branch=' + state['branch'])['workflow_runs']
        matches = [r for r in runs if r['head_sha'] == state['head_sha'] and r['created_at'] >= state['requested_at'][:19] + 'Z']
        if len(matches) != 1:
            raise RuntimeError('Cannot uniquely recover Run ID; do not dispatch again')
        state['run_id'] = matches[0]['id']
        write_json(output / 'submission.json', state)
    deadline = None if wait_seconds is None else time.monotonic() + wait_seconds
    while True:
        run = api('actions/runs/' + str(state['run_id']))
        write_json(output / 'run.json', run)
        print(json.dumps({k: run.get(k) for k in ['id', 'status', 'conclusion', 'run_attempt']}, ensure_ascii=False), flush=True)
        if run['status'] == 'completed':
            break
        if deadline is not None and time.monotonic() >= deadline:
            return None
        time.sleep(10 if deadline is None else min(10, max(0, deadline - time.monotonic())))
    evidence = output / 'evidence'
    evidence.mkdir(exist_ok=True)
    artifact = evidence / ('artifacts-' + str(run['run_attempt']))
    if not artifact.exists():
        download = Path(tempfile.mkdtemp(prefix='download-', dir=evidence))
        try:
            gh('run', 'download', str(run['id']), '-R', REPO, '-n',
               'gpu-results-%s-%s' % (run['id'], run['run_attempt']), '-D', str(download))
            download.rename(artifact)
        except RuntimeError as error:
            (evidence / 'download-error.log').write_text(str(error), encoding='utf-8')
    if run.get('conclusion') != 'success':
        try:
            (evidence / 'failed.log').write_text(gh('run', 'view', str(run['id']), '-R', REPO, '--attempt', str(run['run_attempt']), '--log-failed'), encoding='utf-8')
        except RuntimeError as error:
            (evidence / 'failed-log-error.log').write_text(str(error), encoding='utf-8')
    report = analyze((output / 'manifest.json').read_bytes(), state, run, artifact)
    write_json(evidence / 'report.json', report)
    return report


def merge(config, output):
    from npu_handoff import record_lock
    path = Path(config['results']).resolve()
    path.relative_to(Path(config['run_dir']).resolve())
    with record_lock(path):
        return _merge(config, output, path)


def _merge(config, output, path):
    raw = (output / 'manifest.json').read_bytes()
    manifest = json.loads(raw)
    report_path = output / 'evidence/report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if report['manifest_sha256'] != digest(raw) or report['device'] != 'cuda':
        raise ValueError('Returned report manifest/device mismatch')
    state = json.loads((output / 'submission.json').read_text(encoding='utf-8'))
    if report['head_sha'] != state['head_sha'] or report['run_id'] != state['run_id']:
        raise ValueError('Returned report run/SHA mismatch')
    run_dir = Path(config['run_dir']).resolve()
    relative_report = report_path.resolve().relative_to(run_dir).as_posix()
    result = json.loads(path.read_text(encoding='utf-8'))
    export_inputs(config, result, manifest, legacy_subset=True)
    by_api = {e['api']: e for e in result['results']}
    expected = {e['api']: e['test_files'] for e in manifest['results']}
    if [e['api'] for e in report['results']] != list(expected):
        raise ValueError('Returned API scope mismatch')
    for item in report['results']:
        entry = by_api[item['api']]
        if entry['test_files'] != expected[item['api']]:
            raise ValueError('Current task file scope changed')
        if [f['file'] for f in item['files']] != expected[item['api']] or any(f['sha256'] != manifest['files'][f['file']] for f in item['files']):
            raise ValueError('Returned file scope/hash mismatch')
        metadata = {k: report.get(k) for k in ['environment', 'run_id', 'run_attempt', 'url', 'branch', 'head_sha', 'reason']}
        entry.setdefault('device_validation', {})['cuda'] = dict(item, report=relative_report, **metadata)
    backup = output / 'result.before-wsl-merge.json'
    if not backup.exists():
        shutil.copy2(path, backup)
    write_json(path, result)
    return {'status': 'GPU_RESULTS_IMPORTED', 'report': relative_report,
            'results': [{'api': e['api'], 'execution': e['execution']} for e in report['results']]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['request', 'submit', 'fetch', 'import'])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--branch')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--wait-seconds', type=int, default=0)
    parser.add_argument('--watch', action='store_true', help='Poll the existing run until it completes')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == 'request':
        if not args.config:
            parser.error('request requires source --config')
        result = request(args.config, output)
    elif args.action == 'submit':
        if not args.bundle or not args.branch:
            parser.error('submit requires --bundle and --branch')
        result = submit(args.bundle, output, args.branch)
    elif args.action == 'fetch':
        result = fetch(output, None if args.watch else max(0, min(args.wait_seconds, 60)))
        if result is None:
            return 3
    else:
        if not args.config:
            parser.error('import requires source --config')
        result = merge(json.loads(args.config.read_text(encoding='utf-8')), output)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 2 if result.get('execution') in {'fail', 'partial', 'blocked'} else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        print('STEP1_WSL_GPU_ERROR: ' + str(error), file=sys.stderr)
        sys.exit(2)
