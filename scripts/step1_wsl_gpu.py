#!/usr/bin/env python3
"""Submit an exported Step1 bundle from a GitHub-enabled workstation; return GPU evidence."""
import argparse
from contextlib import closing
import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import time
import xml.etree.ElementTree as ET
import zipfile

from adaptation_target import equivalent_requirements, gpu_target, matches_build
from step1_remote import export_inputs, prepare, write_json

REPO = 'the-fall-moon/git_test_demo'
WORKFLOW = 'wsl-gpu-tests.yml'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def ai_loop_root(path):
    # Walk the components of this exact path, without listing directories.
    for parent in path.parents:
        if parent.parts[-3:] == ('AI_LOOP', 'data', 'run_data'):
            return parent.parent.parent
    return None


def publish_request(path, ticket):
    if path.exists():
        existing = json.loads(path.read_bytes())
        if existing['manifest_sha256'] != ticket['manifest_sha256']:
            raise ValueError('Existing request belongs to another export')
        return existing
    write_json(path, ticket)
    return ticket


def register_path(root, path, sha, ticket=None):
    relative = path.relative_to(root).as_posix()
    with closing(sqlite3.connect(root / 'data/gpu-requests.sqlite3', timeout=20)) as db, db:
        db.execute('CREATE TABLE IF NOT EXISTS requests '
                   '(request_path TEXT PRIMARY KEY, manifest_sha256 TEXT NOT NULL)')
        db.execute('INSERT INTO requests VALUES (?, ?) ON CONFLICT(request_path) DO NOTHING', (relative, sha))
        if path.exists() and json.loads(path.read_bytes()).get('manifest_sha256') != sha:
            raise ValueError('Existing request belongs to another export')
        stored, = db.execute('SELECT manifest_sha256 FROM requests WHERE request_path=?', (relative,)).fetchone()
        if stored != sha:
            raise ValueError('Registered path already belongs to a different manifest: ' + relative)
        if ticket is not None:
            # Persist discoverability first; a crash can never hide a published file.
            db.commit()
            db.execute('BEGIN IMMEDIATE')
            return publish_request(path, ticket)


def register_existing(path):
    path = Path(path).resolve(strict=True)
    root = ai_loop_root(path)
    if root is None or path.name != 'step1-request.json':
        raise ValueError('Expected the exact request file inside AI_LOOP/data/run_data')
    ticket = json.loads(path.read_bytes())
    sha = digest((path.parent / 'manifest.json').read_bytes())
    if (ticket.get('type') != 'step1_wsl_gpu' or ticket.get('schema_version') != 1
            or ticket.get('manifest_sha256') != sha):
        raise ValueError('Invalid GPU request or manifest digest')
    register_path(root, path, sha)
    return ticket


def accepted_rows(config, manifest):
    current = json.loads(Path(config['results']).read_bytes())
    export_inputs(config, current, manifest)
    rows = {row['api']: row for row in current['results']}
    run = Path(config['run_dir']).resolve()
    if manifest.get('npu_snapshots') and ai_loop_root(run):
        from record_api_status import coverage_problem
        for item in manifest['results']:
            row = rows[item['api']]
            issue = coverage_problem(row, run)
            if issue or row.get('next_step') == 'step0.1':
                raise ValueError(f"{row['api']} 尚未通过 Step 1 验收：{issue or '用例正在返工'}")
    return current, rows


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
    ticket = {'schema_version': 1, 'type': 'step1_wsl_gpu', 'status': 'needs_workstation',
              'request_path': str(path), 'manifest_sha256': digest(raw),
              'request_id': output.name + '-' + digest(raw)[:12], 'branch': 'check/' + output.name,
              'source': {'python': sys.executable, 'skill_dir': str(Path(__file__).resolve().parents[1]),
                         'config': str(config_path), 'output': str(output)},
              'apis': [e['api'] for e in manifest['results']],
              'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    root = ai_loop_root(path)
    if root is not None:
        if manifest.get('npu_snapshots'):
            from npu_handoff import record_lock
            results = Path(config['results']).resolve()
            run = Path(config['run_dir']).resolve()
            results.relative_to(run)
            with record_lock(results):
                current, rows = accepted_rows(config, manifest)
                registered = register_path(root, path, digest(raw), ticket)
                for api in ticket['apis']:
                    cuda = rows[api]['device_validation'].setdefault('cuda', {})
                    if cuda.get('execution', 'not_run') == 'not_run':
                        cuda.update(execution='not_run', request=path.relative_to(run).as_posix(),
                                    reason='awaiting_workstation_gpu')
                write_json(results, current)
                return registered
        return register_path(root, path, digest(raw), ticket)
    return publish_request(path, ticket)


def gh(*args, payload=None, timeout=120):
    command = ['gh', *args]
    if payload is not None:
        command += ['--input', '-']
    proc = subprocess.run(command, input=json.dumps(payload) if payload is not None else None,
                          capture_output=True, text=True, encoding='utf-8', timeout=timeout)
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
        _, requirement = gpu_target(manifest)
        if channel not in {'cu132', 'cu134'}:
            raise ValueError('WSL requires cu132 or cu134')
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
        dependencies = archive.read('requirements.txt')
        if requirement != 'torch':
            dependencies = (requirement + '\n').encode() + dependencies
        files['requirements.txt'] = dependencies
    files['step1-manifest.json'] = raw
    files['ci/run-unittests.py'] = Path(__file__).with_name('step1_unittest_audit.py').read_bytes()
    return manifest, raw, files


def submit(bundle, output, branch):
    if output.exists():
        raise ValueError('Output already exists; use fetch, not another dispatch')
    if not re.fullmatch(r'check/[A-Za-z0-9][A-Za-z0-9._/-]*', branch) or '..' in branch:
        raise ValueError('Use a new check/... branch')
    manifest, raw, files = bundle_files(bundle)
    output.mkdir(parents=True)
    (output / 'manifest.json').write_bytes(raw)
    state = {'repository': REPO, 'workflow': WORKFLOW, 'branch': branch, 'driver': 'unittest',
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
    entries = [{'path': 'ci/step1_original_unittests.py', 'mode': '100644', 'type': 'blob',
                'sha': lookup['ci/run-unittests.py']}]
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


def unittest_counts(artifact, receipt, selected):
    """Match each reported method to the workflow selection and raw JUnit."""
    if (not receipt['test_started'] or not receipt['junit_generated'] or receipt['device_kind'] != 'GPU'
            or set(receipt['selected_files']) != selected
            or receipt['test_exit_code'] != int((artifact / 'test-exit-code.txt').read_text())):
        raise ValueError('Unittest receipt does not match the run')
    ids = (artifact / 'selected-test-ids.txt').read_text().splitlines()
    observed = {}
    totals = dict.fromkeys(('tests', 'failures', 'errors', 'skipped'), 0)
    counts = {file: dict(passed=0, failed=0, skipped=0) for file in selected}
    for case in ET.parse(artifact / 'junit.xml').iter('testcase'):
        name = case.attrib['classname'] + '.' + case.attrib['name']
        file = case.get('file')
        if name in observed or file not in selected or not name.startswith(file[:-3].replace('/', '.') + '.'):
            raise ValueError('Unmatched or duplicate unittest case: ' + name)
        observed[name] = file
        tag = next((k for k in ('failure', 'error', 'skipped') if case.find(k) is not None), None)
        counts[file][{'failure': 'failed', 'error': 'failed', 'skipped': 'skipped', None: 'passed'}[tag]] += 1
        totals['tests'] += 1
        if tag:
            totals[{'failure': 'failures', 'error': 'errors', 'skipped': 'skipped'}[tag]] += 1
    if (totals != receipt['junit'] or len(ids) != len(observed) or set(ids) != set(observed)
            or (receipt['test_exit_code'] != 0 and not (totals['errors'] or totals['failures']))):
        raise ValueError('Unittest selection, JUnit totals or exit status disagree')
    return counts


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
    native_counts = None
    receipt_path = artifact / 'execution-result.json'
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
        nightly = '/nightly/' in manifest.get('torch_index', '/nightly/')
        if (audit['run_flag'] != '1' or not env['cuda_build'] or
                not matches_build(env['torch'], manifest.get('torch_requirement', 'torch'), nightly)):
            raise ValueError('CUDA/PyTorch target/run-flag audit failed')
        if requires_cuda and (not env['cuda_available'] or not env.get('gpu_name')):
            raise ValueError('Required CUDA device is unavailable')
        if 'cu' + env['cuda_build'].replace('.', '') != state['cuda_build']:
            raise ValueError('CUDA channel mismatch')
        report['environment'] = env
        # Retain JUnit as independent raw evidence and require it to parse.
        if state.get('driver') == 'unittest' or (artifact / 'unittest-summary.json').is_file():
            receipt = json.loads(receipt_path.read_bytes())
            if (receipt['torch_version'], receipt['cuda_version'], receipt['device_model']) != (
                    env['torch'], env['cuda_build'], env['gpu_name']):
                raise ValueError('Unittest environment and device audit disagree')
            if any(audit['files'].get(f, {}).get('sha256') != h for f, h in manifest['files'].items()):
                raise ValueError('Unittest source hashes differ from the export')
            native_counts = unittest_counts(artifact, receipt, selected)
        else:
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
            collected = sum(native_counts.get(file, {}).values()) if native_counts is not None else info.get('collected')
            if (reason or file not in selected or info.get('sha256') != item['sha256']
                    or not device_independent and info.get('selected_device') != 'cuda'
                    or not collected):
                item['reason'] = reason or 'Missing selection, collected tests, matching source hash or required CUDA evidence'
            elif native_counts is not None:
                counts = item['counts'] = native_counts[file]
                item['execution'] = 'fail' if counts['failed'] else 'partial' if counts['skipped'] else 'pass'
                item['selected_device'] = info.get('selected_device')
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
    if native_counts is not None:
        verified = {f['file'] for item in report['results'] for f in item['files'] if f['execution'] != 'blocked'}
        independent = {f for e in manifest['results'] if e.get('device_independent') for f in e['test_files']}
        write_json(artifact / 'source-files.json', dict(head_sha=state['head_sha'], run_id=run['id'],
                   receipt_sha256=digest(receipt_path.read_bytes()), junit_sha256=digest((artifact / 'junit.xml').read_bytes()),
                   file_hashes=manifest['files'], verified_files=sorted(verified), independent_files=sorted(independent)))
        report['unittest_receipt'] = True
    if reason:
        report['reason'] = reason
    return report


def fetch(output, wait_seconds):
    state = json.loads((output / 'submission.json').read_text(encoding='utf-8'))
    if not state.get('run_id'):
        # A dispatch response can be lost before GitHub indexes the new run.
        until = time.monotonic() + (60 if wait_seconds is None else 0)
        while True:
            runs = api('actions/workflows/' + WORKFLOW + '/runs?event=workflow_dispatch&branch=' + state['branch'])['workflow_runs']
            matches = [r for r in runs if r['head_sha'] == state['head_sha'] and r['created_at'] >= state['requested_at'][:19] + 'Z']
            if len(matches) == 1:
                break
            if matches or time.monotonic() >= until:
                raise RuntimeError('Cannot uniquely recover Run ID; do not dispatch again')
            time.sleep(10)
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
               'gpu-results-%s-%s' % (run['id'], run['run_attempt']), '-D', str(download), timeout=600)
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
    if report.get('unittest_receipt'):
        report_path = report_path.parent / ('artifacts-' + str(report['run_attempt'])) / 'execution-result.json'
        if not report_path.is_file() or not report_path.with_name('source-files.json').is_file():
            raise ValueError('Missing original unittest receipt or source proof')
    relative_report = report_path.resolve().relative_to(run_dir).as_posix()
    result = json.loads(path.read_text(encoding='utf-8'))
    if 'npu_snapshots' not in manifest:
        export_inputs(config, result, manifest, legacy_subset=True)
    elif 'torch_index' in config:
        index, requirement = gpu_target(config)
        if index != manifest['torch_index'] or not equivalent_requirements(requirement, manifest['torch_requirement']):
            raise ValueError('Saved GPU export belongs to a different PyTorch target')
    by_api = {e['api']: e for e in result['results']}
    expected = {e['api']: e['test_files'] for e in manifest['results']}
    if [e['api'] for e in report['results']] != list(expected):
        raise ValueError('Returned API scope mismatch')
    imported = []
    for item in report['results']:
        api = item['api']
        if 'npu_snapshots' in manifest:
            from npu_handoff import evidence_path, verify_evidence
            ref = manifest['npu_snapshots'][api]
            raw_snapshot = evidence_path(run_dir, ref['path']).read_bytes()
            if digest(raw_snapshot) != ref['sha256']:
                raise ValueError('Exported NPU snapshot changed: ' + api)
            original = json.loads(raw_snapshot)
            if original['api'] != api or original['test_files'] != expected[api] or any(
                    manifest['files'].get(name) != sha for name, sha in original['files'].items()):
                raise ValueError('Export does not match the NPU scope: ' + api)
            current = by_api.get(api, {}).get('device_validation', {}).get('npu', {})
            if current.get('handoff_files') != original['files'] or by_api[api]['test_files'] != expected[api]:
                continue
            verify_evidence(by_api[api], run_dir)
        entry = by_api[api]
        if entry['test_files'] != expected[item['api']]:
            raise ValueError('Current task file scope changed')
        if [f['file'] for f in item['files']] != expected[item['api']] or any(f['sha256'] != manifest['files'][f['file']] for f in item['files']):
            raise ValueError('Returned file scope/hash mismatch')
        imported.append({'api': api, 'execution': item['execution']})
        metadata = {k: report.get(k) for k in ['environment', 'run_id', 'run_attempt', 'url', 'branch', 'head_sha', 'reason']}
        entry.setdefault('device_validation', {})['cuda'] = dict(item, report=relative_report, **metadata)
    if not imported:
        return {'status': 'GPU_RESULTS_SUPERSEDED', 'report': relative_report, 'results': []}
    backup = output / 'result.before-wsl-merge.json'
    if not backup.exists():
        shutil.copy2(path, backup)
    write_json(path, result)
    return {'status': 'GPU_RESULTS_IMPORTED', 'report': relative_report,
            'results': imported}


def transient_github_error(error):
    message = str(error).lower()
    return isinstance(error, subprocess.TimeoutExpired) or any(word in message for word in (
        'timeout', 'timed out', 'connection reset', 'connection refused', 'unable to connect',
        'error connecting', 'could not resolve', 'temporary failure', 'network is unreachable',
        'unexpected eof', 'tls handshake', 'http 429', 'http 500', 'http 502', 'http 503', 'http 504'))


def finish_direct(config, direct):
    while True:
        try:
            report = fetch(direct, None)
            artifact = direct / 'evidence' / ('artifacts-' + str(report['run_attempt']))
            error_log = direct / 'evidence/download-error.log'
            if not artifact.exists() and error_log.is_file():
                error = RuntimeError(error_log.read_text(encoding='utf-8'))
                if transient_github_error(error):
                    raise error
            result = merge(config, direct)
            return dict(result, run_id=report['run_id'], execution=report['execution'])
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            if not transient_github_error(error):
                raise
            print('GitHub 暂不可达，10 秒后继续查询同一运行。', file=sys.stderr, flush=True)
            time.sleep(10)


def run_gpu(config_path, output):
    """Choose one route per export and wait for a direct dispatch in this process."""
    from npu_handoff import record_lock
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_bytes())
    run = Path(config['run_dir']).resolve()
    output = Path(output).resolve()
    output.relative_to(run)
    Path(config['results']).resolve().relative_to(run)
    if output == run or config.get('device') != 'cuda':
        raise ValueError('Expected a CUDA export directory inside this batch')
    output.parent.mkdir(parents=True, exist_ok=True)
    # The lock is per export, separate from the short result.json write lock.
    with record_lock(output.parent / (output.name + '.route')):
        if not output.exists():
            prepare(config, output)
        direct = output / 'direct'
        state_path = direct / 'submission.json'
        ticket = output / 'step1-request.json'
        if ticket.exists():
            return request(config_path, output)
        if state_path.exists():
            state = json.loads(state_path.read_bytes())
            if (direct / 'manifest.json').read_bytes() != (output / 'manifest.json').read_bytes():
                raise ValueError('Direct submission belongs to a different export')
            if state.get('run_id') or state['phase'] == 'dispatch_requested':
                return finish_direct(config, direct)
            # submit() durably records dispatch_requested before sending its POST.
            if state['phase'] not in ('preparing', 'committed'):
                raise ValueError('Unknown direct submission phase')
            return request(config_path, output)
        manifest, raw, _ = bundle_files(output / 'bundle.zip')
        if (output / 'manifest.json').read_bytes() != raw:
            raise ValueError('Bundle and export manifest differ')
        accepted_rows(config, manifest)
        try:
            gh('api', '--hostname', 'github.com', f'repos/{REPO}/actions/workflows/{WORKFLOW}',
               '--jq', '.id', timeout=15)
        except (FileNotFoundError, RuntimeError, subprocess.TimeoutExpired) as error:
            print(f'GitHub 工作流探测失败，登记工作站请求：{error}', file=sys.stderr, flush=True)
            return request(config_path, output)
        try:
            submit(output / 'bundle.zip', direct, 'check/direct-' + output.name)
        except (RuntimeError, FileNotFoundError, subprocess.TimeoutExpired) as error:
            state = json.loads(state_path.read_bytes()) if state_path.exists() else {}
            if state.get('phase') != 'dispatch_requested' or re.search(r'HTTP (401|403|404)\b', str(error)):
                print(f'GitHub 未接受直接派发，登记工作站请求：{error}', file=sys.stderr, flush=True)
                return request(config_path, output)
            # The dispatch may have succeeded despite a lost response. Recover it by SHA.
        return finish_direct(config, direct)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'request', 'register', 'submit', 'fetch', 'import'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--request-path', type=Path, help='Exact existing request to register for AI_LOOP')
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--branch')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--wait-seconds', type=int, default=0)
    parser.add_argument('--watch', action='store_true', help='Poll the existing run until it completes')
    args = parser.parse_args()
    if args.action == 'register':
        if not args.request_path:
            parser.error('register requires --request-path')
        register_existing(args.request_path)
        print(args.request_path.resolve())
        return 0
    if not args.output:
        parser.error('this action requires --output')
    output = args.output.resolve()
    if args.action == 'run':
        if not args.config:
            parser.error('run requires source --config')
        result = run_gpu(args.config, output)
    elif args.action == 'request':
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
