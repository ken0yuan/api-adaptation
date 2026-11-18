#!/usr/bin/env python3
"""Remote half of Step1; only tests and portable dependency pins are received."""
import argparse
import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback
import unittest



def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def command(argv, log, timeout):
    with Path(log).open('a') as stream:
        stream.write(json.dumps({'command': argv}) + '\n')
        stream.flush()
        proc = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == 'posix':
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
            proc.wait()
            return 124


def run_file(file, device, report):
    data = {'file': file, 'execution': 'blocked', 'sha256': hashlib.sha256((Path('tests') / file).read_bytes()).hexdigest()}
    try:
        sys.path.insert(0, str(Path('tests').resolve()))
        spec = importlib.util.spec_from_file_location('api_supplement_under_test', Path('tests') / file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        data['selected_device'] = getattr(module, 'DEVICE_NAME', None)
        if data['selected_device'] != device:
            raise RuntimeError('Test selected %r; expected %r' % (data['selected_device'], device))
        suite = unittest.defaultTestLoader.loadTestsFromModule(module)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        data.update(tests_run=result.testsRun, failures=[{'test': str(t), 'traceback': s} for t, s in result.failures],
                    errors=[{'test': str(t), 'traceback': s} for t, s in result.errors],
                    skipped=[{'test': str(t), 'reason': s} for t, s in result.skipped],
                    expected_failures=[str(t) for t, _ in result.expectedFailures], unexpected_successes=[str(t) for t in result.unexpectedSuccesses])
        data['execution'] = 'fail' if not result.wasSuccessful() else 'partial' if result.testsRun == 0 or result.skipped or result.expectedFailures else 'pass'
    except Exception:
        data['reason'] = traceback.format_exc()
        traceback.print_exc()
    save(report, data)
    return 0 if data['execution'] == 'pass' else 2


def probe(device, report):
    from adaptation_target import matches_build
    import torch
    data = {'python': sys.version, 'executable': sys.executable, 'torch': torch.__version__, 'torch_git_version': torch.version.git_version,
            'torch_file': torch.__file__, 'cuda_build': torch.version.cuda, 'cuda_available': torch.cuda.is_available(), 'device': device}
    if importlib.util.find_spec('torch_npu') is not None:
        raise RuntimeError('Remote environment unexpectedly contains torch_npu')
    manifest = json.loads(Path(__file__).with_name('manifest.json').read_bytes())
    if not matches_build(torch.__version__, manifest.get('torch_requirement', 'torch'), '/nightly/' in manifest['torch_index']):
        raise RuntimeError('Installed torch differs from the requested target')
    if device == 'cuda':
        if not torch.version.cuda or not torch.cuda.is_available():
            raise RuntimeError('CUDA build installed, but CUDA is unavailable')
        tensor = torch.ones(2, device='cuda') * 3
        torch.cuda.synchronize()
        if tensor.sum().item() != 6:
            raise RuntimeError('CUDA computation probe failed')
        data['device_count'] = torch.cuda.device_count()
        data['gpu_name'] = torch.cuda.get_device_name(0)
    elif torch.version.cuda is not None:
        raise RuntimeError('CPU validation requires a CPU wheel')
    data['packages'] = {d.metadata['Name'].lower().replace('_', '-'): d.version for d in importlib.metadata.distributions()}
    save(report, data)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file')
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--report')
    args = parser.parse_args()
    if args.file:
        return run_file(args.file, args.device, args.report)
    if args.probe:
        return probe(args.device, args.report)
    root = Path(__file__).resolve().parent
    os.chdir(root)
    evidence = root / 'evidence'
    evidence.mkdir(exist_ok=True)
    manifest_bytes = (root / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    report = {'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(), 'device': manifest['device'], 'execution': 'blocked',
              'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'results': [{'api': e['api'], 'execution': 'blocked', 'files': []} for e in manifest['results']]}
    try:
        for file, digest in manifest['files'].items():
            if hashlib.sha256((root / 'tests' / file).read_bytes()).hexdigest() != digest:
                raise RuntimeError('Transferred source hash mismatch: ' + file)
        if list(sys.version_info[:2]) != manifest['python_minor']:
            raise RuntimeError('Remote Python major/minor must match source: ' + str(manifest['python_minor']))
        if manifest['device'] == 'cuda':
            if command(['nvidia-smi', '-L'], evidence / 'gpu-preflight.log', 30) != 0:
                raise RuntimeError('No usable NVIDIA GPU; CUDA validation cannot fall back to CPU')
        python = root / 'venv/bin/python'
        install_log = evidence / 'install.log'
        commands = [[sys.executable, '-m', 'venv', str(root / 'venv')],
                    [str(python), '-m', 'pip', 'install', '-r', 'requirements.txt'],
                    [str(python), '-m', 'pip', 'install', '--upgrade', *(['--pre'] if '/nightly/' in manifest['torch_index'] else []), manifest['torch_requirement'], '--index-url', manifest['torch_index'], '-c', 'requirements.txt'],
                    [str(python), '-m', 'pip', 'check']]
        os.environ.update(PYTHONNOUSERSITE='1', TORCH_DEVICE_BACKEND_AUTOLOAD='0', RUN_SUPPLEMENT_API_TESTS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        os.environ.pop('PYTHONPATH', None)
        if manifest['device'] == 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        for argv in commands:
            rc = command(argv, install_log, manifest['install_timeout'])
            if rc:
                raise RuntimeError('Environment preparation failed with exit %s; see install.log' % rc)
        envfile = evidence / 'environment.json'
        if command([str(python), 'worker.py', '--probe', '--device', manifest['device'], '--report', str(envfile)], evidence / 'probe.log', 90):
            raise RuntimeError('Target PyTorch/device probe failed; see probe.log')
        report['environment'] = json.loads(envfile.read_text())
        import re
        actual = {re.sub(r'[-_.]+', '-', k).lower(): v for k, v in report['environment']['packages'].items()}
        drift = {n: {'expected': v, 'actual': actual.get(n)} for n, v in manifest['packages'].items() if actual.get(n) != v}
        if drift:
            save(evidence / 'dependency-drift.json', drift)
            raise RuntimeError('Portable dependency versions changed; see dependency-drift.json')
        for index, entry in enumerate(manifest['results']):
            per_api = report['results'][index]
            for j, file in enumerate(entry['test_files']):
                detail = evidence / ('api-%d-file-%d.json' % (index, j))
                log = evidence / ('api-%d-file-%d.log' % (index, j))
                argv = [str(python), 'worker.py', '--file', file, '--device', manifest['device'], '--report', str(detail)]
                rc = command(argv, log, manifest['timeout_per_file'])
                item = json.loads(detail.read_text()) if detail.exists() else {'file': file, 'execution': 'blocked', 'reason': 'Test timed out or exited without report', 'sha256': manifest['files'][file]}
                item.update(command=argv, returncode=rc, log=log.name)
                if rc and item['execution'] == 'pass':
                    item['execution'] = 'blocked'
                per_api['files'].append(item)
            states = [f['execution'] for f in per_api['files']]
            per_api['execution'] = 'blocked' if 'blocked' in states else 'fail' if 'fail' in states else 'partial' if 'partial' in states else 'pass'
            save(evidence / 'report.json', report)
        states = [e['execution'] for e in report['results']]
        report['execution'] = 'blocked' if 'blocked' in states else 'fail' if 'fail' in states else 'partial' if 'partial' in states else 'pass'
    except Exception as error:
        report['reason'] = str(error)
        for entry in report['results']:
            if not entry['files']:
                entry['reason'] = str(error)
        (evidence / 'bootstrap-error.log').write_text(traceback.format_exc())
    report['completed_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save(evidence / 'report.json', report)
    print(json.dumps({'execution': report['execution'], 'reason': report.get('reason')}), flush=True)
    return 0 if report['execution'] == 'pass' else 2


if __name__ == '__main__':
    sys.exit(main())
