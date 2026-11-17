"""Pytest plugin recording file identity, device selection and test phases."""
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent
MANIFEST_BYTES = (ROOT / 'step1-manifest.json').read_bytes()
MANIFEST = json.loads(MANIFEST_BYTES)
AUDIT = {'manifest_sha256': hashlib.sha256(MANIFEST_BYTES).hexdigest(),
         'run_flag': os.environ.get('RUN_SUPPLEMENT_API_TESTS'), 'files': {}, 'reports': []}


def pytest_collection_modifyitems(items):
    required_cuda = {}
    for entry in MANIFEST['results']:
        for path in entry['test_files']:
            required_cuda[path] = required_cuda.get(path, False) or entry.get('device_independent') is not True
    for item in items:
        relative = Path(item.path).resolve().relative_to(ROOT).as_posix()
        if relative not in required_cuda:
            continue
        info = AUDIT['files'].setdefault(relative, {
            'sha256': hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
            'selected_device': getattr(item.module, 'DEVICE_NAME', None), 'collected': 0})
        info['collected'] += 1
    if AUDIT['run_flag'] != '1':
        raise pytest.UsageError('Step1 requires RUN_SUPPLEMENT_API_TESTS=1')
    for relative, info in AUDIT['files'].items():
        if (info['sha256'] != MANIFEST['files'][relative]
                or required_cuda[relative] and info['selected_device'] != 'cuda'):
            raise pytest.UsageError('Step1 source/device mismatch: ' + relative)


def pytest_runtest_logreport(report):
    AUDIT['reports'].append({'nodeid': report.nodeid, 'when': report.when,
                             'outcome': report.outcome, 'wasxfail': bool(getattr(report, 'wasxfail', False)),
                             'detail': str(report.longrepr) if report.failed or report.skipped else ''})


def pytest_sessionfinish(session, exitstatus):
    import torch
    AUDIT['environment'] = {'python': sys.version, 'torch': str(torch.__version__),
                            'torch_git_version': torch.version.git_version,
                            'cuda_build': torch.version.cuda,
                            'cuda_available': torch.cuda.is_available(),
                            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    AUDIT['exitstatus'] = int(exitstatus)
    output = Path(os.environ['GPU_TEST_OUTPUT_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'step1-audit.json').write_text(json.dumps(AUDIT, indent=2) + '\n', encoding='utf-8')
