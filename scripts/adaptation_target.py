"""Resolve the AI_LOOP environment that owns an adaptation task."""
import json
from pathlib import Path
import re

MAIN_TABLE = 'API支持数据'
RELEASE = r'[0-9]+\.[0-9]+(?:\.[0-9]+)*(?:\.post[0-9]+)?'


def run_target(run):
    """Return (strategy table, fixed release or None), or None for standalone use."""
    basic = Path(run) / 'basic_information.json'
    if not basic.exists():
        return None
    raw = json.loads(basic.read_text(encoding='utf-8')).get('venv')
    if isinstance(raw, str):
        path = Path(raw)
        if path.is_absolute():
            run = Path(run).resolve()
            if run.parent.name != 'run_data' or run.parent.parent.name != 'data':
                raise ValueError(f'Cannot resolve AI_LOOP root for {basic}')
            path = path.relative_to(run.parent.parent.parent)
        match = re.fullmatch(r'data/venv/(?:torch-(' + RELEASE + r')/)?python-[0-9]+\.[0-9]+(?:\.[0-9]+)?', path.as_posix())
        if match:
            version = match[1]
            if version is None:
                return MAIN_TABLE, None
            minor = re.match(r'[0-9]+\.[0-9]+', version)[0]
            return MAIN_TABLE + '_' + minor, version
    raise ValueError(f'Invalid Step 0 venv in {basic}: {raw!r}')


def strategy_table(table=None, expected=None):
    if table is not None and expected is not None and table != expected:
        raise ValueError(f'Strategy table mismatch: {table} and {expected}')
    selected = expected if table is None else table
    if selected is None:
        return MAIN_TABLE
    if not isinstance(selected, str) or not re.fullmatch(MAIN_TABLE + r'(?:_[0-9]+\.[0-9]+)?', selected):
        raise ValueError(f'Invalid strategy table: {selected!r}')
    return selected


def public_version(version):
    public = version.split('+', 1)[0]
    return re.sub(r'^([0-9]+\.[0-9]+)(?=\.post|$)', r'\1.0', public)


def task_torch_requirement(run):
    """Forward Step 0's active torch pin to the target platform."""
    basic_path = Path(run) / 'basic_information.json'
    if not basic_path.is_file():
        return None
    from pip._internal.network.session import PipSession
    from pip._internal.req.req_file import parse_requirements
    from pip._vendor.packaging.markers import default_environment
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.version import Version

    basic = json.loads(basic_path.read_text(encoding='utf-8'))
    env = default_environment()
    if basic.get('python_version'):
        full = basic['python_version']
        env.update(python_full_version=full, python_version='.'.join(full.split('.')[:2]))
    if basic.get('architecture'):
        env['platform_machine'] = basic['architecture']
    constraints = []
    pins = set()
    path = basic_path.with_name('requirements.txt')
    if not path.is_file():
        raise ValueError(f'Missing Step 0 requirements: {path}')
    with PipSession() as session:
        for entry in parse_requirements(str(path), session=session):
            if not re.match(r'^torch(?=[^A-Za-z0-9._-]|$)', entry.requirement, re.I):
                continue
            req = Requirement(entry.requirement)
            if req.marker and not req.marker.evaluate(env):
                continue
            constraints.append(req.specifier)
            for spec in req.specifier:
                if spec.operator == '==' and '*' not in spec.version:
                    version = Version(spec.version)
                    if not version.is_prerelease:
                        pins.add(version)
    if not pins:
        return 'torch'
    pin = pins.pop()
    if pins or not all(c.contains(pin) for c in constraints):
        raise ValueError(f'Conflicting torch constraints in {path}')
    return 'torch==' + pin.public


def equivalent_requirements(left, right):
    if left == right:
        return True
    if not all(isinstance(req, str) and req.startswith('torch==') for req in (left, right)):
        return False
    return (public_version(left[7:]) == public_version(right[7:])
            and left.partition('+')[2] == right.partition('+')[2])


def gpu_target(config):
    index = config['torch_index'].rstrip('/')
    match = re.fullmatch(r'https://download\.pytorch\.org/whl/(nightly/)?(cpu|cu[0-9]+)', index)
    device = config.get('device', 'cuda')
    if not match or device not in ('cpu', 'cuda') or (device == 'cpu') != (match[2] == 'cpu'):
        raise ValueError('Expected the official PyTorch index for the target device')
    upstream = task_torch_requirement(config['run_dir']) if 'run_dir' in config else None
    configured = config.get('torch_requirement')
    if upstream is not None and configured is not None and not equivalent_requirements(upstream, configured):
        raise ValueError('torch_requirement conflicts with Step 0 requirements.txt')
    req = upstream if upstream is not None else config.get('torch_requirement', 'torch')
    nightly = bool(match[1])
    if nightly:
        if req != 'torch' and not re.fullmatch(r'torch==[0-9][A-Za-z0-9.+_-]*\.dev[0-9]+(?:\+[A-Za-z0-9.]+)?', req):
            raise ValueError('A nightly index requires torch or an exact nightly pin')
    elif not re.fullmatch('torch==' + RELEASE, req):
        raise ValueError('A release index requires torch==<release> without a local build suffix')
    return index, req


def matches_build(actual, requirement, nightly):
    if not isinstance(actual, str) or ('.dev' in actual) != nightly:
        return False
    if requirement == 'torch':
        return nightly
    version = requirement.removeprefix('torch==')
    if '+' in version:
        return actual == version
    return public_version(actual) == public_version(version)
