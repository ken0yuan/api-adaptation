"""Bind one generator NPU attempt to its test files, report and log."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
import xml.etree.ElementTree as ET

from step1_remote import collect_files, write_json


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(root, snap):
    files = collect_files(root, snap['test_files'], snap['extra_paths'])
    return {name: file_hash(file) for name, file in files.items()}


def snapshot(args):
    if len(set(args.test_file)) != len(args.test_file) or not args.api.strip():
        raise ValueError('需要 API 标识和互不重复的测试文件')
    snap = {'api': args.api, 'test_files': args.test_file, 'extra_paths': args.extra_path,
            'source': str(args.source), 'started_ns': time.time_ns()}
    snap['files'] = source_hashes(args.source, snap)
    if getattr(args, 'results', None):
        results = args.results.resolve()
        row = entry_for(load(results), args.api)
        if not row.get('adaptation') or row.get('next_step') not in ('step0.1', 'step1', 'step3', 'step3.1', 'step4'):
            raise ValueError('实施验证需要已登记方案的 API')
        design = row['adaptation']
        check_design(design, results.parent.parent)
        snap['adaptation_attempt'] = design['attempt']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(snap, output, indent=2, ensure_ascii=False)
    return str(args.output)


def entry_for(data, api):
    rows = [row for row in data['results'] if row['api'] == api]
    if len(rows) != 1:
        raise ValueError(f'本批 result.json 必须包含一条 {api} 记录')
    return rows[0]


def evidence_path(run, name):
    path = (run / name).resolve()
    if not path.is_relative_to(run) or not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f'缺少本次任务的非空交接材料：{name}')
    return path


def verify_evidence(row, run_dir):
    """Used by Step 1 before accepting or exporting a generator result."""
    run = Path(run_dir).resolve()
    npu = row['device_validation']['npu']
    required = {npu['snapshot'], npu['report']}
    for evidence in npu.get('unittest_files', {}).values():
        required.update((evidence['xml'], evidence['receipt']))
    if npu['execution'] != 'blocked' or npu.get('log'):
        required.add(npu['log'])
    if (row.get('execution') != npu['execution'] or row.get('report') != npu['report']
            or required != set(npu['artifacts'])):
        raise ValueError(f"{row['api']} 的 NPU 交接字段不一致")
    for name, expected in npu['artifacts'].items():
        if file_hash(evidence_path(run, name)) != expected:
            raise ValueError(f'NPU 交接材料已修改：{name}')
    snap = load(evidence_path(run, npu['snapshot']))
    if row.get('adaptation'):
        check_design(row['adaptation'], run)
        if snap.get('adaptation_attempt') != row['adaptation']['attempt']:
            raise ValueError('测试证据属于其他适配轮次')
    if (snap['api'] != row['api'] or snap['started_ns'] != npu['started_ns'] or snap['test_files'] != row['test_files']
            or snap['files'] != npu['handoff_files']):
        raise ValueError(f"{row['api']} 的测试范围与执行前快照不一致")
    return snap


@contextmanager
def record_lock(results):
    # Generation can run per API in parallel. Serialize the short result.json update.
    with results.with_suffix('.npu.lock').open('a+b') as lock:
        lock.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def check_design(design, run):
    if file_hash(evidence_path(run, design['design'])) != design['sha256']:
        raise ValueError('方案内容已变化，请保存新版方案并重新交接')


def start_design(args, run):
    path = evidence_path(run, args.design)
    name = path.relative_to(run).as_posix()
    digest = file_hash(path)
    with record_lock(args.results):
        data = load(args.results)
        row = entry_for(data, args.api)
        if row.get('next_step') not in ('step2', 'step3') or row.get('status') != '待适配':
            raise ValueError('仅接收本批待适配并路由到 Step2/3 的 API')
        previous = row.get('adaptation', {})
        if previous.get('design') == name:
            if previous.get('revision_reason'):
                raise ValueError('请登记修订后的新方案文件')
            check_design(previous, run)
            return previous['baseline']
        # Reuse the Step1 failure, checking its recorded evidence before archiving it.
        npu = row.get('device_validation', {}).get('npu', {})
        if not previous and row.get('execution') == 'fail' and ('snapshot' in npu or 'artifacts' in npu):
            verify_evidence(row, run)
        attempt = uuid.uuid4().hex
        archive = run / 'api-loop/adaptation' / attempt / 'before.json'
        archive.parent.mkdir(parents=True)
        write_json(archive, row)
        baseline = archive.relative_to(run).as_posix()
        row['adaptation'] = dict(attempt=attempt, design=name, sha256=digest, baseline=baseline)
        row.update(next_step='step3', execution='not_run', report=None)
        row['device_validation'] = {device: {'execution': 'not_run', 'report': None}
                                    for device in ('npu', 'cuda')}
        write_json(args.results, data)
        return baseline


def mark_for_repair(row, run, reason):
    if not reason.strip():
        raise ValueError('修复交接需要具体问题')
    archive = run / 'api-loop/repairs' / uuid.uuid4().hex / 'previous-result.json'
    archive.parent.mkdir(parents=True)
    write_json(archive, {'reason': reason, 'result': row})
    if row.get('adaptation'):
        row['adaptation'].pop('review', None)
    row.setdefault('device_validation', {}).update(
        npu={'execution': 'not_run', 'report': None},
        cuda={'execution': 'not_run', 'report': None})
    status = '待适配' if row.get('strategy') == '待适配' else '补充testcase'
    row.update(execution='not_run', report=None, status=status, next_step='step0.1',
               repair_requested_ns=time.time_ns(), notes=reason)
    return str(archive)


def repair(args, run):
    with record_lock(args.results):
        data = load(args.results)
        row = entry_for(data, args.api)
        archive = mark_for_repair(row, run, args.reason)
        write_json(args.results, data)
    return archive


def record(args, run):
    snap_path = evidence_path(run, args.snapshot)
    snap = load(snap_path)
    if snap['source'] != str(args.source) or source_hashes(args.source, snap) != snap['files']:
        raise ValueError('源码位置、测试或 helper 与执行前不同，请使用对应尝试的证据')
    paths = {'snapshot': snap_path, 'report': evidence_path(run, args.report)}
    if args.log:
        paths['log'] = evidence_path(run, args.log)
    elif args.execution != 'blocked':
        raise ValueError('已执行的测试需要原始日志 --log')
    if len(set(paths.values())) != len(paths) or any(
            not path.is_relative_to(snap_path.parent) for path in paths.values()):
        raise ValueError('快照、报告及日志应为同一次尝试目录中的不同文件')
    names = {key: p.relative_to(run).as_posix() for key, p in paths.items()}
    npu = dict(names, execution=args.execution, started_ns=snap['started_ns'], handoff_files=snap['files'],
               artifacts={names[key]: file_hash(p) for key, p in paths.items()})
    unittest_files = {}
    for filename in snap['test_files']:
        folder = snap_path.parent / 'unittest' / filename
        receipt = folder / 'execution-result.json'
        if not receipt.is_file():
            if getattr(args, 'require_junit', False) and args.execution in ('pass', 'skip'):
                raise ValueError(f'缺少本次 unittest 回执：{filename}')
            continue
        data = load(receipt)
        if (data.get('file_hashes') != {filename: snap['files'][filename]}
                or data.get('selected_files') != [filename] or data.get('junit_generated') is not True):
            raise ValueError(f'unittest 回执与执行前快照不符：{filename}')
        xml_path = evidence_path(run, folder / 'junit.xml')
        try:
            suites = list(ET.parse(xml_path).iter('testsuite'))
            counts = {k: sum(int(s.attrib[k]) for s in suites)
                      for k in ('tests', 'errors', 'failures', 'skipped')}
        except (ET.ParseError, KeyError, ValueError) as exc:
            raise ValueError(f'无效的 JUnit XML：{filename}') from exc
        if not suites or data.get('junit') != counts:
            raise ValueError(f'JUnit 统计与回执不一致：{filename}')
        if args.execution == 'pass' and (data.get('test_exit_code') != 0 or not counts['tests']
                                         or any(counts[k] for k in ('errors', 'failures', 'skipped'))):
            raise ValueError(f'JUnit 实测结果与 pass 不符：{filename}')
        item = {}
        for key, file in [('xml', folder / 'junit.xml'), ('receipt', receipt)]:
            relative = evidence_path(run, file).relative_to(run).as_posix()
            item[key] = relative
            npu['artifacts'][relative] = file_hash(file)
        unittest_files[filename] = item
    if unittest_files:
        npu['unittest_files'] = unittest_files
    with record_lock(args.results):
        data = load(args.results)
        row = entry_for(data, snap['api'])
        if snap['started_ns'] <= row.get('repair_requested_ns', 0):
            raise ValueError('此快照早于用例返修，请补用例并重新执行')
        design = row.get('adaptation')
        if design and design.get('revision_reason'):
            raise ValueError('当前方案已退回 Step2，请先登记新版方案')
        if design or 'adaptation_attempt' in snap:
            if not design or snap.get('adaptation_attempt') != design['attempt']:
                raise ValueError('测试快照与本次适配方案不匹配')
            check_design(design, run)
        elif (row.get('generation') != 'generated' or row.get('strategy') != '补充testcase'
              or row.get('test_files') != snap['test_files']):
            raise ValueError('快照中的 API 或测试文件与本次生成结果不匹配')
        devices = row.setdefault('device_validation', {})
        previous = devices.get('npu', {})
        if previous.get('snapshot') == npu['snapshot']:
            if previous != npu:
                raise ValueError('同一次尝试已登记，不能替换原报告、日志或执行结论')
            verify_evidence(row, run)
            return names['report']
        if previous.get('started_ns', 0) >= snap['started_ns']:
            raise ValueError('已经登记更新的 NPU 尝试，不能用旧报告覆盖')
        if previous.get('handoff_files') != snap['files']:
            devices['cuda'] = {'execution': 'not_run', 'report': None}
        if design:
            design.pop('review', None)
        devices['npu'] = npu
        row['test_files'] = snap['test_files']
        row.update(execution=args.execution, report=names['report'], next_step='step1')
        write_json(args.results, data)
    return names['report']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    for action in ('snapshot', 'record', 'verify', 'repair', 'design'):
        sub = actions.add_parser(action)
        if action != 'design':
            sub.add_argument('--source', type=Path, required=True)
        if action != 'record':
            sub.add_argument('--api', required=True)
        if action == 'snapshot':
            sub.add_argument('--results', type=Path, help='Step3: bind snapshot to the active design')
            sub.add_argument('--test-file', action='append', required=True)
            sub.add_argument('--extra-path', action='append', default=[])
            sub.add_argument('--output', type=Path, required=True)
        else:
            sub.add_argument('--results', type=Path, required=True)
        if action == 'record':
            sub.add_argument('--require-junit', action='store_true', help='Require per-file XML for completed pass/skip runs')
            sub.add_argument('--snapshot', required=True)
            sub.add_argument('--report', required=True)
            sub.add_argument('--log')
            sub.add_argument('--execution', choices=('pass', 'fail', 'blocked', 'partial', 'skip'), required=True)
        if action == 'repair':
            sub.add_argument('--reason', required=True)
        if action == 'design':
            sub.add_argument('--design', required=True, help='Reviewed design file in this run')
    args = parser.parse_args()
    if args.action != 'design':
        args.source = args.source.resolve()
    try:
        if args.action == 'snapshot':
            print(snapshot(args))
            return
        args.results = args.results.resolve()
        run = args.results.parent.parent
        if args.action == 'design':
            print(start_design(args, run))
        elif args.action == 'record':
            print(record(args, run))
        elif args.action == 'repair':
            print(repair(args, run))
        else:
            row = entry_for(load(args.results), args.api)
            snap = verify_evidence(row, run)
            if source_hashes(args.source, snap) != snap['files']:
                raise ValueError('当前测试或 helper 与 NPU 执行前不同')
            print(json.dumps({'api': args.api, 'execution': row['execution'], 'report': row['report']},
                             ensure_ascii=False))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(1, f'NPU_HANDOFF: {exc}\n')


if __name__ == '__main__':
    main()
