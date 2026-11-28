#!/usr/bin/env python3
"""Record this run's API progress without changing the support database."""
from __future__ import annotations

import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import xml.etree.ElementTree as ET

from adaptation_target import run_target, strategy_table
from npu_handoff import entry_for, evidence_path, file_hash, mark_for_repair, record_lock, verify_evidence
from prepare_testcase_handoff import representative
from step1_wsl_gpu import ai_loop_root

INITIAL = {'补充testcase': '补充testcase', '待适配': '待适配', '已支持': '支持'}


def read_object(path):
    result = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(result, dict):
        raise ValueError('JSON must be an object: ' + str(path))
    return result


def write_object(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                     prefix='.' + path.name, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def apply_decision(entry, decision, run):
    action, reason, snapshot = decision
    design = entry.get('adaptation')
    if not design or not reason.strip():
        raise ValueError('需要适配方案和具体复核结论或返工原因')
    if action == 'revise':
        if entry.get('next_step') not in ('step1', 'step2', 'step3', 'step3.1'):
            raise ValueError('当前 API 未在实施或核验中')
        design['revision_reason'] = reason
        design.pop('review', None)
        return
    npu = entry.get('device_validation', {}).get('npu', {})
    if (entry.get('execution') != 'pass' or not snapshot
            or (run / snapshot).resolve() != (run / npu['snapshot']).resolve()):
        raise ValueError('复核结论必须对应当前通过的 NPU 快照')
    review = dict(outcome=action, reason=reason, snapshot=npu['snapshot'])
    if design.get('review') == review:
        return
    if entry.get('next_step') != 'step3.1':
        raise ValueError('当前 API 未进入 Step3.1')
    design['review'] = review


class MissingTests(ValueError):
    pass


def coverage_problem(entry, run):
    """Step 1 judges the semantics. Only use its review for this API and attempt."""
    npu = entry.get('device_validation', {}).get('npu', {})
    if not npu.get('snapshot'):
        return '缺本次 NPU 快照，先恢复或登记本次尝试'
    snapshot = evidence_path(run, npu['snapshot'])
    path = snapshot.with_suffix('.coverage.json')
    if not path.is_file():
        return f'待 Step1 核对功能并保存 {path.relative_to(run)}'
    try:
        review = read_object(path)
    except ValueError as error:
        return f'{path.name}: {error}'
    digest = file_hash(snapshot)
    expected = npu.get('artifacts', {}).get(npu['snapshot'], digest)
    if review.get('api') != entry['api'] or review.get('snapshot_sha256') != digest or expected != digest:
        return '功能核对记录与本次 API 或快照不符'
    features = review.get('features')
    if not isinstance(features, list) or not features:
        return '缺少逐功能核对记录'
    gaps = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        name = feature.get('feature') or '未命名场景'
        if feature.get('outcome') == 'needs_test':
            reason = feature.get('reason') or '按契约补目标调用及可检出错误的断言'
            plan = feature.get('npu_plan')
            gaps.append(f"{name}：{reason}" + (f"（NPU 方案：{plan}）" if plan else ''))
        elif feature.get('outcome') == 'excluded' and not all(
                isinstance(feature.get(k), str) and feature[k].strip() for k in ('source', 'quote', 'reason')):
            gaps.append(f'{name}：排除理由缺可定位的原文或适用条件，补该场景的调用及断言')
    if gaps:
        raise MissingTests('；'.join(gaps))
    tested = False
    executed_methods = {}
    for index, feature in enumerate(features, 1):
        if not isinstance(feature, dict):
            return f'第 {index} 项功能记录格式有误'
        fields = ['feature', 'source']
        outcome = feature.get('outcome')
        if outcome == 'tested':
            fields += ['npu_plan', 'test', 'assertion', 'would_fail_if']
        elif outcome == 'excluded':
            fields += ['quote', 'reason']
        else:
            return f'第 {index} 项尚未测到或未说明排除依据'
        for field in fields:
            if not isinstance(feature.get(field), str) or not feature[field].strip():
                return f'第 {index} 项缺少 {field}'
        if outcome == 'tested':
            filename, _, method = feature['test'].partition('::')
            if filename not in entry.get('test_files', []) or not method.strip():
                return f'第 {index} 项未定位到本次测试文件和方法'
            junit = npu.get('unittest_files', {}).get(filename)
            if junit:
                if '.' not in method:
                    return f'第 {index} 项需按 JUnit 指明类名和方法'
                if filename not in executed_methods:
                    xml = evidence_path(run, junit['xml'])
                    if file_hash(xml) != npu['artifacts'].get(junit['xml']):
                        return f'{filename} 的 JUnit 已变更，请核对本轮材料'
                    executed_methods[filename] = [
                        f"{case.get('classname')}.{case.get('name')}" for case in ET.parse(xml).iter('testcase')
                        if case.find('skipped') is None]
                suffix = '.' + method
                if not any(('.' + name).endswith(suffix) for name in executed_methods[filename]):
                    return f'{filename} 的本轮 JUnit 未执行所列方法 {method}'
            tested = True
    return None if tested else '全部场景均被排除，缺少目标功能的实际测试'


def gpu_request_problem(entry, run):
    cuda = entry.get('device_validation', {}).get('cuda', {})
    if cuda.get('execution', 'not_run') != 'not_run':
        return None
    path = cuda.get('request') or cuda.get('request_path')
    if not (ai_loop_root(run) or path or (run / 'api-loop/wsl-export.json').is_file()
            or str(cuda.get('reason', '')).startswith('awaiting_workstation_gpu')):
        return None  # Standalone runs can submit directly to GitHub.
    if not path:
        return '本次验收还没有对应的 GPU 请求'
    try:
        ticket_path = evidence_path(run, path)
        ticket = read_object(ticket_path)
        manifest_path = ticket_path.with_name('manifest.json')
        sha = file_hash(manifest_path)
        if ticket_path.name != 'step1-request.json' or ticket.get('manifest_sha256') != sha:
            return 'GPU 票据与清单不符'
        if ticket.get('type') != 'step1_wsl_gpu' or ticket.get('status') not in ('needs_workstation', 'running', 'result_received'):
            return 'GPU 票据已失效'
        root = ai_loop_root(ticket_path)
        if root:
            db_path = root / 'data/gpu-requests.sqlite3'
            with closing(sqlite3.connect(db_path.as_uri() + '?mode=ro', uri=True)) as db:
                registered = db.execute('SELECT manifest_sha256 FROM requests WHERE request_path=?',
                                        (ticket_path.relative_to(root).as_posix(),)).fetchone()
            if registered != (sha,):
                return '工作站队列中没有这张 GPU 票据'
        manifest = read_object(manifest_path)
        npu = entry['device_validation']['npu']
        ref = manifest.get('npu_snapshots', {}).get(entry['api'])
        if ref:
            snapshot = evidence_path(run, ref['path'])
            if file_hash(snapshot) != ref['sha256'] or read_object(snapshot)['files'] != npu['handoff_files']:
                return 'GPU 请求对应旧版测试或辅助文件'
        elif len(manifest['results']) > 1:
            return '旧批量票据没有逐项快照，按本次验收重新登记'
        elif manifest['files'] != npu['handoff_files']:
            return 'GPU 请求的测试或辅助文件集合已变化'
        scope = [row['test_files'] for row in manifest['results'] if row['api'] == entry['api']]
        if (scope != [entry['test_files']] or not npu['handoff_files']
                or any(manifest['files'].get(f) != h for f, h in npu['handoff_files'].items())):
            return 'GPU 请求未覆盖当前测试及辅助文件'
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as error:
        return f'无法核对 GPU 请求：{error}'
    return None


def record_status(adapt_information, database, results, *, table=None, api=None, decision=None):
    with record_lock(Path(results).resolve()):
        return _record_status(adapt_information, database, results, table=table, api=api, decision=decision)


def _record_status(adapt_information, database, results, *, table, api, decision):
    adapt_path, results_path = Path(adapt_information).resolve(), Path(results).resolve()
    target = run_target(adapt_path.parent)
    table = strategy_table(table, target[0] if target else None)
    adapt, result = read_object(adapt_path), read_object(results_path)
    if not results_path.is_relative_to(adapt_path.parent):
        raise ValueError('结果文件必须属于本次运行目录')
    entries = result.get('results')
    if not isinstance(entries, list):
        raise ValueError('结果文件缺少results数组')
    if decision:
        apply_decision(entry_for(result, api), decision, adapt_path.parent)
    tasks = adapt.get('task-branches')
    if not isinstance(tasks, list):
        raise ValueError('adapt-information缺少task-branches数组')
    statuses = dict(adapt.get('api-status', {}))
    changes, routes = {}, {}
    pending_adaptations = {}
    pending_coverage = {}
    pending_gpu = {}
    review_holds = {}
    with closing(sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro', uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        allowed = set()
        for task in tasks:
            if task.startswith('adapt-'):
                allowed.add(representative(conn, task[len('adapt-'):], table)['API名称'])
        for entry in entries:
            strategy_table(table, entry.get('classification_source', {}).get('table'))
            row = representative(conn, entry['api'], table)
            name = row['API名称']
            if name not in allowed or entry.get('database_api', name) != name:
                raise ValueError('结果不属于本次任务或正名已变化: ' + entry['api'])
            strategy = row['适配策略']
            if strategy not in INITIAL:
                continue
            status = statuses.get(name, INITIAL[strategy])
            execution = entry.get('execution', 'not_run')
            if execution not in {'pass', 'fail', 'not_run', 'skip', 'blocked', 'partial'}:
                raise ValueError('无效execution: ' + str(execution))
            generation_pending = (entry.get('next_step') == 'step0.1'
                                  and (strategy == '补充testcase' or entry.get('adaptation')))
            needs_review = (execution in ('pass', 'fail') and not generation_pending
                            and not (entry.get('adaptation') or {}).get('revision_reason'))
            if needs_review and (not entry.get('report') or not (adapt_path.parent / entry['report']).is_file()):
                raise ValueError('缺少真实执行报告: ' + name)
            if needs_review or execution in ('partial', 'skip', 'blocked'):
                try:
                    problem = coverage_problem(entry, adapt_path.parent)
                except MissingTests as error:
                    mark_for_repair(entry, adapt_path.parent, str(error))
                    generation_pending = True
                    problem = None
                except (OSError, ValueError, KeyError, ET.ParseError) as error:
                    problem = f'无法核对本轮测试材料：{error}'
                if needs_review and problem:
                    pending_coverage[entry['api']] = problem
                    status = '待适配' if entry.get('adaptation') or strategy == '待适配' else '补充testcase'
                    review_holds.setdefault(name, (status, 'step1'))
                    statuses[name] = changes[name] = entry['status'] = status
                    routes[name] = entry['next_step'] = 'step1'
                    continue
            if needs_review and not generation_pending:
                gpu_issue = gpu_request_problem(entry, adapt_path.parent)
                if gpu_issue:
                    pending_gpu[entry['api']] = gpu_issue
                    devices = entry.setdefault('device_validation', {})
                    devices['cuda'] = dict(execution='not_run', report=None, reason=gpu_issue)
                    status = '待适配' if entry.get('adaptation') or strategy == '待适配' else '补充testcase'
                    review_holds.setdefault(name, (status, 'step1.1'))
                    statuses[name] = changes[name] = entry['status'] = status
                    routes[name] = entry['next_step'] = 'step1.1'
                    continue
            if generation_pending:
                status = '待适配' if strategy == '待适配' else '补充testcase'
                review_holds[name] = (status, 'step0.1')
            elif entry.get('adaptation'):
                if execution != 'not_run' and not entry['adaptation'].get('revision_reason'):
                    verify_evidence(entry, adapt_path.parent)
                design = entry['adaptation']
                npu = entry.get('device_validation', {}).get('npu', {})
                review = design.get('review', {})
                route = 'step3'
                if design.get('revision_reason'):
                    route = 'step2'
                elif execution == 'pass':
                    route = 'step3.1'
                    if review.get('snapshot') == npu['snapshot']:
                        route = 'step4' if review['outcome'] == 'approve' else 'step3'
                status = '支持' if route == 'step4' else '待适配'
                if route != 'step4':
                    pending_adaptations[name] = route
            elif execution in {'pass', 'fail'}:
                status = '支持' if execution == 'pass' else '待适配'
            elif strategy == '补充testcase' and status == '支持':
                status = '补充testcase'
            statuses[name] = status
            entry['status'] = status
            if generation_pending:
                entry['next_step'] = 'step0.1'
            elif entry.get('adaptation'):
                entry['next_step'] = route
            elif execution in {'skip', 'blocked', 'partial'}:
                entry['next_step'] = 'step1'
            elif status == '支持':
                entry['next_step'] = 'step4'
            elif status == '待适配':
                entry['next_step'] = 'step2'
            else:
                entry['next_step'] = 'step1' if entry.get('generation') == 'generated' else 'step0.1'
            changes[name], routes[name] = status, entry['next_step']
    for name, route in pending_adaptations.items():
        statuses[name] = changes[name] = '待适配'
        routes[name] = route
    for name, (status, route) in review_holds.items():
        statuses[name] = changes[name] = status
        routes[name] = route
    adapt['api-status'] = statuses
    write_object(results_path, result)
    write_object(adapt_path, adapt)
    response = {'status': 'API_STATUS_RECORDED', 'api-status': changes, 'next_steps': routes,
                'pending_coverage': pending_coverage}
    if pending_gpu:
        response['pending_gpu'] = pending_gpu
    return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapt-information', required=True, type=Path)
    parser.add_argument('--database', required=True, type=Path)
    parser.add_argument('--table', help='Expected table from the current handoff')
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--api', help='本次复核或修订方案的 API 全名')
    parser.add_argument('--decision', choices=('approve', 'rework', 'revise'))
    parser.add_argument('--reason', help='复核依据或需要修订的具体原因')
    parser.add_argument('--snapshot', help='实际复核的本轮 NPU 快照')
    args = parser.parse_args()
    decision = None
    if any((args.api, args.decision, args.reason, args.snapshot)):
        if not (args.api and args.decision and args.reason):
            parser.error('--api、--decision、--reason 必须同时提供')
        if args.decision != 'revise' and not args.snapshot:
            parser.error('复核需要 --snapshot')
        decision = (args.decision, args.reason, args.snapshot)
    try:
        result = record_status(args.adapt_information, args.database, args.results, table=args.table,
                               api=args.api, decision=decision)
    except (ValueError, KeyError, TypeError, OSError, sqlite3.Error) as error:
        print('API_STATUS_FAILED: ' + str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
