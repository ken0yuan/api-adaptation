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

from adaptation_target import run_target, strategy_table
from prepare_testcase_handoff import representative

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


def record_status(adapt_information, database, results, *, table=None):
    adapt_path, results_path = Path(adapt_information).resolve(), Path(results).resolve()
    target = run_target(adapt_path.parent)
    table = strategy_table(table, target[0] if target else None)
    adapt, result = read_object(adapt_path), read_object(results_path)
    if not results_path.is_relative_to(adapt_path.parent):
        raise ValueError('结果文件必须属于本次运行目录')
    entries = result.get('results')
    if not isinstance(entries, list):
        raise ValueError('结果文件缺少results数组')
    tasks = adapt.get('task-branches')
    if not isinstance(tasks, list):
        raise ValueError('adapt-information缺少task-branches数组')
    statuses = dict(adapt.get('api-status', {}))
    changes, routes = {}, {}
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
            generation_pending = strategy == '补充testcase' and entry.get('next_step') == 'step0.1'
            if generation_pending:
                status = '补充testcase'
            elif execution in {'pass', 'fail'}:
                report = entry.get('report')
                if not isinstance(report, str) or not report:
                    raise ValueError('测试完成后必须提供真实执行报告: ' + name)
                if not (adapt_path.parent / report).is_file():
                    raise ValueError('执行报告不存在: ' + report)
                status = '支持' if execution == 'pass' else '待适配'
            statuses[name] = status
            entry['status'] = status
            if generation_pending:
                entry['next_step'] = 'step0.1'
            elif execution in {'skip', 'blocked', 'partial'}:
                entry['next_step'] = 'step1'
            elif status == '支持':
                entry['next_step'] = 'step4'
            elif status == '待适配':
                entry['next_step'] = 'step2'
            else:
                entry['next_step'] = 'step1' if entry.get('generation') == 'generated' else 'step0.1'
            changes[name], routes[name] = status, entry['next_step']
    adapt['api-status'] = statuses
    write_object(results_path, result)
    write_object(adapt_path, adapt)
    return {'status': 'API_STATUS_RECORDED', 'api-status': changes, 'next_steps': routes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapt-information', required=True, type=Path)
    parser.add_argument('--database', required=True, type=Path)
    parser.add_argument('--table', help='Expected table from the current handoff')
    parser.add_argument('--results', required=True, type=Path)
    args = parser.parse_args()
    try:
        result = record_status(args.adapt_information, args.database, args.results, table=args.table)
    except (ValueError, KeyError, TypeError, OSError, sqlite3.Error) as error:
        print('API_STATUS_FAILED: ' + str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
