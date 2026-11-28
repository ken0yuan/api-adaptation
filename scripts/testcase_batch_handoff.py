"""Write or verify a generator request against the dispatched AI_LOOP plan item."""
import argparse
import json
from pathlib import Path
import sqlite3
import sys
import tempfile

from adaptation_target import run_target
from prepare_testcase_handoff import full_entry, prepare


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def pending(run, step):
    plan = read_json(run / 'adapt-information.json')['adapt-plan'][step]
    if plan['agentname'] != 'api-adaptation':
        raise ValueError(f'{step}: 不是 api-adaptation 计划项')
    tasks = plan['task-branches']
    if (not isinstance(tasks, list) or any(not isinstance(t, str) or not t.startswith('adapt-') for t in tasks)
            or len(set(tasks)) != len(tasks)):
        raise ValueError(f'{step}: task-branches 格式异常或有重复项')
    results = read_json(run / 'api-loop/result.json')['results']
    rows = {}
    for row in results:
        task = 'adapt-' + row['api']
        if task in tasks:
            if task in rows:
                raise ValueError(f'Step0 重复记录 {task}')
            rows[task] = row
    selected = []
    for task in tasks:
        if task not in rows:
            raise ValueError(f'Step0 未逐名分流：缺少 {task}')
        row = rows[task]
        if row.get('next_step') == 'step0.1':
            selected.append((task, row))
        elif not row.get('next_step') and not row.get('notes'):
            raise ValueError(f'{task}: 缺少 Step0 分流结论')
    return selected


def verify(selected, items):
    if not isinstance(items, list):
        raise ValueError('生成器输入应为逐 API 交接数组')
    expected = [(task, row['api'], full_entry(row['api'])) for task, row in selected]
    received = [(x['task'], x['api'], x['api_full_name']) for x in items]
    if expected != received:
        missing = [item for item in expected if item not in received]
        extra = [item for item in received if item not in expected]
        raise ValueError(f'计划要求 {len(expected)} 项，生成器收到 {len(received)} 项；'
                         f'缺项={missing}，额外或被改名的项={extra}。逐项检查重复和顺序。')
    return [name for _, _, name in received]


def write_request(path, items):
    tmp = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False)
    staging = Path(tmp.name)
    try:
        with tmp:
            json.dump(items, tmp, ensure_ascii=False, indent=2)
            tmp.write('\n')
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--plan-step', required=True)
    parser.add_argument('--database', type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--write', type=Path)
    mode.add_argument('--received', type=Path)
    args = parser.parse_args()
    try:
        selected = pending(args.run_dir, args.plan_step)
        if args.write:
            if not args.database:
                parser.error('--write requires --database')
            target = run_target(args.run_dir)
            if target is None:
                raise ValueError('缺少 basic_information.json，无法确定本次策略表')
            items = []
            for task, row in selected:
                item = prepare(args.database, row['api'], task, table=target[0],
                               category=row.get('generator_category'), generation_only=False, repair_entry=row)
                items.append(item)
        else:
            items = read_json(args.received)
        names = verify(selected, items)
        if args.write:
            write_request(args.write, items)
        print(json.dumps({'count': len(names), 'api_full_names': names}, ensure_ascii=False))
    except (ValueError, OSError, KeyError, TypeError, sqlite3.Error) as error:
        print(f'GENERATION_INPUT_FAILED: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
