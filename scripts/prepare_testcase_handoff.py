#!/usr/bin/env python3
"""Read the current API strategy and classification for functional-test generation."""
from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys

from adaptation_target import run_target, strategy_table
CATEGORIES = {'算子类API': '计算类', '框架类API': '框架类',
              '分布式类API': '分布式类', '图模式类API': '图模式类'}
ADAPTATION_TYPES = {category: db_type for db_type, category in CATEGORIES.items()}
GENERATOR_CATEGORIES = tuple(ADAPTATION_TYPES)


def equivalent_names(name):
    yield name
    if name == 'Tensor' or name.startswith('Tensor.'):
        yield 'torch.' + name
    elif name == 'torch.Tensor' or name.startswith('torch.Tensor.'):
        yield name[len('torch.'):]


def available(row):
    if row['是否对外'] != '是':
        raise ValueError(f'{row["API名称"]}: API不是对外入口')
    if row['是否支持'] == 'API删除，待下线':
        raise ValueError(f'{row["API名称"]}: API已从wheel下线，不能生成适配用例')


def representative(conn, api, table):
    row = None
    for name in equivalent_names(api):
        row = conn.execute(f'SELECT * FROM "{table}" WHERE "API名称"=?', (name,)).fetchone()
        if row is not None:
            break
    if row is None:
        raise ValueError(f'{api}: 数据库中没有对应API')
    available(row)
    target = row['API真正名字']
    if target:
        row = conn.execute(f'SELECT * FROM "{table}" WHERE "API名称"=?', (target,)).fetchone()
        if row is None or row['API真正名字']:
            raise ValueError(f'{api}: 正名指向无效，先修正数据库关系')
        available(row)
    return row


def prepare(database, api, task=None, *, category=None, generation_only=True, table=None, strategy_only=False):
    table = strategy_table(table)
    if category is not None and category not in GENERATOR_CATEGORIES:
        raise ValueError('生成器类别须为计算类、框架类、分布式类或图模式类')
    database = Path(database).resolve()
    with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        row = representative(conn, api, table)
        origin = {'database':str(database), 'table':table, 'column':'API适配分类'}
        if strategy_only:
            return {'api':api, 'database_api':row['API名称'], 'strategy':row['适配策略'],
                    'api_adaptation_type':row['API适配分类'], 'classification_source':origin}
        if row['适配策略'] != '补充testcase':
            raise ValueError(f'{api}: 当前适配策略不是补充testcase，返回Step0重新分流')
        adaptation_type = row['API适配分类']
        if not adaptation_type and category is None:
            raise ValueError(f'{api}: API适配分类缺失，交回调用方补齐')
        full_name = row['API名称']
        if full_name == 'Tensor' or full_name.startswith('Tensor.'):
            full_name = 'torch.' + full_name
        generator_category = category if category is not None else CATEGORIES.get(adaptation_type)
        if generator_category is None:
            raise ValueError(f'{api}: {adaptation_type}尚无对应的生成器初分类别')
        return {
            'task': task if task is not None else api,
            'api': api,
            'database_api': row['API名称'],
            'api_full_name': full_name,
            'strategy': row['适配策略'],
            'api_adaptation_type': adaptation_type,
            'generator_category': generator_category,
            'category_is_preliminary': category is None,
            'test_type': 'functional',
            'generation_only': generation_only,
            'classification_source': origin,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', required=True, type=Path)
    parser.add_argument('--table', help='API strategy table supplied by the caller')
    parser.add_argument('--run-dir', type=Path, help='AI_LOOP run directory containing basic_information.json')
    parser.add_argument('--strategy-only', action='store_true', help='Read the strategy before choosing a step')
    parser.add_argument('--api', required=True)
    parser.add_argument('--task')
    parser.add_argument('--category', choices=GENERATOR_CATEGORIES)
    parser.add_argument('--execute-npu', action='store_true')
    args = parser.parse_args()
    try:
        target = run_target(args.run_dir) if args.run_dir else None
        if args.run_dir and target is None:
            raise ValueError('Missing basic_information.json in ' + str(args.run_dir))
        table = strategy_table(args.table, target[0] if target else None)
        result = prepare(args.database, args.api, args.task, category=args.category,
                         generation_only=not args.execute_npu, table=table, strategy_only=args.strategy_only)
    except (ValueError, sqlite3.Error, KeyError, OSError) as error:
        print('GENERATION_INPUT_FAILED: ' + str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
