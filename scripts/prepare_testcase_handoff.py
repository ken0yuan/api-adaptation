#!/usr/bin/env python3
"""Read the current API strategy and classification for functional-test generation."""
from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys

TABLE = 'API支持数据'
CATEGORIES = {'算子类API': '计算类', '框架类API': '框架类'}


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


def prepare(database, api, task=None):
    database = Path(database).resolve()
    with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        row = None
        for name in equivalent_names(api):
            row = conn.execute('SELECT * FROM "API支持数据" WHERE "API名称"=?', (name,)).fetchone()
            if row is not None:
                break
        if row is None:
            raise ValueError(f'{api}: 数据库中没有对应API')
        available(row)
        target = row['API真正名字']
        if target:
            row = conn.execute('SELECT * FROM "API支持数据" WHERE "API名称"=?', (target,)).fetchone()
            if row is None or row['API真正名字']:
                raise ValueError(f'{api}: 正名指向无效，先修正数据库关系')
            available(row)
        if row['适配策略'] != '补充testcase':
            raise ValueError(f'{api}: 当前适配策略不是补充testcase，返回Step0重新分流')
        adaptation_type = row['API适配分类']
        if not adaptation_type:
            raise ValueError(f'{api}: API适配分类缺失，交回调用方补齐')
        full_name = row['API名称']
        if full_name == 'Tensor' or full_name.startswith('Tensor.'):
            full_name = 'torch.' + full_name
        category = CATEGORIES.get(adaptation_type)
        if category is None:
            raise ValueError(f'{api}: {adaptation_type}尚无对应的生成器初分类别')
        return {
            'task': task if task is not None else api,
            'api': api,
            'database_api': row['API名称'],
            'api_full_name': full_name,
            'strategy': row['适配策略'],
            'api_adaptation_type': adaptation_type,
            'generator_category': category,
            'category_is_preliminary': True,
            'test_type': 'functional',
            'generation_only': True,
            'classification_source': {'database': str(database), 'table': TABLE,
                                      'column': 'API适配分类'},
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', required=True, type=Path)
    parser.add_argument('--api', required=True)
    parser.add_argument('--task')
    args = parser.parse_args()
    try:
        result = prepare(args.database, args.api, args.task)
    except (ValueError, sqlite3.Error, KeyError) as error:
        print('GENERATION_INPUT_FAILED: ' + str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
