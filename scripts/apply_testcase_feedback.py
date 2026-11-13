#!/usr/bin/env python3
"""Apply a generated testcase's category correction to its public representative."""
from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys

from prepare_testcase_handoff import CATEGORIES, prepare

ADAPTATION_TYPES = {'计算类': '算子类API', '框架类': '框架类API', '分布式类': '框架类API'}


def apply_feedback(database, feedback, torch_npu_source):
    category = feedback['generator_category']
    if category not in ADAPTATION_TYPES:
        raise ValueError('生成器必须返回计算类、框架类或分布式类')
    if feedback.get('generation') != 'generated' or not feedback.get('test_files'):
        raise ValueError('仅在对应类别的用例生成成功后回填分类')
    source = Path(torch_npu_source).resolve()
    test_root = source / 'test/api_supplement'
    test_paths = []
    for name in feedback['test_files']:
        path = (source / name).resolve()
        if not path.is_relative_to(test_root) or not path.is_file():
            raise ValueError('生成文件不存在或不属于本次源码的test/api_supplement: ' + name)
        if not path.read_text(encoding='utf-8').strip():
            raise ValueError('生成文件为空: ' + name)
        test_paths.append(path)
    original_type = feedback['api_adaptation_type']
    final_type = ADAPTATION_TYPES[category]
    reason = feedback.get('category_reason', '').strip()
    if category != CATEGORIES.get(original_type) and not reason:
        raise ValueError('调整适配分类时需要生成器说明原因')
    database = Path(database).resolve()
    with closing(sqlite3.connect(database.as_uri() + '?mode=rw', uri=True)) as conn:
        conn.execute('BEGIN IMMEDIATE')
        current = prepare(database, feedback['api'])
        if current['database_api'] != feedback['database_api']:
            raise ValueError('正名关系已变化，请重新读取当前API条目')
        full_name = current['api_full_name']
        expected = (test_root.joinpath(*full_name.split('.')) /
                    ('test_' + full_name.replace('.', '_') + '.py')).resolve()
        if any(path != expected for path in test_paths):
            raise ValueError('生成文件与当前正名API的约定输出不一致: ' +
                             str(expected.relative_to(source)))
        current_type = current['api_adaptation_type']
        if current_type not in {original_type, final_type}:
            raise ValueError('API适配分类已被其他任务修改，请核对后再回填')
        changed = current_type != final_type
        if changed:
            conn.execute('UPDATE "API支持数据" SET "API适配分类"=?,"分类待审核"=? WHERE "API名称"=?',
                         (final_type, '是', current['database_api']))
        conn.commit()
    return {'database_api': current['database_api'], 'initial_api_adaptation_type': original_type,
            'api_adaptation_type': final_type, 'generator_category': category,
            'classification_update': 'updated' if changed else 'unchanged', 'category_reason': reason}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', required=True, type=Path)
    parser.add_argument('--feedback', required=True, type=Path)
    parser.add_argument('--torch-npu-source', required=True, type=Path)
    args = parser.parse_args()
    try:
        feedback = json.loads(args.feedback.read_text(encoding='utf-8'))
        result = apply_feedback(args.database, feedback, args.torch_npu_source)
    except (ValueError, KeyError, TypeError, OSError, sqlite3.Error) as error:
        print('CLASSIFICATION_FEEDBACK_FAILED: ' + str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
