import importlib.util
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/prepare_testcase_handoff.py'
spec = importlib.util.spec_from_file_location('handoff', SCRIPT)
handoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handoff)


class HandoffTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.database = Path(temp.name) / 'api.sqlite3'
        with closing(sqlite3.connect(self.database)) as conn, conn:
            conn.execute('CREATE TABLE "API支持数据" ("API名称" TEXT PRIMARY KEY, "是否支持" TEXT, '
                         '"是否对外" TEXT, "适配策略" TEXT, "API适配分类" TEXT, "API真正名字" TEXT)')

    def insert(self, name, kind='算子类API', target=None, public='是', status='支持', strategy='补充testcase'):
        with closing(sqlite3.connect(self.database)) as conn, conn:
            conn.execute('INSERT INTO "API支持数据" VALUES (?,?,?,?,?,?)',
                         (name, status, public, strategy, kind, target))

    def test_compute_handoff_uses_database_type_and_preserves_task(self):
        self.insert('torch.abs')
        before = self.database.read_bytes()
        result = handoff.prepare(self.database, 'torch.abs', 'adapt-torch.abs')
        self.assertEqual(result['api_adaptation_type'], '算子类API')
        self.assertEqual(result['generator_category'], '计算类')
        self.assertEqual(result['task'], 'adapt-torch.abs')
        self.assertEqual(result['test_type'], 'functional')
        self.assertTrue(result['generation_only'])
        self.assertEqual(self.database.read_bytes(), before)

    def test_framework_type_is_used_without_reclassifying_api_semantics(self):
        self.insert('torch.flatten', kind='框架类API')
        self.assertEqual(handoff.prepare(self.database, 'torch.flatten')['generator_category'], '框架类')

    def test_alias_uses_representative_strategy_and_classification(self):
        self.insert('torch.long_name', kind='框架类API', target='torch.a')
        self.insert('torch.a')
        result = handoff.prepare(self.database, 'torch.long_name')
        self.assertEqual(result['api'], 'torch.long_name')
        self.assertEqual(result['database_api'], 'torch.a')
        self.assertEqual(result['api_adaptation_type'], '算子类API')

    def test_tensor_shorthand_keeps_database_key_and_provides_full_entry(self):
        self.insert('Tensor.flatten')
        result = handoff.prepare(self.database, 'torch.Tensor.flatten')
        self.assertEqual(result['database_api'], 'Tensor.flatten')
        self.assertEqual(result['api_full_name'], 'torch.Tensor.flatten')

    def test_distributed_api_receives_initial_database_category_before_generator_review(self):
        self.insert('torch.distributed.all_reduce', kind='框架类API')
        result = handoff.prepare(self.database, 'torch.distributed.all_reduce')
        self.assertEqual(result['api_adaptation_type'], '框架类API')
        self.assertEqual(result['generator_category'], '框架类')
        self.assertTrue(result['category_is_preliminary'])

    def test_missing_type_does_not_default_to_framework(self):
        self.insert('torch.abs', kind=None)
        with self.assertRaisesRegex(ValueError, '分类缺失'):
            handoff.prepare(self.database, 'torch.abs')

    def test_removed_document_does_not_block_wheel_api(self):
        self.insert('torch.abs', status='文档删除，待下线')
        self.assertEqual(handoff.prepare(self.database, 'torch.abs')['api_full_name'], 'torch.abs')

    def test_offline_or_internal_original_name_is_not_rescued_by_alias(self):
        self.insert('torch.a')
        for name, settings in [('torch.offline', {'status':'API删除，待下线'}),
                               ('torch.internal', {'public':'否'})]:
            self.insert(name, target='torch.a', **settings)
            with self.assertRaises(ValueError):
                handoff.prepare(self.database, name)

    def test_unavailable_representative_is_rejected(self):
        self.insert('torch.alias', target='torch.a')
        self.insert('torch.a', status='API删除，待下线')
        with self.assertRaises(ValueError):
            handoff.prepare(self.database, 'torch.alias')

    def test_special_type_needs_an_explicit_generator_route(self):
        self.insert('torch.special', kind='专项适配API')
        with self.assertRaisesRegex(ValueError, '尚无对应'):
            handoff.prepare(self.database, 'torch.special')

    def test_changed_strategy_returns_to_step0(self):
        self.insert('torch.abs', strategy='已支持')
        with self.assertRaisesRegex(ValueError, 'Step0'):
            handoff.prepare(self.database, 'torch.abs')

    def test_cli_emits_generator_input(self):
        self.insert('torch.abs')
        result = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(SCRIPT), '--database', str(self.database),
                                 '--api', 'torch.abs'], capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['generator_category'], '计算类')


if __name__ == '__main__':
    unittest.main()
