from contextlib import closing
import copy
import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from prepare_testcase_handoff import prepare
from apply_testcase_feedback import apply_feedback


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.database = self.root/'api.sqlite3'
        self.source = self.root/'torch_npu'
        test = self.source/'test/api_supplement/torch/det/test_torch_det.py'
        test.parent.mkdir(parents=True)
        test.write_text('import unittest\n')
        with closing(sqlite3.connect(self.database)) as c, c:
            c.execute('CREATE TABLE "API支持数据" ("API名称" TEXT PRIMARY KEY,"是否对外" TEXT,"是否支持" TEXT,'
                      '"API真正名字" TEXT,"适配策略" TEXT,"API适配分类" TEXT,"分类待审核" TEXT,'
                      '"分类" TEXT,"细分模块" TEXT,"api首次出现版本" TEXT)')
            c.execute('INSERT INTO "API支持数据" VALUES (?,?,?,?,?,?,?,?,?,?)',
                      ('torch.det','是','支持',None,'补充testcase','框架类API','否','基础计算','基础计算','2.14'))
            c.execute('INSERT INTO "API支持数据" VALUES (?,?,?,?,?,?,?,?,?,?)',
                      ('torch.long_alias','是','支持','torch.det',None,None,'否',None,None,'2.14'))
        self.feedback = prepare(self.database, 'torch.long_alias')
        self.feedback.update({'generator_category':'计算类','generation':'generated',
                              'test_files':['test/api_supplement/torch/det/test_torch_det.py'],
                              'category_reason':'行列式是矩阵数值计算，应使用计算类规范。'})

    def rows(self):
        with closing(sqlite3.connect(self.database)) as c:
            return c.execute('SELECT * FROM "API支持数据" ORDER BY "API名称"').fetchall()

    def test_feedback_changes_only_representative_type_and_review_flag(self):
        before = self.rows()
        result = apply_feedback(self.database, self.feedback, self.source)
        after = self.rows()
        self.assertEqual(result['classification_update'], 'updated')
        expected = list(before[0]); expected[5:7] = ['算子类API','是']
        self.assertEqual(after[0], tuple(expected))
        self.assertEqual(after[1], before[1])

    def test_repeat_feedback_is_idempotent(self):
        apply_feedback(self.database, self.feedback, self.source)
        before = self.rows()
        self.assertEqual(apply_feedback(self.database, self.feedback, self.source)['classification_update'], 'unchanged')
        self.assertEqual(self.rows(), before)

    def test_framework_feedback_can_correct_an_initial_compute_category(self):
        with closing(sqlite3.connect(self.database)) as c, c:
            c.execute('UPDATE "API支持数据" SET "API适配分类"=? WHERE "API名称"=?', ('算子类API','torch.det'))
        feedback = dict(self.feedback, api_adaptation_type='算子类API', generator_category='框架类', category_reason='测试分类修正示例')
        self.assertEqual(apply_feedback(self.database, feedback, self.source)['api_adaptation_type'], '框架类API')

    def test_distributed_category_keeps_framework_database_type(self):
        feedback = dict(self.feedback, generator_category='分布式类', category_reason='采用分布式专项规范')
        self.assertEqual(apply_feedback(self.database, feedback, self.source)['classification_update'], 'unchanged')
        self.assertEqual(self.rows()[0][6], '否')

    def test_failed_generation_or_missing_file_cannot_update_database(self):
        for change in [{'generation':'failed'}, {'test_files':[]}, {'test_files':['test/api_supplement/missing.py']}]:
            with self.subTest(change=change):
                before = self.rows()
                with self.assertRaises(ValueError):
                    apply_feedback(self.database, dict(self.feedback, **change), self.source)
                self.assertEqual(self.rows(), before)

    def test_wrong_api_or_filename_cannot_update_database(self):
        for name in ['test/api_supplement/torch/abs/test_torch_abs.py',
                     'test/api_supplement/torch/det/test_wrong_name.py']:
            with self.subTest(name=name):
                path = self.source/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('import unittest\n', encoding='utf-8')
                before = self.rows()
                with self.assertRaisesRegex(ValueError, '约定输出不一致'):
                    apply_feedback(self.database, dict(self.feedback, test_files=[name]), self.source)
                self.assertEqual(self.rows(), before)

    def test_empty_or_whitespace_file_cannot_update_database(self):
        path = self.source/self.feedback['test_files'][0]
        for content in ['', ' \n\t\n']:
            with self.subTest(content=content):
                path.write_text(content, encoding='utf-8')
                before = self.rows()
                with self.assertRaisesRegex(ValueError, '生成文件为空'):
                    apply_feedback(self.database, self.feedback, self.source)
                self.assertEqual(self.rows(), before)

    def test_adjustment_without_reason_cannot_update_database(self):
        before = self.rows()
        with self.assertRaises(ValueError):
            apply_feedback(self.database, dict(self.feedback, category_reason=''), self.source)
        self.assertEqual(self.rows(), before)

    def test_offline_representative_cannot_receive_feedback(self):
        with closing(sqlite3.connect(self.database)) as c, c:
            c.execute('UPDATE "API支持数据" SET "是否支持"=? WHERE "API名称"=?', ('API删除，待下线','torch.det'))
        before = self.rows()
        with self.assertRaises(ValueError):
            apply_feedback(self.database, self.feedback, self.source)
        self.assertEqual(self.rows(), before)

    def test_feedback_for_wrong_representative_is_rejected(self):
        before = self.rows()
        with self.assertRaises(ValueError):
            apply_feedback(self.database, dict(self.feedback, database_api='torch.other'), self.source)
        self.assertEqual(self.rows(), before)

    def test_existing_different_category_is_not_overwritten(self):
        with closing(sqlite3.connect(self.database)) as c, c:
            c.execute('UPDATE "API支持数据" SET "API适配分类"=? WHERE "API名称"=?', ('专项适配API','torch.det'))
        before = self.rows()
        with self.assertRaises(ValueError):
            apply_feedback(self.database, self.feedback, self.source)
        self.assertEqual(self.rows(), before)

    def test_cli_applies_generator_feedback(self):
        path = self.root/'feedback.json'
        path.write_text(json.dumps(self.feedback, ensure_ascii=False), encoding='utf-8')
        result = subprocess.run([sys.executable,'-X','utf8','-B',str(SCRIPTS/'apply_testcase_feedback.py'),
                                 '--database',str(self.database),'--feedback',str(path),
                                 '--torch-npu-source',str(self.source)], capture_output=True,text=True,encoding='utf-8')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['classification_update'], 'updated')


if __name__ == '__main__':
    unittest.main()
