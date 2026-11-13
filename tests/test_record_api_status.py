from contextlib import closing
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from record_api_status import record_status


class StatusTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.db = self.root/'api.sqlite3'
        with closing(sqlite3.connect(self.db)) as c, c:
            c.execute('CREATE TABLE "API支持数据" ("API名称" TEXT PRIMARY KEY,"是否对外" TEXT,'
                      '"是否支持" TEXT,"API真正名字" TEXT,"适配策略" TEXT)')
            c.executemany('INSERT INTO "API支持数据" VALUES (?,?,?,?,?)', [
                ('torch.foo','是','未提供',None,'补充testcase'),
                ('torch.alias','是','未提供','torch.foo',None),
                ('torch.bar','是','未提供',None,'补充testcase')])
        self.adapt = self.root/'adapt-information.json'
        self.adapt.write_text(json.dumps({'task-branches':['adapt-torch.foo','adapt-torch.bar'],
                                         'development-branch':'adapt-set-test'}), encoding='utf-8')
        self.results = self.root/'api-loop/result.json'
        self.results.parent.mkdir()
        (self.root/'test.log').write_text('Ran 1 test\nOK\n', encoding='utf-8')
        self.entry = {'api':'torch.alias','database_api':'torch.foo','strategy':'补充testcase',
                      'generation':'generated','execution':'not_run','report':None}
        self.digest = hashlib.sha256(self.db.read_bytes()).hexdigest()

    def run_status(self, execution, entries=None):
        rows = entries or [dict(self.entry, execution=execution,
                               report='test.log' if execution in {'pass','fail'} else None)]
        self.results.write_text(json.dumps({'results':rows}), encoding='utf-8')
        result = record_status(self.adapt, self.db, self.results)
        self.assertEqual(hashlib.sha256(self.db.read_bytes()).hexdigest(), self.digest)
        return result

    def test_generation_initializes_run_status_without_database_write(self):
        result = self.run_status('not_run')
        self.assertEqual(result['api-status'], {'torch.foo':'补充testcase'})
        self.assertEqual(result['next_steps'], {'torch.foo':'step1'})

    def test_pass_finishes_api_and_hands_back_for_pr(self):
        result = self.run_status('pass')
        self.assertEqual(result['api-status'], {'torch.foo':'支持'})
        self.assertEqual(result['next_steps'], {'torch.foo':'step4'})
        self.assertEqual(json.loads(self.adapt.read_text())['development-branch'], 'adapt-set-test')

    def test_fail_dispatches_adaptation_design(self):
        result = self.run_status('fail')
        self.assertEqual(result['api-status'], {'torch.foo':'待适配'})
        self.assertEqual(result['next_steps'], {'torch.foo':'step2'})

    def test_skip_blocked_and_partial_do_not_advance(self):
        for execution in ['skip','blocked','partial']:
            with self.subTest(execution=execution):
                result = self.run_status(execution)
                self.assertEqual(result['api-status']['torch.foo'], '补充testcase')
                self.assertEqual(result['next_steps']['torch.foo'], 'step1')

    def test_mixed_results_record_each_api_independently(self):
        entries = [dict(self.entry,execution='pass',report='test.log'),
                   dict(self.entry,api='torch.bar',database_api='torch.bar',execution='fail',report='test.log')]
        result = self.run_status(None, entries)
        self.assertEqual(result['api-status'], {'torch.foo':'支持','torch.bar':'待适配'})

    def test_no_report_cannot_claim_test_completed(self):
        self.results.write_text(json.dumps({'results':[dict(self.entry,execution='pass')]}),encoding='utf-8')
        before = self.adapt.read_bytes()
        with self.assertRaises(ValueError):
            record_status(self.adapt, self.db, self.results)
        self.assertEqual(self.adapt.read_bytes(), before)

    def test_foreign_result_cannot_change_run(self):
        self.adapt.write_text(json.dumps({'task-branches':['adapt-torch.bar']}),encoding='utf-8')
        self.results.write_text(json.dumps({'results':[self.entry]}),encoding='utf-8')
        before = self.adapt.read_bytes()
        with self.assertRaises(ValueError):
            record_status(self.adapt, self.db, self.results)
        self.assertEqual(self.adapt.read_bytes(), before)

    def test_repeated_result_is_idempotent(self):
        self.run_status('pass')
        before = self.adapt.read_bytes()
        self.run_status('pass')
        self.assertEqual(self.adapt.read_bytes(), before)

    def test_run_progress_overrides_unchanged_database_strategy(self):
        self.run_status('fail')
        result = self.run_status('not_run')
        self.assertEqual(result['api-status']['torch.foo'], '待适配')
        self.assertEqual(result['next_steps']['torch.foo'], 'step2')


if __name__ == '__main__':
    unittest.main()
