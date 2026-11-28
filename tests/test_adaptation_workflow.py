from contextlib import closing
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from npu_handoff import load, record, repair, snapshot, start_design
from record_api_status import record_status
from step1_remote import export_inputs, write_json
from test_record_api_status import write_coverage


class AdaptationWorkflowTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.run = Path(temp.name).resolve()
        self.results = self.run / 'api-loop/result.json'
        self.results.parent.mkdir()
        self.source = self.run / 'source'
        self.source.mkdir()
        (self.source / 'test_api.py').write_text('def test_api(): pass')
        self.api = 'torch.foo'
        self.status = self.run / 'adapt-information.json'
        write_json(self.status, {'task-branches': ['adapt-' + self.api]})
        self.db = self.run / 'api.sqlite'
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute('CREATE TABLE "API支持数据" ("API名称" TEXT,"是否对外" TEXT,'
                         '"是否支持" TEXT,"API真正名字" TEXT,"适配策略" TEXT)')
            conn.execute('INSERT INTO "API支持数据" VALUES (?,?,?,?,?)',
                         (self.api, '是', '未提供', None, '待适配'))
        self.design = SimpleNamespace(results=self.results, api=self.api, design='design.md')
        (self.run / self.design.design).write_text('Design v1')

    def route(self):
        record_status(self.status, self.db, self.results)
        return load(self.results)['results'][0]

    def prepare(self, strategy):
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute('UPDATE "API支持数据" SET "适配策略"=?', (strategy,))
        row = dict(api=self.api, strategy=strategy, execution='not_run',
                   generation='not_generated', test_files=[])
        if strategy == '补充testcase':
            row.update(generation='generated', test_files=['test_api.py'])
        write_json(self.results, {'results': [row]})
        if strategy == '补充testcase':
            record(self.attempt('baseline', 'fail', bind_design=False), self.run)
        before = self.route()
        self.assertEqual(before['next_step'], 'step2')
        baseline = start_design(self.design, self.run)
        self.assertEqual(load(self.run / baseline), before)
        self.assertEqual(start_design(self.design, self.run), baseline)
        self.assertEqual(self.route()['next_step'], 'step3')

    def attempt(self, folder, outcome, bind_design=True):
        dest = self.run / folder
        path = dest / 'snap.json'
        snapshot(SimpleNamespace(results=self.results if bind_design else None, source=self.source, api=self.api,
                                 test_file=['test_api.py'], extra_path=[], output=path))
        (dest / 'report.md').write_text('offline report fixture')
        (dest / 'test.log').write_text('offline log fixture')
        write_coverage(path, self.api, 'test_api.py')
        return SimpleNamespace(results=self.results, source=self.source, snapshot=str(path),
                               execution=outcome, report=str(dest / 'report.md'), log=str(dest / 'test.log'))

    def test_direct_adaptation_retries_and_reference_export(self):
        self.prepare('待适配')
        for state in ('blocked', 'fail', 'partial', 'skip', 'pass'):
            args = self.attempt(state, state)
            record(args, self.run)
            record(args, self.run)
            row = self.route()
            self.assertEqual(row['next_step'], 'step3.1' if state == 'pass' else 'step3')
            self.assertEqual(row['status'], '待适配')
        self.assertEqual(row['generation'], 'not_generated')
        entries, _, _, handoff = export_inputs(
            dict(source_root=self.source, run_dir=self.run, apis=[self.api]), load(self.results))
        self.assertTrue(handoff)
        self.assertEqual(entries[0]['api'], self.api)

    def decide(self, action, snap=None):
        record_status(self.status, self.db, self.results, api=self.api,
                      decision=(action, 'fixture review evidence', snap))
        return self.route()

    def test_review_is_bound_to_the_current_snapshot(self):
        self.prepare('待适配')
        first = self.attempt('first', 'pass')
        record(first, self.run)
        self.assertEqual(self.route()['next_step'], 'step3.1')
        self.assertEqual(self.decide('rework', first.snapshot)['next_step'], 'step3')
        second = self.attempt('second', 'pass')
        record(second, self.run)
        self.assertEqual(self.route()['next_step'], 'step3.1')
        with self.assertRaises(ValueError):
            self.decide('approve', first.snapshot)
        self.assertEqual(self.decide('approve', second.snapshot)['next_step'], 'step4')
        self.assertEqual(self.decide('approve', second.snapshot)['status'], '支持')
        record(self.attempt('third', 'pass'), self.run)
        self.assertEqual(self.route()['next_step'], 'step3.1')

    def test_global_redesign_survives_sync_and_late_results(self):
        self.prepare('待适配')
        old = self.attempt('old', 'pass')
        self.assertEqual(self.decide('revise')['next_step'], 'step2')
        with self.assertRaises(ValueError):
            record(old, self.run)
        with self.assertRaises(ValueError):
            start_design(self.design, self.run)
        self.design.design = 'revised.md'
        (self.run / self.design.design).write_text('revised design')
        start_design(self.design, self.run)
        self.assertEqual(self.route()['next_step'], 'step3')

    def test_adaptation_survives_generated_test_repair(self):
        self.prepare('补充testcase')
        record(self.attempt('passed', 'pass'), self.run)
        self.assertEqual(self.route()['next_step'], 'step3.1')
        repair(SimpleNamespace(results=self.results, api=self.api, reason='fix assertion'), self.run)
        self.assertEqual(self.route()['next_step'], 'step0.1')
        record(self.attempt('repaired', 'pass'), self.run)
        self.assertEqual(self.route()['next_step'], 'step3.1')

    def test_direct_design_can_return_to_generator_for_missing_behavior(self):
        from prepare_testcase_handoff import prepare
        self.prepare('待适配')
        previous = self.attempt('first', 'pass')
        record(previous, self.run)
        repair(SimpleNamespace(results=self.results, api=self.api, reason='缺首次调用的断言'), self.run)
        row = self.route()
        self.assertEqual((row['next_step'], row['status']), ('step0.1', '待适配'))
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute('ALTER TABLE "API支持数据" ADD COLUMN "API适配分类" TEXT DEFAULT "框架类API"')
        item = prepare(self.db, self.api, generation_only=False, repair_entry=row)
        self.assertEqual(item['generator_category'], '框架类')
        with self.assertRaisesRegex(ValueError, '早于用例返修'):
            record(previous, self.run)
        record(self.attempt('fixed', 'pass'), self.run)
        self.assertEqual(self.route()['next_step'], 'step3.1')

    def test_step1_failure_is_reused_with_its_original_evidence(self):
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute('UPDATE "API支持数据" SET "适配策略"=?', ('补充testcase',))
        write_json(self.results, {'results': [dict(api=self.api, strategy='补充testcase',
                   generation='generated', test_files=['test_api.py'])]})
        folder = self.run / 'step1'
        snap = folder / 'snapshot.json'
        snapshot(SimpleNamespace(source=self.source, api=self.api, test_file=['test_api.py'],
                                 extra_path=[], output=snap))
        report, log = folder / 'report.md', folder / 'test.log'
        report.write_text('test_api.py::test_api failed')
        log.write_text('original assertion')
        write_coverage(snap, self.api, 'test_api.py')
        record(SimpleNamespace(source=self.source, results=self.results, snapshot=str(snap),
                               report=str(report), log=str(log), execution='fail'), self.run)
        original = self.route()
        log.write_text('changed log')
        with self.assertRaisesRegex(ValueError, '交接材料已修改'):
            start_design(self.design, self.run)
        self.assertEqual(load(self.results)['results'][0], original)
        log.write_text('original assertion')
        archive = start_design(self.design, self.run)
        self.assertEqual(load(self.run / archive), original)
        self.assertEqual(load(self.results)['results'][0]['next_step'], 'step3')

    def test_failed_supplement_rejects_old_snapshot_after_redesign(self):
        self.prepare('补充testcase')
        old = self.attempt('old', 'pass')
        (self.run / self.design.design).write_text('changed in place')
        with self.assertRaisesRegex(ValueError, '方案内容已变化'):
            record(old, self.run)
        (self.run / self.design.design).write_text('Design v1')
        self.design.design = 'v2.md'
        (self.run / self.design.design).write_text('Design v2')
        start_design(self.design, self.run)
        before = self.results.read_bytes()
        with self.assertRaisesRegex(ValueError, '不匹配'):
            record(old, self.run)
        self.assertEqual(self.results.read_bytes(), before)
        record(self.attempt('new', 'pass'), self.run)
        row = self.route()
        self.assertEqual((row['strategy'], row['status']), ('补充testcase', '待适配'))
