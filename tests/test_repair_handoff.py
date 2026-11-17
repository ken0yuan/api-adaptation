"""Offline result-handoff regressions for returning a generated case for repair."""
from contextlib import closing
import copy
from pathlib import Path
from types import SimpleNamespace
import sqlite3
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import npu_handoff
import prepare_testcase_handoff
from step1_remote import export_inputs, write_json


class RepairHandoffTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.api = 'torch.foo'
        self.test_file = 'test/api_supplement/torch/foo/test_torch_foo.py'
        case = self.source / self.test_file
        case.parent.mkdir(parents=True)
        case.write_text('VALUE = 1\n', encoding='utf-8')
        self.results = self.root / 'run/api-loop/result.json'
        self.results.parent.mkdir(parents=True)
        write_json(self.results, {'results': [
            {'api': self.api, 'task': 'adapt-torch.foo', 'database_api': self.api,
             'api_full_name': self.api, 'generator_category': '图模式类',
             'strategy': '补充testcase', 'generation': 'generated',
             'test_files': [self.test_file], 'execution': 'not_run', 'report': None},
            {'api': 'torch.other', 'notes': 'untouched'}]})
        self.record_attempt('first')

    def record_attempt(self, name):
        folder = self.results.parent / name
        snap = folder / 'snapshot.json'
        npu_handoff.snapshot(SimpleNamespace(source=self.source, api=self.api,
                                            test_file=[self.test_file], extra_path=[], output=snap))
        report, log = folder / 'report.md', folder / 'test.log'
        report.write_text('Offline outcome fixture\n', encoding='utf-8')
        log.write_text('Offline log fixture\n', encoding='utf-8')
        self.record_args = SimpleNamespace(
            source=self.source, results=self.results, snapshot=str(snap),
            execution='pass', report=str(report), log=str(log))
        npu_handoff.record(self.record_args, self.results.parent.parent)

    def test_generated_and_completed_results_can_reenter_and_reregister(self):
        initial = npu_handoff.load(self.results)
        initial['results'][0]['device_validation']['cuda'] = {'execution': 'pass', 'report': 'old-gpu.json'}
        for status, execution, next_step in [('支持', 'pass', 'step4'),
                                              ('待适配', 'fail', 'step2'),
                                              ('补充testcase', 'blocked', 'step1')]:
            with self.subTest(status=status):
                data = copy.deepcopy(initial)
                data['results'][0].update(status=status, execution=execution, next_step=next_step)
                write_json(self.results, data)
                archive = npu_handoff.repair(SimpleNamespace(
                    results=self.results, api=self.api, reason='修复测试方法命名'), self.results.parent.parent)
                self.assertEqual(npu_handoff.load(Path(archive))['result'], data['results'][0])
                row, other = npu_handoff.load(self.results)['results']
                self.assertEqual(row['generation'], 'generated')
                self.assertEqual(row['next_step'], 'step0.1')
                self.assertEqual(row['execution'], 'not_run')
                self.assertIsNone(row['report'])
                self.assertEqual(row['generator_category'], '图模式类')
                self.assertEqual(row['notes'], '修复测试方法命名')
                self.assertEqual(row['device_validation']['cuda']['execution'], 'not_run')
                self.assertEqual(other, initial['results'][1])
        self.assertTrue(Path(self.record_args.report).is_file())
        database = self.root / 'api.sqlite3'
        with closing(sqlite3.connect(database)) as conn, conn:
            conn.execute('CREATE TABLE "API支持数据" ("API名称" TEXT, "是否对外" TEXT, '
                         '"是否支持" TEXT, "API真正名字" TEXT, "适配策略" TEXT, "API适配分类" TEXT)')
            conn.execute('INSERT INTO "API支持数据" VALUES (?,?,?,?,?,?)',
                         (self.api, '是', '支持', None, '补充testcase', '框架类API'))
        inputs = prepare_testcase_handoff.prepare(
            database, self.api, category=row['generator_category'], generation_only=False)
        self.assertFalse(inputs['generation_only'])
        self.record_attempt('repaired')
        current = npu_handoff.load(self.results)
        self.assertEqual(current['results'][0]['next_step'], 'step1')
        npu_handoff.verify_evidence(current['results'][0], self.results.parent.parent)
        export_inputs({'source_root': str(self.source), 'apis': [self.api],
                       'run_dir': str(self.results.parent.parent)}, current)

    def test_empty_repair_reason_does_not_change_results(self):
        before = self.results.read_bytes()
        with self.assertRaises(ValueError):
            npu_handoff.repair(SimpleNamespace(results=self.results, api=self.api, reason=' '),
                               self.results.parent.parent)
        self.assertEqual(self.results.read_bytes(), before)

    def test_repair_cli_accepts_a_previously_generated_result(self):
        script = Path(npu_handoff.__file__)
        done = subprocess.run(
            [sys.executable, '-B', str(script), 'repair', '--source', str(self.source),
             '--results', str(self.results), '--api', self.api, '--reason', '修复方法命名'],
            capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertTrue(Path(done.stdout.strip()).is_file())
        self.assertEqual(npu_handoff.load(self.results)['results'][0]['next_step'], 'step0.1')


if __name__ == '__main__':
    unittest.main()
