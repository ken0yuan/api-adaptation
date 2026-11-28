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


def write_coverage(snapshot, api, filename):
    review = dict(api=api, snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                  features=[dict(feature='写回输入', source='fixture.py:write', outcome='tested',
                                 npu_plan='复用共享写回实现', test=filename + '::T.test_write',
                                 assertion='调用后检查原输入已改变', would_fail_if='只返回副本时原输入不变')])
    path = snapshot.with_suffix('.coverage.json')
    path.write_text(json.dumps(review), encoding='utf-8')
    return path


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

    def run_status(self, execution, entries=None, reviewed=True):
        rows = entries or [dict(self.entry, execution=execution,
                               report='test.log' if execution in {'pass','fail'} else None)]
        for entry in rows:
            if reviewed and entry.get('execution') in ('pass', 'fail'):
                snap = self.root / (entry['api'] + '.snapshot.json')
                snap.write_text(json.dumps({'api': entry['api']}))
                entry.update(test_files=['test.py'], device_validation={'npu': {'snapshot': snap.name}})
                write_coverage(snap, entry['api'], 'test.py')
        self.results.write_text(json.dumps({'results':rows}), encoding='utf-8')
        result = record_status(self.adapt, self.db, self.results)
        self.assertEqual(hashlib.sha256(self.db.read_bytes()).hexdigest(), self.digest)
        return result

    def test_generation_initializes_run_status_without_database_write(self):
        result = self.run_status('not_run')
        self.assertEqual(result['api-status'], {'torch.foo':'补充testcase'})
        self.assertEqual(result['next_steps'], {'torch.foo':'step1'})

    def test_run_venv_selects_strategy_and_rejects_other_table(self):
        (self.root/'basic_information.json').write_text(json.dumps({
            'venv':'data/venv/torch-2.15.1/python-3.11.17'}))
        with closing(sqlite3.connect(self.db)) as c, c:
            c.execute('CREATE TABLE "API支持数据_2.15" AS SELECT * FROM "API支持数据"')
            c.execute('UPDATE "API支持数据_2.15" SET "适配策略"=? WHERE "API名称"=?', ('已支持','torch.foo'))
        self.digest = hashlib.sha256(self.db.read_bytes()).hexdigest()
        result = self.run_status('not_run')
        self.assertEqual(result['api-status'], {'torch.foo':'支持'})
        saved = self.adapt.read_bytes(), self.results.read_bytes()
        with self.assertRaisesRegex(ValueError, 'table mismatch'):
            record_status(self.adapt, self.db, self.results, table='API支持数据')
        self.assertEqual(saved, (self.adapt.read_bytes(), self.results.read_bytes()))

    def test_repaired_helper_requires_a_new_gpu_request(self):
        (self.results.parent / 'wsl-export.json').write_text('{}')
        pending = self.run_status('fail')
        self.assertEqual(pending['next_steps']['torch.foo'], 'step1.1')
        data = json.loads(self.results.read_bytes())
        row = data['results'][0]
        api = row['api']
        files = {'test.py': 'test-hash', 'helper.py': 'old-helper'}
        row['device_validation']['npu']['handoff_files'] = files.copy()
        base = self.root / 'request'
        base.mkdir()
        manifest = {'device': 'cuda', 'files': files, 'results': [{'api': api, 'test_files': ['test.py']}]}
        raw = json.dumps(manifest).encode()
        (base / 'manifest.json').write_bytes(raw)
        (base / 'step1-request.json').write_text(json.dumps({
            'status': 'needs_workstation', 'type': 'step1_wsl_gpu',
            'manifest_sha256': hashlib.sha256(raw).hexdigest()}))
        row['device_validation']['cuda'] = {'execution': 'not_run', 'request': 'request/step1-request.json'}
        self.results.write_text(json.dumps(data))
        valid = record_status(self.adapt, self.db, self.results)
        self.assertEqual(valid['next_steps']['torch.foo'], 'step2')
        row['device_validation']['npu']['handoff_files']['helper.py'] = 'repaired'
        self.results.write_text(json.dumps(data))
        stale = record_status(self.adapt, self.db, self.results)
        self.assertEqual(stale['next_steps']['torch.foo'], 'step1.1')
        self.assertIn(api, stale['pending_gpu'])

    def test_pass_finishes_api_and_hands_back_for_pr(self):
        result = self.run_status('pass')
        self.assertEqual(result['api-status'], {'torch.foo':'支持'})
        self.assertEqual(result['next_steps'], {'torch.foo':'step4'})
        self.assertEqual(json.loads(self.adapt.read_text())['development-branch'], 'adapt-set-test')

    def test_fail_dispatches_adaptation_design(self):
        result = self.run_status('fail')
        self.assertEqual(result['api-status'], {'torch.foo':'待适配'})
        self.assertEqual(result['next_steps'], {'torch.foo':'step2'})

    def test_pass_waits_for_current_review_and_justified_exclusions(self):
        self.assertEqual(self.run_status('pass', reviewed=False)['next_steps']['torch.foo'], 'step1')
        self.run_status('pass')
        path = (self.root / 'torch.alias.snapshot.coverage.json')
        original = json.loads(path.read_text())
        excluded = dict(feature='越界输入', source='fixture.py:write', outcome='excluded',
                        quote='仅接受范围内的输入', reason='该值违反前置条件')
        for invalid in [dict(original, snapshot_sha256='old'), dict(original, api='other'),
                        dict(original, features=[excluded]),
                        dict(original, features=[dict(original['features'][0], test='other.py::test_write')])]:
            path.write_text(json.dumps(invalid))
            result = record_status(self.adapt, self.db, self.results)
            self.assertEqual(result['next_steps']['torch.foo'], 'step1')
            self.assertIn('torch.alias', result['pending_coverage'])
        path.write_text(json.dumps(dict(original, features=original['features'] + [excluded])))
        self.assertEqual(record_status(self.adapt, self.db, self.results)['next_steps']['torch.foo'], 'step4')

    def test_unjustified_exclusion_returns_pass_and_partial_to_generation(self):
        for execution in ('pass', 'partial'):
            self.run_status('pass')
            data = json.loads(self.results.read_text())
            data['results'][0]['execution'] = execution
            self.results.write_text(json.dumps(data))
            path = self.root / 'torch.alias.snapshot.coverage.json'
            review = json.loads(path.read_text())
            review['features'] += [dict(feature='首次写回', outcome='excluded', reason='缺少 CUDA'),
                                   dict(feature='再次调用', outcome='needs_test', reason='原断言取错结果，补查第二次写回')]
            path.write_text(json.dumps(review))
            state = record_status(self.adapt, self.db, self.results)
            self.assertEqual(state['next_steps']['torch.foo'], 'step0.1')
            row = json.loads(self.results.read_text())['results'][0]
            self.assertEqual(row['execution'], 'not_run')
            self.assertIn('首次写回', row['notes'])
            self.assertIn('补查第二次写回', row['notes'])
            archives = list(self.results.parent.glob('repairs/*/previous-result.json'))
            self.assertTrue(any(json.loads(a.read_text())['result'] == data['results'][0] for a in archives))

    def test_skip_blocked_and_partial_do_not_advance(self):
        for execution in ['skip','blocked','partial']:
            with self.subTest(execution=execution):
                result = self.run_status(execution)
                self.assertEqual(result['api-status']['torch.foo'], '补充testcase')
                self.assertEqual(result['next_steps']['torch.foo'], 'step1')

    def test_review_uses_methods_actually_executed_in_this_junit(self):
        self.run_status('pass')
        data = json.loads(self.results.read_text())
        npu = data['results'][0]['device_validation']['npu']
        xml = self.root / 'run.xml'
        npu['unittest_files'] = {'test.py': {'xml': xml.name}}
        for case, expected in [('<testcase classname="T" name="other"/>', 'step1'),
                               ('<testcase classname="T" name="test_write"><skipped/></testcase>', 'step1'),
                               ('<testcase classname="T" name="test_write"/>', 'step4')]:
            xml.write_text(f'<testsuite>{case}</testsuite>')
            npu['artifacts'] = {xml.name: hashlib.sha256(xml.read_bytes()).hexdigest()}
            self.results.write_text(json.dumps(data))
            self.assertEqual(record_status(self.adapt, self.db, self.results)['next_steps']['torch.foo'], expected)

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

    def test_pending_repair_overrides_old_support_and_adaptation_status(self):
        for previous in ['支持', '待适配']:
            with self.subTest(previous=previous):
                self.adapt.write_text(json.dumps({
                    'task-branches': ['adapt-torch.foo', 'adapt-torch.bar'],
                    'api-status': {'torch.foo': previous, 'torch.bar': '支持'}}), encoding='utf-8')
                pending = dict(self.entry, next_step='step0.1', notes='修复方法命名')
                result = self.run_status('not_run', [pending])
                self.assertEqual(result['api-status']['torch.foo'], '补充testcase')
                self.assertEqual(result['next_steps']['torch.foo'], 'step0.1')
                self.assertEqual(json.loads(self.adapt.read_text())['api-status']['torch.bar'], '支持')

    def test_new_attempt_can_complete_after_pending_repair(self):
        self.run_status('pass')
        self.run_status('not_run', [dict(self.entry, next_step='step0.1')])
        result = self.run_status('pass', [dict(self.entry, next_step='step1', execution='pass', report='test.log')])
        self.assertEqual(result['api-status']['torch.foo'], '支持')
        self.assertEqual(result['next_steps']['torch.foo'], 'step4')

    def test_pending_repair_does_not_hide_invalid_execution(self):
        with self.assertRaisesRegex(ValueError, '无效execution'):
            self.run_status('unknown', [dict(self.entry, next_step='step0.1', execution='unknown')])


if __name__ == '__main__':
    unittest.main()
