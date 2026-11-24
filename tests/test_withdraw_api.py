"""Integration checks using only a temporary AI_LOOP batch, SQLite and local Git."""
from contextlib import closing
from pathlib import Path
import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import withdraw_api as w
from dispatch_adaptation_tasks import dispatch
from npu_handoff import verify_evidence
from prepare_adapt_information import prepare
from prepare_development_branch import prepare as prepare_branch
from publish_pr import check_gpu_evidence, owned_status
from reconcile_prs import finish_merged


class WithdrawTests(unittest.TestCase):
    def put(self, p, obj):
        p.parent.mkdir(parents=True, exist_ok=True)
        w.write_json(p, obj)

    def commit(self, msg):
        w.git(self.repo, 'add', '-A')
        w.git(self.repo, '-c', 'user.name=fixture', '-c', 'user.email=test@invalid',
              '-c', 'commit.gpgsign=false', 'commit', '-qm', msg)

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.data = self.root/'data'
        self.run = self.data/'run_data/batch'
        self.loop = self.run/'api-loop'
        self.repo = self.data/'source_code/torch_npu'
        self.repo.mkdir(parents=True)
        w.git(self.repo, 'init', '-q', '-b', 'master')
        (self.repo/'base').write_text('base')
        self.commit('base')
        self.basic = self.run/'basic_information.json'
        self.adapt = self.run/'adapt-information.json'
        self.result = self.loop/'result.json'
        self.db = self.data/'api-support.sqlite3'
        self.mapping = self.data/'branch-mapping.json'
        self.classification = self.data/'api-classification/classification-evidence.jsonl'
        self.classification.parent.mkdir()
        self.classification.write_text('{"api":"torch.a"}\n')
        self.put(self.basic, {'venv':'data/venv/python-3.11','problem-description':'API `torch.alias` and `torch.b`',
                              'torch_npu_latest_source':'data/source_code/torch_npu'})
        with closing(sqlite3.connect(self.db)) as db, db:
            db.execute('CREATE TABLE "API支持数据" ("API名称", "API真正名字", "本地是否适配", "是否对外", "是否支持", "适配策略")')
            db.executemany('INSERT INTO "API支持数据" VALUES (?,?,"是","是","未提供","补充testcase")',
                           [('torch.a',None),('torch.alias','torch.a'),('torch.b',None)])
        self.put(self.mapping, {'nightly':{},'2.15':{'adapt-torch.a':'another-branch'}})
        self.prepare_all()
        self.branch = w.read_object(self.adapt)['development-branch']
        self.rows = []
        helper = self.repo/'test/api_supplement/helper.py'
        helper.parent.mkdir(parents=True)
        helper.write_text('common helper')
        self.helper = str(helper.relative_to(self.repo))
        for suffix in ('a','b'):
            api = 'torch.' + suffix
            test = f'test/api_supplement/torch/{suffix}/test_torch_{suffix}.py'
            path = self.repo/test
            path.parent.mkdir(parents=True)
            path.write_text('def test_contract(): pass\n')
            attempt = self.loop/'npu-attempts'/suffix
            files = {test:w.file_hash(path),self.helper:w.file_hash(helper)}
            self.put(attempt/'snapshot.json',dict(api=api, started_ns=1, files=files, test_files=[test], extra_paths=[self.helper]))
            (attempt/'report.md').write_text('result');(attempt/'test.log').write_text('log')
            evidence = {k:str((attempt/f).relative_to(self.run)) for k,f in
                        [('snapshot','snapshot.json'),('report','report.md'),('log','test.log')]}
            npu = dict(**evidence, execution='pass', started_ns=1, handoff_files=files,
                       artifacts={v:w.file_hash(self.run/v) for v in evidence.values()})
            self.rows.append(dict(api=api,database_api=api,status='支持',report=evidence['report'],
                                  execution='pass',test_files=[test],device_validation={'npu':npu}))
            self.put(self.loop/'handoffs'/(suffix+'.json'),{'api':api})
        gpu = self.loop/'reports/gpu'
        self.manifest = gpu/'manifest.json'
        self.report = gpu/'evidence/report.json'
        merged_files = {k:v for row in self.rows for k,v in row['device_validation']['npu']['handoff_files'].items()}
        self.put(self.manifest,dict(results=[{'api':r['api'],'test_files':r['test_files']} for r in self.rows],files=merged_files))
        gpu_items=[]
        for row in self.rows:
            test=row['test_files'][0]
            item=dict(api=row['api'], execution='pass',files=[dict(file=test,sha256=merged_files[test],execution='pass')])
            gpu_items.append(item)
            row['device_validation']['cuda']=dict(item,report=str(self.report.relative_to(self.run)))
        self.put(self.report,dict(device='cuda',results=gpu_items,manifest_sha256=w.file_hash(self.manifest)))
        self.put(self.result,{'results':self.rows})
        a=w.read_object(self.adapt);a['api-status']=dict.fromkeys(['torch.a','torch.b'],'支持');self.put(self.adapt,a)
        self.commit('batch tests')
        self.db_hash=w.file_hash(self.db)
        self.class_hash=w.file_hash(self.classification)

    def prepare_all(self):
        p=prepare(self.basic,root=self.root)
        b=prepare_branch(self.basic,root=self.root)
        d=dispatch(self.basic,self.adapt,root=self.root)
        return p['status'],b['status'],d['status']

    def checked(self):
        self.assertEqual(w.file_hash(self.db),self.db_hash)
        self.assertEqual(w.file_hash(self.classification),self.class_hash)

    def test_mixed_batch_and_real_resumption(self):
        old_report=self.report.read_bytes()
        ticket=self.manifest.with_name('step1-request.json')
        self.put(ticket,dict(status='needs_workstation',apis=['torch.a','torch.b']))
        preview=w.withdraw(self.run,'torch.alias',dry=True)
        self.assertEqual(preview['remove_tests'],self.rows[0]['test_files'])
        self.assertEqual(len(w.read_object(self.result)['results']),2)
        outcome=w.withdraw(self.run,'torch.alias')
        self.assertEqual(outcome['reexport_apis'],['torch.b'])
        self.assertFalse(ticket.exists())
        self.assertFalse((self.loop/'handoffs/a.json').exists())
        self.assertTrue((self.loop/'handoffs/b.json').is_file())
        self.assertFalse((self.repo/self.rows[0]['test_files'][0]).exists())
        self.assertFalse((self.run/self.rows[0]['report']).exists())
        self.assertTrue((self.repo/self.helper).is_file())
        retained=w.read_object(self.result)['results']
        self.assertEqual(retained,[self.rows[1]])
        verify_evidence(retained[0],self.run)
        self.assertEqual(self.report.read_bytes(),old_report)
        self.commit('withdraw a')
        self.assertEqual(self.prepare_all(),('ADAPT_INFORMATION_READY','BRANCH_READY','DISPATCH_READY'))
        a=w.read_object(self.adapt)
        self.assertEqual(a['task-branches'],['adapt-torch.b'])
        self.assertEqual(a['adapt-plan']['step1']['task-branches'],['adapt-torch.b'])
        self.assertEqual(owned_status(a),{'torch.b':'支持'})
        self.assertEqual(w.read_object(self.mapping),{'nightly':{'adapt-torch.b':self.branch},'2.15':{'adapt-torch.a':'another-branch'}})
        check_gpu_evidence(self.run,self.repo,w.git(self.repo,'rev-parse','HEAD'),self.result,
                           [r['test_files'][0] for r in self.rows],'')
        self.assertEqual(w.withdraw(self.run,'torch.a')['status'],'ALREADY_WITHDRAWN')
        self.checked()

    def test_nightly_input_and_empty_batch(self):
        basic=w.read_object(self.basic);basic['problem-description']='New API needs adaptation';self.put(self.basic,basic)
        added=self.run/'api-refresh/added-apis.json'
        self.put(added,['torch.alias','torch.b'])
        w.withdraw(self.run,'torch.a');w.withdraw(self.run,'torch.b')
        self.assertEqual(self.prepare_all(),('NO_ADAPTATION_REQUIRED','NO_ADAPTATION_REQUIRED','NO_TASKS_TO_DISPATCH'))
        self.assertEqual(json.loads(added.read_bytes()),['torch.alias','torch.b'])
        self.checked()

    def test_other_owner_shared_test_and_running_request(self):
        a=self.rows[0]['test_files'][0]
        self.rows[1]['device_validation']['npu']['handoff_files'][a]='shared'
        self.put(self.result,{'results':self.rows})
        maps=w.read_object(self.mapping);maps['nightly']['adapt-torch.a']='another';self.put(self.mapping,maps)
        ticket=self.manifest.with_name('step1-request.json')
        self.put(ticket,dict(status='running',apis=['torch.a']))
        with self.assertRaisesRegex(ValueError,'active test'):
            w.withdraw(self.run,'torch.a')
        self.assertNotIn('withdrawing-api',w.read_object(self.adapt))
        ticket.unlink()
        w.withdraw(self.run,'torch.a')
        self.assertTrue((self.repo/a).exists())
        self.assertEqual(w.read_object(self.mapping)['nightly']['adapt-torch.a'],'another')
        self.checked()

    def test_recovery_after_partial_write_and_old_pr_receipt(self):
        old=dict(head=self.branch,commit='1'*40,fork='mine/pytorch',repository='Ascend/pytorch',
                 number=1,status='created',**{'api-status':{'torch.alias':'支持','torch.b':'支持'}})
        pr=self.run/'pr.json';self.put(pr,old)
        writer=w.write_json
        def interrupt(p,value):
            if p==self.result:raise OSError('fixture interruption')
            writer(p,value)
        with patch.object(w,'write_json',side_effect=interrupt),self.assertRaisesRegex(OSError,'interruption'):
            w.withdraw(self.run,'torch.a')
        with self.assertRaisesRegex(ValueError,'incomplete'):
            prepare(self.basic,root=self.root)
        w.withdraw(self.run,'torch.a')
        self.assertEqual(w.read_object(pr),old)
        # Invoke only the local receipt consumer, with a fixture remote response.
        remote={'head':{'ref':self.branch,'sha':old['commit'],'repo':{'full_name':old['fork']}},
                'base':{'repo':{'full_name':old['repository']}}}
        with self.assertRaisesRegex(ValueError,'撤项'):
            finish_merged(self.db,self.mapping,pr,old,remote)
        self.checked()

    def test_path_escape_rejected_before_any_edit(self):
        self.rows[0]['test_files']=['../outside.py']
        self.put(self.result,{'results':self.rows})
        before=self.adapt.read_bytes()
        with self.assertRaisesRegex(ValueError,'Invalid relative path'):
            w.withdraw(self.run,'torch.a')
        self.assertEqual(before,self.adapt.read_bytes())
        self.checked()

    def test_cli_retains_changed_source_and_shared_draft(self):
        changed = self.repo / self.rows[0]['test_files'][0]
        changed.write_text('locally edited after the recorded run')
        draft = self.run / 'pr-body.md'
        draft.write_text('torch.a and torch.b')
        argv = [sys.executable, '-B', w.__file__, '--run-dir', str(self.run), '--api', 'torch.a']
        plan = json.loads(subprocess.check_output(argv + ['--dry-run']))
        self.assertIn(self.rows[0]['test_files'][0], plan['review_source'])
        self.assertEqual(plan['drafts_naming_api'], ['pr-body.md'])
        rejected = subprocess.run(argv, capture_output=True)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertEqual(draft.read_text(), 'torch.a and torch.b')
        self.assertEqual(len(w.read_object(self.result)['results']), 2)
        draft.write_text('torch.b')
        rejected = subprocess.run(argv, capture_output=True)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertEqual(changed.read_text(), 'locally edited after the recorded run')
        changed.unlink()  # Subagent reviewed the isolated draft and removed its own change.
        self.assertEqual(json.loads(subprocess.check_output(argv))['remaining'], ['adapt-torch.b'])
        self.assertEqual(draft.read_text(), 'torch.b')
        self.checked()

    def test_fixed_release_is_scoped_and_fresh_run_can_select_again(self):
        basic = w.read_object(self.basic)
        basic['venv'] = 'data/venv/torch-2.15/python-3.11'
        self.put(self.basic, basic)
        with closing(sqlite3.connect(self.db)) as db, db:
            db.execute('CREATE TABLE "API支持数据_2.15" AS SELECT * FROM "API支持数据"')
        self.db_hash = w.file_hash(self.db)
        adapt = w.read_object(self.adapt)
        adapt['git-control']['branch-group'] = '2.15'
        self.put(self.adapt, adapt)
        maps = w.read_object(self.mapping)
        maps['2.15'] = dict(maps['nightly'])
        self.put(self.mapping, maps)
        w.withdraw(self.run, 'torch.alias')
        self.assertIn('adapt-torch.a', w.read_object(self.mapping)['nightly'])
        self.assertNotIn('adapt-torch.a', w.read_object(self.mapping)['2.15'])
        prepare(self.basic, root=self.root)
        self.assertEqual(w.read_object(self.adapt)['task-branches'], ['adapt-torch.b'])
        fresh = self.run.parent / 'fresh/basic_information.json'
        self.put(fresh, basic)
        prepare(fresh, root=self.root)
        self.assertIn('adapt-torch.a', w.read_object(fresh.parent / self.adapt.name)['task-branches'])
        self.checked()

    def test_old_attempt_and_single_api_gpu_evidence(self):
        old = self.loop/'npu-attempts/old-a'
        shutil.copytree(self.loop/'npu-attempts/a', old)
        private = self.loop/'reports/old-gpu-a'
        self.put(private/'manifest.json', {'results':[{'api':'torch.a'}]})
        self.put(private/'evidence/report.json', {'results':[{'api':'torch.a'}],
                 'manifest_sha256':w.file_hash(private/'manifest.json')})
        junit = private/'evidence/artifacts-1/junit.xml'
        junit.parent.mkdir()
        junit.write_text('<testsuite/>')
        old_mixed = self.report.parent/'artifacts-1/junit.xml'
        old_mixed.parent.mkdir()
        old_mixed.write_text('<shared/>')
        w.withdraw(self.run,'torch.a')
        self.assertFalse((old/'test.log').exists())
        self.assertFalse((private/'evidence/report.json').exists())
        self.assertFalse(junit.exists())
        self.assertEqual(old_mixed.read_text(),'<shared/>')
        self.assertTrue(self.report.is_file())
        self.checked()
