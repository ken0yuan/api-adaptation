import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from step1_wsl_gpu import analyze, merge, request, submit, write_json
import step1_wsl_bridge as bridge
from unittest.mock import patch
import zipfile


class WslTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / 'source'
        (self.source / 'test').mkdir(parents=True)
        (self.source / 'test/test_sample.py').write_text('original\n', encoding='utf-8')
        self.file = 'test/test_sample.py'
        sha = hashlib.sha256((self.source / self.file).read_bytes()).hexdigest()
        self.manifest = {'device': 'cuda', 'files': {self.file: sha},
                         'results': [{'api': 'torch.sample', 'test_files': [self.file]}]}
        self.raw = json.dumps(self.manifest).encode()
        self.state = {'head_sha': 'abc', 'branch': 'check/sample', 'run_id': 123, 'cuda_build': 'cu132'}
        self.run = {'id': 123, 'run_attempt': 1, 'head_sha': 'abc', 'head_branch': 'check/sample',
                    'html_url': 'https://example.test/run', 'event': 'workflow_dispatch', 'conclusion': 'success'}
        self.artifact = self.root / 'artifact'
        self.artifact.mkdir()
        (self.artifact / 'commit.txt').write_text('abc\n')
        (self.artifact / 'selected-files.txt').write_text(self.file + '\n')
        (self.artifact / 'junit.xml').write_text('<testsuites><testsuite tests="1"/></testsuites>')
        self.audit = {'manifest_sha256': hashlib.sha256(self.raw).hexdigest(), 'run_flag': '1',
                      'environment': {'cuda_available': True, 'cuda_build': '13.2', 'torch': '2.16.0.dev20261116+cu132', 'gpu_name': 'NVIDIA GeForce RTX 5070 Ti'},
                      'files': {self.file: {'sha256': sha, 'selected_device': 'cuda', 'collected': 1}},
                      'reports': [{'nodeid': self.file + '::Tests::test_real', 'when': when,
                                   'outcome': 'passed', 'wasxfail': False} for when in ['setup', 'call', 'teardown']]}

    def report(self):
        data = json.dumps(self.audit).encode()
        with tarfile.open(self.artifact / 'test-output.tar.gz', 'w:gz') as archive:
            item = tarfile.TarInfo('./step1-audit.json')
            item.size = len(data)
            archive.addfile(item, io.BytesIO(data))
        return analyze(self.raw, self.state, self.run, self.artifact)

    def test_fixed_gpu_target_requires_exact_public_version(self):
        self.manifest.update(torch_index='https://download.pytorch.org/whl/cu132',torch_requirement='torch==2.15.0')
        self.raw = json.dumps(self.manifest).encode()
        self.audit['manifest_sha256'] = hashlib.sha256(self.raw).hexdigest()
        for version, expected in [('2.15.0+cu132','pass'),('2.15.1+cu132','blocked'),
                                  ('2.15.0.dev20261118+cu132','blocked')]:
            self.audit['environment']['torch'] = version
            self.assertEqual(self.report()['execution'], expected)

    def test_ai_loop_gpu_uses_step0_requirements_for_nightly_and_retry(self):
        from adaptation_target import gpu_target
        from step1_remote import export_inputs
        run = self.root/'run'
        run.mkdir()
        write_json(run/'basic_information.json', {'python_version':'3.11.17', 'venv':'data/venv/python-3.11.17'})
        path = run/'requirements.txt'
        path.write_text('torch-npu==2.15.0\ntorch==2.14; python_version < "3.11"\n')
        nightly = 'https://download.pytorch.org/whl/nightly/cu132'
        config = {'run_dir':str(run),'source_root':str(self.source),'torch_index':nightly}
        self.assertEqual(gpu_target(config), (nightly, 'torch'))
        with self.assertRaisesRegex(ValueError, 'conflicts with Step 0'):
            gpu_target(dict(config, torch_requirement='torch==2.15'))
        path.write_text('torch==2.15\n')
        release = 'https://download.pytorch.org/whl/cu132'
        config['torch_index'] = release
        manifest = dict(self.manifest, torch_requirement='torch==2.15.0', torch_index=release)
        export_inputs(config, {'results':self.manifest['results']}, manifest)
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'Missing Step 0 requirements'):
            gpu_target(config)

    def test_fixed_export_translates_cpu_source_to_workflow_release_pin(self):
        from unittest.mock import patch
        from step1_remote import prepare, export_inputs
        from step1_wsl_gpu import bundle_files
        run = self.root/'run'
        run.mkdir()
        write_json(run/'basic_information.json', {'venv':'data/venv/torch-2.15.0/python-3.11.17','python_version':'3.11.17'})
        (run/'requirements.txt').write_text('-c torch-constraints.txt\n')
        (run/'torch-constraints.txt').write_text('torch==2.15; python_version == "3.11"\n')
        write_json(run/'result.json', {'results': self.manifest['results']})
        config = {'source_root':str(self.source), 'source_python':'fixture','run_dir':str(run),
                  'results':str(run/'result.json'), 'torch_index':'https://download.pytorch.org/whl/cu132'}
        inventory = {'python':[3,11,17], 'packages':[{'name':'torch','version':'2.15.0+cpu','direct':{}}]}
        with patch('step1_remote.inventory', return_value=inventory):
            manifest = prepare(config,run/'export')
        self.assertEqual(manifest['torch_requirement'], 'torch==2.15')
        _, _, files = bundle_files(run/'export/bundle.zip')
        self.assertEqual(files['requirements.txt'], b'torch==2.15\n')
        (run/'requirements.txt').write_text('torch==2.14\n')
        with self.assertRaisesRegex(ValueError, 'different PyTorch target'):
            export_inputs(config, {'results': self.manifest['results']}, manifest)

    def test_gpu_pass_requires_complete_cuda_execution_and_matches_file(self):
        report = self.report()
        self.assertEqual(report['results'][0]['execution'], 'pass')
        self.assertEqual(report['results'][0]['counts'], {'passed': 1, 'failed': 0, 'skipped': 0})
        self.audit['files'][self.file]['selected_device'] = 'cpu'
        self.assertEqual(self.report()['execution'], 'blocked')

    def test_other_gpu_models_are_accepted_with_matching_cuda_environment(self):
        for name in ['NVIDIA A100-SXM4-80GB', 'NVIDIA H100 80GB HBM3', 'NVIDIA RTX 4090']:
            with self.subTest(gpu=name):
                self.audit['environment']['gpu_name'] = name
                self.assertEqual(self.report()['execution'], 'pass')

    def test_device_independent_case_can_pass_without_cuda_selection(self):
        self.manifest['results'][0]['device_independent'] = True
        self.raw = json.dumps(self.manifest).encode()
        self.audit['manifest_sha256'] = hashlib.sha256(self.raw).hexdigest()
        self.audit['environment'].update(cuda_available=False, gpu_name=None)
        for selected in [None, 'cpu', 'cuda']:
            with self.subTest(selected=selected):
                self.audit['files'][self.file]['selected_device'] = selected
                result = self.report()
                self.assertEqual(result['execution'], 'pass')
                self.assertTrue(result['results'][0]['device_independent'])
                self.assertEqual(result['results'][0]['files'][0]['selected_device'], selected)

    def test_device_independent_flag_does_not_bypass_raw_result_checks(self):
        self.manifest['results'][0]['device_independent'] = True
        self.raw = json.dumps(self.manifest).encode()
        self.audit['manifest_sha256'] = hashlib.sha256(self.raw).hexdigest()
        self.audit['files'][self.file]['selected_device'] = None
        self.audit['files'][self.file]['collected'] = 0
        self.assertEqual(self.report()['execution'], 'blocked')
        self.audit['files'][self.file]['collected'] = 1
        self.audit['reports'][1]['outcome'] = 'failed'
        self.assertEqual(self.report()['execution'], 'fail')
        self.audit['reports'][1]['outcome'] = 'passed'
        self.audit['reports'][1]['wasxfail'] = True
        self.assertEqual(self.report()['execution'], 'partial')

    def test_device_independent_export_is_bound_to_manifest(self):
        from step1_remote import export_inputs
        row = {'api': 'torch.sample', 'test_files': [self.file]}
        config = {'source_root': str(self.source), 'device_independent_apis': ['torch.sample']}
        entries, files, hashes, _ = export_inputs(config, {'results': [row]})
        self.assertTrue(entries[0]['device_independent'])
        manifest = dict(self.manifest, results=entries)
        entries_after, _, _, _ = export_inputs({'source_root': str(self.source)}, {'results': [row]}, manifest)
        self.assertEqual(entries_after, entries)
        with self.assertRaisesRegex(ValueError, 'scope changed'):
            export_inputs(dict(config, device_independent_apis=[]), {'results': [row]}, manifest)
        with self.assertRaisesRegex(ValueError, 'device_independent_apis'):
            export_inputs(dict(config, device_independent_apis=['typo']), {'results': [row]})

    def test_skips_and_missing_teardown_cannot_claim_pass(self):
        self.audit['reports'][1]['outcome'] = 'skipped'
        self.assertEqual(self.report()['execution'], 'partial')
        self.audit['reports'][1]['outcome'] = 'passed'
        self.audit['reports'].pop()
        self.assertEqual(self.report()['execution'], 'partial')

    def test_artifact_sha_or_bundle_mismatch_cannot_claim_pass(self):
        (self.artifact / 'commit.txt').write_text('other-sha')
        self.assertEqual(self.report()['execution'], 'blocked')
        (self.artifact / 'commit.txt').write_text('abc')
        self.audit['manifest_sha256'] = 'other-bundle'
        self.assertEqual(self.report()['execution'], 'blocked')

    def test_import_preserves_every_npu_and_task_field_and_rejects_stale_source(self):
        output = self.root / 'run' / 'api-loop' / 'reports' / 'wsl'
        (output / 'evidence').mkdir(parents=True)
        (output / 'manifest.json').write_bytes(self.raw)
        write_json(output / 'submission.json', self.state)
        write_json(output / 'evidence/report.json', self.report())
        path = self.root / 'run/api-loop/result.json'
        entry = {'api': 'torch.sample', 'test_files': [self.file], 'execution': 'fail',
                 'report': 'npu.log', 'status': '待适配', 'next_step': 'step2',
                 'device_validation': {'npu': {'execution': 'fail'}, 'cpu': {'execution': 'pass'}}}
        write_json(path, {'results': [entry]})
        config = {'source_root': str(self.source), 'run_dir': str(self.root / 'run'), 'results': str(path)}
        merge(config, output)
        updated = json.loads(path.read_text(encoding='utf-8'))['results'][0]
        self.assertEqual(updated['device_validation'].pop('cuda')['execution'], 'pass')
        self.assertEqual(updated, entry)
        before = path.read_bytes()
        (self.source / self.file).write_text('changed\n')
        with self.assertRaisesRegex(ValueError, 'Source changed'):
            merge(config, output)
        self.assertEqual(path.read_bytes(), before)

    def test_existing_submission_cannot_be_dispatched_again(self):
        with self.assertRaisesRegex(ValueError, 'use fetch'):
            submit(self.root / 'unused.zip', self.artifact, 'check/sample')

    def test_server_request_is_idempotent_and_preserves_npu_result(self):
        run = self.root / 'run'
        output = run / 'reports/wsl-request'
        output.mkdir(parents=True)
        manifest = dict(self.manifest, python_minor=[3, 11],
                        torch_index='https://download.pytorch.org/whl/nightly/cu132')
        raw = json.dumps(manifest).encode()
        (output / 'manifest.json').write_bytes(raw)
        with zipfile.ZipFile(output / 'bundle.zip', 'w') as archive:
            archive.writestr('manifest.json', raw)
            archive.writestr('requirements.txt', '')
            archive.writestr('tests/' + self.file, (self.source / self.file).read_bytes())
        results = run / 'result.json'
        write_json(results, {'results': [dict(manifest['results'][0], execution='fail', report='npu.log')]})
        config = run / 'export.json'
        write_json(config, {'device': 'cuda', 'source_root': str(self.source), 'run_dir': str(run),
                            'results': str(results), 'torch_index': manifest['torch_index']})
        original = results.read_bytes()
        ticket = request(config, output)
        self.assertEqual(ticket['status'], 'needs_workstation')
        ticket['status'] = 'running'
        write_json(output / 'step1-request.json', ticket)
        self.assertEqual(request(config, output)['status'], 'running')
        self.assertEqual(results.read_bytes(), original)
        (self.source / self.file).write_text('changed\n')
        with self.assertRaisesRegex(ValueError, 'Source changed'):
            request(config, output)

    def test_assist_resumes_existing_submission_without_dispatching_again(self):
        output = self.artifact
        (output / 'manifest.json').write_bytes(self.raw)
        write_json(output / 'submission.json', self.state)
        ticket = {'type': 'step1_wsl_gpu', 'schema_version': 1, 'request_path': '/task/request.json',
                  'manifest_sha256': hashlib.sha256(self.raw).hexdigest(), 'branch': self.state['branch'],
                  'source': {'python': 'python3', 'config': '/task/export.json',
                             'output': '/task/artifact', 'skill_dir': '/skills/api-adaptation'}}
        with patch.object(bridge, 'NativeClient'), patch.object(bridge, 'SourceShell') as shell, patch.object(bridge, 'execute', return_value=3) as execute:
            shell.return_value.call.return_value = ticket
            self.assertEqual(bridge.assist('source-alias', '/task/request.json', output, 0), 3)
            self.assertEqual(execute.call_args.args[0], 'fetch')
            self.assertEqual(execute.call_args.args[1]['source']['ssh_alias'], 'source-alias')

    def test_assist_rejects_other_request_types(self):
        with patch.object(bridge, 'NativeClient'), patch.object(bridge, 'SourceShell') as shell, patch.object(bridge, 'execute') as execute:
            shell.return_value.call.return_value = {'type': 'arbitrary_command', 'schema_version': 1}
            with self.assertRaisesRegex(ValueError, 'Only Step1'):
                bridge.assist('source-alias', '/task/request.json', self.artifact, 0)
            execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
