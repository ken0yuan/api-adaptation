import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import step1_remote as controller
import step1_remote_worker as worker
import step1_relay as relay


class RemoteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()

    def test_portable_packages_exclude_runtime_and_ascend_with_reasons(self):
        kept, omitted = controller.portable_packages([
            {'name': 'numpy', 'version': '1.26.4'},
            {'name': 'torch', 'version': '2.16'},
            {'name': 'torch_npu', 'version': '2.16'},
            {'name': 'hccl', 'version': '1', 'direct': {'url': 'file:///usr/local/Ascend/hccl.whl'}},
        ], {})
        self.assertEqual(kept, {'numpy': '1.26.4'})
        self.assertEqual(len(omitted), 3)
        self.assertTrue(all(e['reason'] for e in omitted))

    def test_unknown_local_dependency_cannot_silently_disappear(self):
        with self.assertRaisesRegex(ValueError, 'portable replacements'):
            controller.portable_packages([{'name': 'private-helper', 'version': '1', 'direct': {'url': 'file:///private/helper'}}], {})

    def test_lint_packages_do_not_enter_runtime_requirements(self):
        def package(name, *requires):
            return {'name': name, 'version': '1', 'requires': requires}

        installed = [
            package('Flake8', 'pyflakes>=1', 'mccabe', 'pycodestyle'),
            package('flake8-bugbear', 'flake8', 'attrs>=1'),
            package('flake8-comprehensions', 'flake8'),
            package('flake8-simplify', 'flake8'),
            package('attrs'), package('pyflakes'), package('mccabe'), package('pycodestyle'),
            package('pytest', 'attrs; extra == "dev"'),
            package('torch-npu', 'attrs'),
        ]
        kept, omitted = controller.portable_packages(installed, {})
        self.assertEqual(kept, {'pytest': '1'})
        self.assertEqual(len(omitted), 9)

        kept, _ = controller.portable_packages(installed + [package('test-helper', 'attrs>=1')], {})
        self.assertEqual(kept, {'pytest': '1', 'test-helper': '1', 'attrs': '1'})

        after_uninstall = [package('attrs'), package('torch-npu', 'attrs'), package('pytest')]
        kept, _ = controller.portable_packages(after_uninstall, {})
        self.assertEqual(kept, {'pytest': '1'})
        kept, _ = controller.portable_packages(after_uninstall + [package('test-helper', 'attrs')], {})
        self.assertEqual(kept['attrs'], '1')

    def test_paths_cannot_escape_source(self):
        with self.assertRaises(ValueError):
            controller.contained(self.root, '../outside.py')

    def test_relay_runs_framed_commands_inside_forced_container_shell(self):
        import shlex
        class Channel:
            closed = False
            data = b''
            def sendall(self, command):
                if command.startswith('stty'):
                    return
                text = io.StringIO()
                with contextlib.redirect_stdout(text):
                    exec(shlex.split(command)[-1], {})
                self.data = b'\x1b[?2004hroot# ' + text.getvalue().encode()
            def recv_ready(self):
                return bool(self.data)
            def recv(self, count):
                data, self.data = self.data, b''
                return data
        class Client:
            def invoke_shell(self, **kwargs):
                return Channel()
        shell = relay.SourceShell(Client(), 'python3')
        self.assertEqual(shell.call("result={'value': 7}"), {'value': 7})
        with self.assertRaisesRegex(RuntimeError, 'fixture error'):
            shell.call("raise ValueError('fixture error')")

    def test_fixture_execution_distinguishes_failure_skip_and_device_mismatch(self):
        tests = self.root / 'tests'
        tests.mkdir()
        original = Path.cwd()
        self.addCleanup(os.chdir, original)
        os.chdir(self.root)
        for body, expected in [('self.assertEqual(2+2, 4)', 'pass'), ('self.fail("broken behavior")', 'fail'), ('self.skipTest("missing capability")', 'partial')]:
            (tests / 'fixture.py').write_text('import unittest\nDEVICE_NAME="cpu"\nclass Checks(unittest.TestCase):\n def test_behavior(self):\n  ' + body + '\n')
            with contextlib.redirect_stderr(io.StringIO()):
                worker.run_file('fixture.py', 'cpu', 'detail.json')
            r = json.loads(Path('detail.json').read_text())
            self.assertEqual(r['execution'], expected)
            self.assertEqual(r['tests_run'], 1)
        with contextlib.redirect_stderr(io.StringIO()):
            worker.run_file('fixture.py', 'cuda', 'detail.json')
        self.assertEqual(json.loads(Path('detail.json').read_text())['execution'], 'blocked')

    def test_timeout_is_not_success(self):
        code = worker.command([sys.executable, '-c', 'import time; time.sleep(5)'], self.root / 'timeout.log', 0.1)
        self.assertEqual(code, 124)

    def test_bootstrap_failure_produces_fetchable_report_for_every_api(self):
        manifest = {'device': 'cuda', 'files': {}, 'python_minor': [0, 0], 'results': [{'api': 'torch.foo', 'test_files': ['x.py']}]}
        (self.root / 'manifest.json').write_text(json.dumps(manifest))
        script = Path(worker.__file__).read_text()
        (self.root / 'worker.py').write_text(script)
        proc = subprocess.run([sys.executable, str(self.root / 'worker.py')], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 2)
        report = json.loads((self.root / 'evidence/report.json').read_text())
        self.assertEqual(report['execution'], 'blocked')
        self.assertEqual(report['results'][0]['execution'], 'blocked')
        self.assertIn('Python', report['reason'])

    def test_reference_result_does_not_overwrite_npu_outcome(self):
        source = self.root / 'source'
        source.mkdir()
        (source / 'x.py').write_text('pass\n')
        results = self.root / 'result.json'
        results.write_text(json.dumps({'results': [{'api': 'torch.foo', 'test_files': ['x.py'], 'execution': 'fail'}]}))
        output = self.root / 'remote-run'
        output.mkdir()
        (output / 'evidence').mkdir()
        controller.write_json(output / 'manifest.json', {'device': 'cuda', 'files': {'x.py': hashlib.sha256((source / 'x.py').read_bytes()).hexdigest()}, 'results': [{'api': 'torch.foo', 'test_files': ['x.py']}]})
        config = {'results': str(results), 'source_root': str(source), 'run_dir': str(self.root)}
        controller.merge(config, output, {'results': [{'api': 'torch.foo', 'execution': 'pass'}]})
        entry = json.loads(results.read_text())['results'][0]
        self.assertEqual(entry['execution'], 'fail')
        self.assertEqual(entry['device_validation']['cuda']['execution'], 'pass')
        (source / 'x.py').write_text('changed\n')
        with self.assertRaisesRegex(ValueError, 'Source changed'):
            controller.merge(config, output, {'results': [{'api': 'torch.foo', 'execution': 'pass'}]})


if __name__ == '__main__':
    unittest.main()
