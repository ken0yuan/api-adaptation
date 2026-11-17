"""Exercise GPU collection auditing with explicit device-independent manifests."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class GPUAuditTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.name = 'test/test_python_api.py'
        file = self.root / self.name
        file.parent.mkdir()
        file.write_text('VALUE = 1\n', encoding='utf-8')
        self.sha = hashlib.sha256(file.read_bytes()).hexdigest()
        self.item = SimpleNamespace(path=file, module=SimpleNamespace())
        self.script = self.root / 'step1_gpu_audit.py'
        shutil.copyfile(Path(__file__).resolve().parents[1] / 'scripts/step1_gpu_audit.py', self.script)

    def load_plugin(self, independent):
        manifest = {'files': {self.name: self.sha},
                    'results': [{'api': 'torch.python_api', 'test_files': [self.name],
                                 'device_independent': independent}]}
        (self.root / 'step1-manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        pytest_stub = SimpleNamespace(UsageError=RuntimeError)
        with patch.dict('sys.modules', {'pytest': pytest_stub}), patch.dict(os.environ, RUN_SUPPLEMENT_API_TESTS='1'):
            return runpy.run_path(str(self.script))

    def test_pure_python_collection_accepts_no_device_selector(self):
        plugin = self.load_plugin(True)
        plugin['pytest_collection_modifyitems']([self.item])
        self.assertEqual(plugin['AUDIT']['files'][self.name]['collected'], 1)
        self.assertIsNone(plugin['AUDIT']['files'][self.name]['selected_device'])

    def test_hardware_collection_still_requires_cuda(self):
        plugin = self.load_plugin(False)
        with self.assertRaisesRegex(RuntimeError, 'source/device mismatch'):
            plugin['pytest_collection_modifyitems']([self.item])

    def test_pure_python_collection_still_rejects_changed_source(self):
        plugin = self.load_plugin(True)
        self.item.path.write_text('VALUE = 2\n', encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError, 'source/device mismatch'):
            plugin['pytest_collection_modifyitems']([self.item])


if __name__ == '__main__':
    unittest.main()
