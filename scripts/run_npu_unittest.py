"""Run one selected NPU unittest file and write the JUnit receipt for this execution."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import unittest
import xml.etree.ElementTree as ET


def run_file(source, filename, attempt):
    import torch
    import torch_npu
    import xmlrunner

    source = source.resolve()
    path = (source / filename).resolve()
    relative = path.relative_to(source).as_posix()
    if not path.is_file() or path.suffix != '.py':
        raise ValueError(f'Expected one existing Python test file: {relative}')
    if not torch.npu.is_available():
        raise RuntimeError('NPU unavailable')
    model = torch.npu.get_device_name(0)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    folder = attempt / 'unittest' / relative
    folder.mkdir(parents=True, exist_ok=False)

    # A real importable module also lets multiprocessing.spawn resolve rank workers.
    sys.path.insert(0, str(path.parent))
    module = importlib.import_module(path.stem)
    if Path(module.__file__).resolve() != path:
        raise RuntimeError(f'Imported a different file for {relative}')
    if getattr(module, 'DEVICE_NAME', 'npu') != 'npu':
        raise RuntimeError(f'{relative} selected a non-NPU backend')
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    if not suite.countTestCases():
        raise ValueError(f'No unittest cases collected from {relative}')
    temporary = folder / 'junit.xml.partial'
    with temporary.open('xb') as output:
        result = xmlrunner.XMLTestRunner(output=output, verbosity=2).run(suite)

    tree = ET.parse(temporary)
    prefix = relative[:-3].replace('/', '.')
    for case in tree.iter('testcase'):
        name = case.get('classname', '')
        if name.startswith(path.stem + '.'):
            case.set('classname', prefix + name[len(path.stem):])
            case.set('file', relative)
    counts = {key: sum(int(s.attrib[key]) for s in tree.iter('testsuite'))
              for key in ('tests', 'errors', 'failures', 'skipped')}
    if counts['tests'] != sum(1 for _ in tree.iter('testcase')):
        raise ValueError('JUnit case count differs from the suite totals')
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f'Test file changed during execution: {relative}')
    tree.write(temporary, encoding='utf-8', xml_declaration=True)
    temporary.replace(folder / 'junit.xml')
    code = int(not result.wasSuccessful())
    status = ('失败' if code else '全部跳过' if counts['tests'] == counts['skipped']
              else '部分跳过' if counts['skipped'] else '通过')
    receipt = dict(device_kind='NPU', device_model=model, torch_version=str(torch.__version__),
                   torch_npu_version=str(torch_npu.__version__), selected_files=[relative],
                   file_hashes={relative: digest}, junit=counts, junit_generated=True,
                   test_started=True, scope=relative, test_exit_code=code,
                   test_exit_code_display=str(code), execution_status=status)
    (folder / 'execution-result.json').write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--file', required=True)
    parser.add_argument('--attempt', required=True, type=Path)
    args = parser.parse_args()
    private_tools = Path.home() / '.cache/npu-unittest' / (
        sys.implementation.cache_tag + '-' + platform.machine())
    sys.path.append(os.environ.get('NPU_JUNIT_TOOLS', str(private_tools)))
    sys.exit(run_file(args.source, args.file, args.attempt))
