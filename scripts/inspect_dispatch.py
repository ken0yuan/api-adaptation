"""Inspect registrations in the installed NPU build for exact operator overloads."""
import argparse
import json
import sys

import torch
import torch_npu


def inspect(name):
    op, _, overload = name.partition('.')
    if '::' not in op:
        raise ValueError(f'Expected namespace::operator[.overload], got {name}')
    schema = torch._C._dispatch_find_schema_or_throw(op, overload).schema()
    table = torch._C._dispatch_dump_table(name).splitlines()
    keys = ('PrivateUse1', 'AutogradPrivateUse1', 'CPU', 'CUDA', 'Meta')
    return {
        'op': name,
        'schema': str(schema),
        'direct': {key: torch._C._dispatch_has_kernel_for_dispatch_key(name, key) for key in keys},
        'table': [line for line in table if line.partition(':')[0] in keys],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('op', nargs='+', help='Exact dispatcher name, e.g. aten::add.out')
    args = parser.parse_args()
    result = dict(python=sys.executable, torch=torch.__version__, torch_file=torch.__file__,
                  torch_npu=torch_npu.__version__, torch_npu_file=torch_npu.__file__, operators=[])
    failed = False
    for name in args.op:
        try:
            result['operators'].append(inspect(name))
        except (ValueError, RuntimeError) as error:
            result['operators'].append({'op': name, 'error': str(error)})
            failed = True
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
