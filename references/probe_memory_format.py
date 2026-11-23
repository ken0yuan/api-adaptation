"""Separate clone, same-size resize_, and a candidate strided allocation/copy."""
import argparse
import json
import math

import torch


def observe(x, source, fmt):
    return dict(stride=list(x.stride()), requested_format=x.is_contiguous(memory_format=fmt),
                values_equal=torch.equal(x.cpu(), source), device=str(x.device))


def probe(shape, fmt, device):
    source = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
    expected = source.clone(memory_format=fmt)
    x = source.to(device)
    result = dict(shape=list(shape), expected_stride=list(expected.stride()))
    # Use separate tensors so clone cannot mask the behavior of resize_.
    for name, operation in (
        ('clone', lambda: x.clone(memory_format=fmt)),
        ('resize_only', lambda: x.clone().resize_(shape, memory_format=fmt)),
        ('strided_copy', lambda: torch.empty_strided(shape, expected.stride(),
                                                    dtype=x.dtype, device=device).copy_(x)),
    ):
        try:
            result[name] = observe(operation(), source, fmt)
        except RuntimeError as error:
            result[name] = {'error': str(error)}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', required=True)
    selected = parser.parse_args().device
    if selected.split(':')[0] in ('npu', 'privateuseone'):
        import torch_npu
    device = torch.device(selected)
    print(json.dumps([probe((2, 3, 4, 5), torch.channels_last, device),
                      probe((2, 3, 4, 5, 6), torch.channels_last_3d, device)], indent=2))
