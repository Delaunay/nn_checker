import traceback
import json

from operator import mul
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as f

from benchutils.statstream import StatStream
from benchutils.chrono import MultiStageChrono

#
# repeat = 30
# number = 1000


def get_time(v: StatStream):
    val = v.avg
    if not val:
        val = v.val
    return val


def args_str(args):
    return ", ".join(map(lambda x: str(x), args))


# Convolutions
def conv_inputs(conv, batch_size, size):
    print(f'  Tensor({batch_size}, {conv.in_channels}, {args_str(size)})')
    return torch.randn(batch_size, conv.in_channels, *size).cuda(),


def backward(out):
    return out


def get_output_size(shape):
    return reduce(mul, shape[1:])


conv1d = [(1024,), (2048,), (8096,)]
conv2d = [(32, 32), (64, 64), (128, 128), (224, 224)]
conv3d = [(8, 8, 8), (16, 16, 16), (32, 32, 32)]

convolutions = {
    'args': [
        dict(in_channels=3,   out_channels=128, kernel_size=3),
        dict(in_channels=128, out_channels=128, kernel_size=3),
        dict(in_channels=128, out_channels=256, kernel_size=3),
        dict(in_channels=256, out_channels=128, kernel_size=3),
        # k=4
        dict(in_channels=3,   out_channels=128, kernel_size=4),
        dict(in_channels=128, out_channels=128, kernel_size=4),
        dict(in_channels=128, out_channels=256, kernel_size=4),
        dict(in_channels=256, out_channels=128, kernel_size=4),
        # k=6
        dict(in_channels=3,   out_channels=128, kernel_size=6),
        dict(in_channels=128, out_channels=128, kernel_size=6),
        dict(in_channels=128, out_channels=256, kernel_size=6),
        dict(in_channels=256, out_channels=128, kernel_size=6),
     ],
    'batch_size': [32, 64, 128, 256],
    'inputs': conv_inputs,
    'algos': [
        # Layer Constructor  | (After Batch & Channel) Tensor Size
        (nn.Conv1d,          conv1d),
        (nn.ConvTranspose1d, conv1d),
        (nn.Conv2d,          conv2d),
        (nn.ConvTranspose2d, conv2d),
        (nn.Conv3d,          conv3d),
        (nn.ConvTranspose1d, conv3d)
    ],
    'get_output_layer': backward
}


# Pooling
def pool_inputs(conv, batch_size, size):
    print(f'  Tensor({batch_size}, {args_str(size)})')
    return torch.randn(batch_size, *size).cuda(),


def pool_backward(out, indices):
    return out


channels = [3, 128, 256]
pool_1d = [(c,) + s for s in conv1d for c in channels]
pool_2d = [(c,) + s for s in conv2d for c in channels]
pool_3d = [(c,) + s for s in conv3d for c in channels]

pooling = {
    'args': [
        dict(kernel_size=3),
        dict(kernel_size=4),
        dict(kernel_size=6),
    ],
    'batch_size': [32, 64, 128, 256],
    'inputs': pool_inputs,
    'algos': [
        (nn.MaxPool1d,         pool_1d),
        (nn.MaxPool2d,         pool_2d),
        (nn.MaxPool3d,         pool_3d),
        # The input is the output of MaxPool1d so this sucks
        # (nn.MaxUnpool1d,       pool_1d),
        # (nn.MaxUnpool2d,       pool_2d),
        # (nn.MaxUnpool3d,       pool_3d),
        # (nn.AvgPool1d,         pool_1d),
        # (nn.AvgPool2d,         pool_2d),
        # (nn.AvgPool3d,         pool_3d),
        # (nn.AdaptiveAvgPool1d, pool_1d),
        # (nn.AdaptiveAvgPool2d, pool_2d),
        # (nn.AdaptiveAvgPool3d, pool_3d),
        # (nn.AdaptiveMaxPool1d, pool_1d),
        # (nn.AdaptiveMaxPool2d, pool_2d),
        # (nn.AdaptiveMaxPool3d, pool_3d),
    ],
    'get_output_layer': pool_backward
}


# Normalization
def norm_inputs(norm, batch_size, size):
    print(f'  Tensor({batch_size}, {norm.num_features}, {args_str(size)})')
    return torch.randn(batch_size, norm.num_features, *size).cuda(),


norm_1d = conv1d
norm_2d = conv2d
norm_3d = conv3d

batch_norm = {
    'args': [
        dict(num_features=3),
        dict(num_features=128),
        dict(num_features=256),
    ],
    'batch_size': [32, 64, 128, 256],
    'inputs': norm_inputs,
    'algos': [
        (nn.BatchNorm1d,    norm_1d),
        (nn.BatchNorm2d,    norm_2d),
        (nn.BatchNorm3d,    norm_3d),
        (nn.InstanceNorm1d, norm_1d),
        (nn.InstanceNorm2d, norm_2d),
        (nn.InstanceNorm3d, norm_3d),
    ],
    'get_output_layer': backward
}


# Recurrent Layers
def rnn_inputs(rnn, batch_size, size):
    print(f'  Tensor({args_str(size)}, {batch_size}, {rnn.input_size})')
    num_directions = 1

    input = torch.randn(*size, batch_size, rnn.input_size).cuda()
    h0 = torch.randn(rnn.num_layers * num_directions, batch_size, rnn.hidden_size).cuda()

    if isinstance(rnn, nn.LSTM):
        c0 = torch.randn(rnn.num_layers * num_directions, batch_size, rnn.hidden_size).cuda()
        h0 = (h0, c0)

    return input, h0


def rnn_tanh(**kwargs):
    return nn.RNN(**kwargs, nonlinearity='tanh')


def rnn_relu(**kwargs):
    return nn.RNN(**kwargs, nonlinearity='relu')


def rnn_backward(out, state):
    return out


def rnn_outsize(shape):
    # Batch is in the middle this time
    return shape[0] * shape[2]


seq_length = [(6,), (12,), (24,), (32,)]
recurrent = {
    'args': [
        dict(input_size=32, hidden_size=64, num_layers=16),
        dict(input_size=64, hidden_size=32, num_layers=16),
        dict(input_size=16, hidden_size=32, num_layers=64),
    ],
    'batch_size': [32, 64, 128, 256],
    'inputs': rnn_inputs,
    'algos': [
        (nn.LSTM,     seq_length),
        (rnn_tanh,    seq_length),
        (rnn_relu,    seq_length),
        (nn.GRU,      seq_length)
    ],
    'get_output_layer': rnn_backward,
    'get_output_size': rnn_outsize
}


def run_check(spec, repeat=10, number=20, report_name=None):
    chrono = MultiStageChrono(skip_obs=2)

    args = spec['args']
    input_gen = spec['inputs']
    algos = spec['algos']
    batch_sizes = spec['batch_size']
    get_output_layer = spec['get_output_layer']
    get_output_size = spec['get_output_size']

    for algo, tensor_sizes in algos:
        for arg in args:
            # initialize the conv layer that we will benchmark
            layer = algo(**arg).cuda()

            for batch_size in batch_sizes:
                for tensor_size in tensor_sizes:
                    name = f'algo={algo.__name__},batch={batch_size},tensor={tensor_size},arg={arg}'
                    print(name)
                    try:
                        input = input_gen(layer, batch_size, tensor_size)
                        target = None
                        size = None
                        criterion = nn.MSELoss()

                        # Benchmark the layer
                        for i in range(0, repeat):
                            # ---
                            with chrono.time(name) as timer:
                                for _ in range(0, number):
                                    out = layer(*input)
                                    out = get_output_layer(*out)

                                    if target is None:
                                        if get_output_size is None:
                                            size = reduce(mul, out.shape[1:])
                                        else:
                                            size = get_output_size(out.shape)

                                        target = torch.randn(batch_size, size).cuda()

                                    loss = criterion(target, out.view(-1, size))
                                    loss.backward()

                                    torch.cuda.synchronize()

                        print(f'  Ran in {timer.avg:5.2f}s {timer.avg * repeat:5.2f}s')
                        # ---
                    except Exception as e:
                        print(f'[!] > {e}')
                        print(traceback.format_exc())

    report = chrono.to_json(indent=2)
    print(report)

    if report_name is not None:
        with open(report_name, 'w') as file:
            file.write(report)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bench', nargs='*', type=str,
            choices=['convolutions', 'pooling', 'batch_norm', 'recurrent'],
            default=['recurrent'])

    args = parser.parse_args()
    variables = locals()
    for bench in args.bench:
        run_check(variables[bench], 20, 5, report_name='recurrent.json')

# run_check(convolutions, 20, 5, name='convolutions.json')
# run_check(pooling, 20, 5, name='pooling.json')
# run_check(normalization, 20, 5, name='norm.json')

