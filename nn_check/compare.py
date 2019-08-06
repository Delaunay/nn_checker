import argparse
import json
from collections import defaultdict
from benchutils.statstream import StatStream

parser = argparse.ArgumentParser()
parser.add_argument('file1', default='recurrent_2.json')
parser.add_argument('file2', default='recurrent_base.json')
parser.add_argument('--metrics', nargs='*', default=['avg', 'min', 'max'])
args = parser.parse_args()


def fix_report(r):
    if isinstance(r, str):
        r = json.loads(r)
    return r


report1 = fix_report(json.loads(open(args.file1, 'r').read()))
report2 = fix_report(json.loads(open(args.file2, 'r').read()))
acc = defaultdict(lambda: StatStream(drop_first_obs=0))


bench_names = set(report1.keys()).union(set(report2.keys()))

for bench_name in bench_names:
    metrics1 = report1.get(bench_name)
    metrics2 = report2.get(bench_name)

    print(bench_name)

    metrics = set(metrics1.keys()).union(metrics2.keys())
    for metric in metrics:
        if metric == 'unit':
            continue

        value1 = metrics1[metric]
        value2 = 'NA'
        value3 = 'NA'

        if metrics2 is not None:
            value2 = metrics2.get(metric)

            if value2 is not None:
                speedup = value1 / value2
                acc[metric].update(speedup)

                value3 = f'{speedup:8.4f}'
                value2 = f'{value2:8.4f}'

        if metric in args.metrics:
            print(f'    - {metric:>6} : {value1:8.4f} {value2} x{value3}')
    print()

print('Summary')
print('-------')
stat = list(acc.items())
stat.sort()
for m, v in stat:
    print(f'    - {m:>6} [avg: {v.avg:8.2f} sd: {v.sd:8.2f} min: {v.min:8.2f} max: {v.max:8.2f}]')

