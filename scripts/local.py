import sys
import os
sys.path.append(os.getcwd() + '/src')

import io
import argparse
import subprocess
import contextlib
from glob import glob

from utils.results import gather_missing_indices
import experiment.ExperimentModel as Experiment

parser = argparse.ArgumentParser()
parser.add_argument('--runs', type=int, required=True)
parser.add_argument('-e', type=str, nargs='+', required=True)
parser.add_argument('--entry', type=str, default='src/main.py')
parser.add_argument('--results', type=str, default='./')

if __name__ == "__main__":
    cmdline = parser.parse_args()

    # Expand glob patterns
    expanded_paths = []
    for pattern in cmdline.e:
        matches = glob(pattern, recursive=True)
        if matches:
            expanded_paths.extend(matches)
        else:
            expanded_paths.append(pattern)
    cmdline.e = sorted(expanded_paths)

    # Suppress the built-in print from gather_missing_indices; we print our own format.
    with contextlib.redirect_stdout(io.StringIO()):
        e_to_missing = gather_missing_indices(cmdline.e, cmdline.runs, loader=Experiment.load, base=cmdline.results)

    # Print per-file status: "path missing/total"
    grand_total = 0
    for path in cmdline.e:
        exp = Experiment.load(path)
        total = exp.numPermutations() * cmdline.runs
        grand_total += total
        missing = len(e_to_missing[path])
        print(f'{path} {missing}/{total}')

    total_missing = sum(len(v) for v in e_to_missing.values())
    remaining = total_missing

    for path, missing in e_to_missing.items():
        if not missing:
            continue
        idx_args = ' '.join(map(str, missing))
        subprocess.run(f'python {cmdline.entry} --silent -e {path} -i {idx_args}', shell=True, check=True)
        remaining -= len(missing)
        print(f'{path} done  [{grand_total - remaining}/{grand_total} completed]')

    print(f'all done!  [{grand_total}/{grand_total} completed]')
