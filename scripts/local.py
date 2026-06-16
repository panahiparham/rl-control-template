from time import sleep
import sys
import os
sys.path.append(os.getcwd() + '/src')

import io
import argparse
import subprocess
import contextlib
from glob import glob
from multiprocessing.pool import Pool

from utils.results import gather_missing_indices
import experiment.ExperimentModel as Experiment

parser = argparse.ArgumentParser()
parser.add_argument('--runs', type=int, required=True)
parser.add_argument('-e', type=str, nargs='+', required=True)
parser.add_argument('--entry', type=str, default='src/main.py')
parser.add_argument('--results', type=str, default='./')
parser.add_argument('--cpus', type=int, default=os.cpu_count())
parser.add_argument('--cores-per-task', type=int, default=1)


def run_file(args: tuple[str, list[int], str, int]) -> tuple[str, int]:
    path, missing, entry, cores = args
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(cores)
    subprocess.run(
        f'python {entry} --silent -e {path} -i {" ".join(map(str, missing))}',
        shell=True, check=True, env=env,
    )
    return path, len(missing)


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

    if total_missing == 0:
        print(f'all done!  [0/{grand_total} remaining]')
        sys.exit(0)

    num_parallel = max(1, cmdline.cpus // cmdline.cores_per_task)
    tasks = [
        (path, missing, cmdline.entry, cmdline.cores_per_task)
        for path, missing in e_to_missing.items() if missing
    ]

    with Pool(num_parallel) as pool:
        for path, n in pool.imap_unordered(run_file, tasks):
            remaining -= n
            print(f'{path} done  [{remaining}/{grand_total} remaining]')
            sleep(0.1)  # small delay to ensure print order is consistent
