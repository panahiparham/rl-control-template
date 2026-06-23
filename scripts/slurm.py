import sys
import os
sys.path.append(os.getcwd() + '/src')

import json
import math
import time
import argparse
import dataclasses
import PyExpUtils.runner.Slurm as Slurm
import experiment.ExperimentModel as Experiment

from functools import partial
from glob import glob
from PyExpUtils.utils.generator import group
from PyExpUtils.runner.utils import approximate_cost
from utils.results import gather_missing_indices

parser = argparse.ArgumentParser()
parser.add_argument('--cluster', type=str, required=True)
parser.add_argument('--runs', type=int, required=True)
parser.add_argument('-e', type=str, nargs='+', required=True)
parser.add_argument('--entry', type=str, default='src/main.py')
parser.add_argument('--results', type=str, default='./')
parser.add_argument('--debug', action='store_true', default=False)

cmdline = parser.parse_args()

# Expand glob patterns in -e
expanded_paths = []
for pattern in cmdline.e:
    matches = glob(pattern, recursive=True)
    if matches:
        expanded_paths.extend(matches)
    else:
        expanded_paths.append(pattern)
cmdline.e = sorted(expanded_paths)

ANNUAL_ALLOCATION = 724

# -------------------------------
# Generate scheduling bash script
# -------------------------------
cwd = os.getcwd()
project_name = os.path.basename(cwd)

venv_origin = f'{cwd}/venv.tar.xz'
venv = '$SLURM_TMPDIR'

def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def _sbatch_flags(opts, threads):
    flags = [
        f'--account={opts.account}',
        f'--time={opts.time}',
        f'--mem-per-cpu={opts.mem_per_core}',
        f'--output={opts.log_path}',
        f'--ntasks={opts.cores}',
        f'--cpus-per-task={threads}',
    ]
    if isinstance(opts, Slurm.SingleNodeOptions):
        flags.append('--nodes=1')
    return ' '.join(flags)

def _schedule(script, flags, script_name='auto_slurm.sh'):
    with open(script_name, 'w') as f:
        f.write(script)
    os.system(f'sbatch {flags} {script_name}')
    os.remove(script_name)

# the contents of the string below will be the bash script that is scheduled on compute canada
# change the script accordingly (e.g. add the necessary `module load X` commands)
def getJobScript(batch_cmds):
    return f"""#!/bin/bash

#SBATCH --signal=B:SIGTERM@180

cd {cwd}
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 tar -xf {venv_origin} -C {venv}

export MPLBACKEND=TKAgg
export OMP_NUM_THREADS={threads}
{batch_cmds}
wait
    """

# -----------------
# Environment check
# -----------------
# if not cmdline.debug and not os.path.exists(venv_origin):
#     print("WARNING: zipped virtual environment not found at:", venv_origin)
#     print("Make sure to run `scripts/setup_cc.sh` first.")
#     exit(1)

# ----------------
# Scheduling logic
# ----------------
with open(cmdline.cluster) as f:
    cluster_raw = json.load(f)

n_parallel = cluster_raw.pop('parallel', 1)
node_type = cluster_raw.pop('type')
if node_type == 'single_node':
    slurm = Slurm.SingleNodeOptions(**cluster_raw)
elif node_type == 'multi_node':
    slurm = Slurm.MultiNodeOptions(**cluster_raw)
else:
    raise ValueError(f'Unknown scheduling strategy: {node_type}')

threads = slurm.threads_per_task if isinstance(slurm, Slurm.SingleNodeOptions) else 1

# parallel slots per job × indices per slot = total indices per job
n_procs = int(slurm.cores / threads)
groupSize = n_procs * n_parallel

# compute how much time the jobs are going to take
hours, minutes, seconds = slurm.time.split(':')
total_hours = int(hours) + (int(minutes) / 60) + (int(seconds) / 3600)

# gather missing
missing = gather_missing_indices(cmdline.e, cmdline.runs, loader=Experiment.load, base=cmdline.results)

# compute cost
memory = Slurm.memory_in_mb(slurm.mem_per_core)
compute_cost = partial(approximate_cost, cores_per_job=slurm.cores, mem_per_core=memory, hours=total_hours)
cost = sum(compute_cost(math.ceil(len(job_list) / groupSize)) for job_list in missing.values())
perc = (cost / ANNUAL_ALLOCATION) * 100

print(f"Expected to use {cost:.2f} core years, which is {perc:.4f}% of our annual allocation")
if not cmdline.debug:
    input("Press Enter to confirm or ctrl+c to exit")

# start scheduling
for path in missing:
    for g in group(missing[path], groupSize):
        l = list(g)
        batches = list(_chunks(l, n_parallel))
        par_tasks = len(batches)
        sub = dataclasses.replace(slurm, cores=par_tasks)

        runner_base = f'{venv}/.venv/bin/python {cmdline.entry} -e {path} --save_path {cmdline.results} -i '
        batch_lines = [f'{runner_base}{" ".join(map(str, b))} &' for b in batches]
        batch_cmds = '\n'.join(batch_lines)

        script = getJobScript(batch_cmds)
        flags = _sbatch_flags(sub, threads)

        if cmdline.debug:
            print(flags)
            print(script)
            exit()

        _schedule(script, flags)

        # DO NOT REMOVE. This will prevent you from overburdening the slurm scheduler. Be a good citizen.
        time.sleep(2)
