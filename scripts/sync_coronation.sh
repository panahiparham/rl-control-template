#!/usr/bin/env bash
exp="${1:?Usage: sync_coronation.sh <experiment>}"
rsync -az coronation:/local/scratch1/parham1/lab/projects/rl-control-template/results/"${exp}" ./results/
