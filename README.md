# RL Research Template

A minimal fork of [RL Control Template](https://github.com/andnp/rl-control-template) to be used within [Jax Research Template](https://github.com/andnp/jax-research-template).


# Mono-repo setup
TODO

# Project setup
TODO

# Installing dependencies
TODO


Install dependencies:
```bash
uv sync
```

# Experiment workflow
Run a single experiment:
```bash
uv run src/main.py -e experiments/example/MountainCar/dqn.json -i 0
```

Run a batch of experiments locally:
```bash
uv run scripts/local.py --runs 10 -e experiments/example/**/*.json
```

Plot results:
```bash
uv run experiments/example/learning_curve.py
```

# Algorithm development workflow
TODO


# Automated Tests
TODO

## Backlog

- [X] Local setup
- [X] Run individual experiments locally
- [x] Reproduce results within mono-repo
- [x] Fix missing idx issue when changing json files
- [ ] Run vmaped seeds locally
- [ ] Run vmaped over non-static hypers locally
- [ ] Rework data collector to collect data from inside the jitted region
- [ ] Setup and test running on cluster
- [ ] Support for Atari
- [ ] Support for Craftax
- [ ] Support for Minigrid
- [ ] Support for checkpointing with preemption
- [ ] Support for tracknig performace during runs
