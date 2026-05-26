# RL Research Template

A simplified version [RL Control Template](https://github.com/andnp/rl-control-template) using pure-jax implementation of DQN from [Jax Research Template](https://github.com/andnp/jax-research-template).

## Local setup

Install dependencies:
```bash
uv sync
```

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

## Cluster setup

TODO!
