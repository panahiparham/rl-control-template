import pytest
from pathlib import Path
from PyExpUtils.models.ExperimentDescription import loadExperiment
from PyExpUtils.results.tools import getParamsAsDict, getHeader
from utils.results import detect_missing_indices
from experiment.ExperimentModel import ExperimentModel
from conftest import FIXTURE_EXP, make_metadata_db


def load_fixture_exp():
    return loadExperiment(str(FIXTURE_EXP), ExperimentModel)


def _get_db_path(exp, project_env) -> Path:
    return Path(exp.buildSaveContext(0).resolve('results.db'))


class TestDetectMissingIndices:
    def test_no_db_yields_all_indices(self, project_env):
        exp = load_fixture_exp()
        runs = 3
        missing = list(detect_missing_indices(exp, runs))
        assert len(missing) == exp.numPermutations() * runs

    def test_all_present_yields_nothing(self, project_env):
        exp = load_fixture_exp()
        db_path = _get_db_path(exp, project_env)
        runs = 2
        n_params = exp.numPermutations()
        header = getHeader(exp)

        rows = []
        for seed in range(runs):
            for param_id in range(n_params):
                params = getParamsAsDict(exp, param_id, header=header)
                idx = seed * n_params + param_id
                rows.append({'id': idx, 'seed': seed, **params})
        make_metadata_db(db_path, rows)

        missing = list(detect_missing_indices(exp, runs))
        assert missing == []

    def test_missing_seeds_for_one_param(self, project_env):
        exp = load_fixture_exp()
        db_path = _get_db_path(exp, project_env)
        runs = 3
        n_params = exp.numPermutations()
        header = getHeader(exp)

        # Insert all seeds for param_id=0 only
        rows = []
        for seed in range(runs):
            params = getParamsAsDict(exp, 0, header=header)
            idx = seed * n_params + 0
            rows.append({'id': idx, 'seed': seed, **params})
        make_metadata_db(db_path, rows)

        missing = sorted(detect_missing_indices(exp, runs))
        expected_missing = sorted(
            seed * n_params + pid
            for seed in range(runs)
            for pid in range(1, n_params)
        )
        assert missing == expected_missing

    def test_partial_seeds_detected(self, project_env):
        exp = load_fixture_exp()
        db_path = _get_db_path(exp, project_env)
        runs = 4
        n_params = exp.numPermutations()
        header = getHeader(exp)

        # seed 0 and 2 present for all params; seeds 1 and 3 absent
        rows = []
        for seed in [0, 2]:
            for param_id in range(n_params):
                params = getParamsAsDict(exp, param_id, header=header)
                idx = seed * n_params + param_id
                rows.append({'id': idx, 'seed': seed, **params})
        make_metadata_db(db_path, rows)

        missing = sorted(detect_missing_indices(exp, runs))
        expected_missing = sorted(
            seed * n_params + pid
            for seed in [1, 3]
            for pid in range(n_params)
        )
        assert missing == expected_missing
