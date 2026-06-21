import pytest
from pathlib import Path
from PyExpUtils.models.ExperimentDescription import loadExperiment
from PyExpUtils.results.tools import getParamsAsDict, getHeader
from utils.results import Result
from experiment.ExperimentModel import ExperimentModel
from conftest import FIXTURE_EXP, make_results_db


def load_fixture_exp():
    return loadExperiment(str(FIXTURE_EXP), ExperimentModel)


def _get_db_path(exp) -> Path:
    return Path(exp.buildSaveContext(0).resolve('results.db'))


def _make_rows_for_params(params: dict, run_id: int, frames=(0, 100, 200)):
    metric_rows = [(f, run_id, float(run_id)) for f in frames]
    meta_row = {'id': run_id, 'seed': run_id // 10, **params}
    return metric_rows, meta_row


class TestResultLoad:
    def test_returns_none_when_no_db(self, project_env):
        exp = load_fixture_exp()
        r = Result(str(FIXTURE_EXP), exp, metrics=['return'])
        assert r.load() is None

    def test_returns_dataframe_for_valid_params(self, project_env):
        exp = load_fixture_exp()
        db_path = _get_db_path(exp)
        n_params = exp.numPermutations()
        header = getHeader(exp)

        metric_rows, metadata_rows = [], []
        for param_id in range(n_params):
            params = getParamsAsDict(exp, param_id, header=header)
            m, meta = _make_rows_for_params(params, run_id=param_id)
            metric_rows.extend(m)
            metadata_rows.append(meta)
        make_results_db(db_path, metric_rows, metadata_rows)

        r = Result(str(FIXTURE_EXP), exp, metrics=['return'])
        df = r.load()
        assert df is not None
        assert len(df) == n_params * 3  # 3 frames per run

    def test_filters_out_stale_param_values(self, project_env):
        exp = load_fixture_exp()
        db_path = _get_db_path(exp)
        n_params = exp.numPermutations()
        header = getHeader(exp)

        metric_rows, metadata_rows = [], []
        # Valid runs
        for param_id in range(n_params):
            params = getParamsAsDict(exp, param_id, header=header)
            m, meta = _make_rows_for_params(params, run_id=param_id)
            metric_rows.extend(m)
            metadata_rows.append(meta)
        # Stale run: TN=16 is NOT in the fixture JSON
        stale_params = {'LR': 0.001, 'TARGET_NETWORK_FREQUENCY': 16}
        m_stale, meta_stale = _make_rows_for_params(stale_params, run_id=99)
        metric_rows.extend(m_stale)
        metadata_rows.append(meta_stale)
        make_results_db(db_path, metric_rows, metadata_rows)

        r = Result(str(FIXTURE_EXP), exp, metrics=['return'])
        df = r.load()
        assert df is not None
        assert 16 not in df['TARGET_NETWORK_FREQUENCY'].to_list()
        assert set(df['TARGET_NETWORK_FREQUENCY'].to_list()).issubset({1, 8})

    def test_duplicate_metadata_rows_filtered_to_json_params(self, project_env):
        exp = load_fixture_exp()
        db_path = _get_db_path(exp)
        header = getHeader(exp)
        params_0 = getParamsAsDict(exp, 0, header=header)

        # id=0: one row with valid params, one stale row with TN=99
        metric_rows = [(0, 0, 1.0), (100, 0, 2.0)]
        metadata_rows = [
            {'id': 0, 'seed': 0, **params_0},
            {'id': 0, 'seed': 0, 'LR': params_0['LR'], 'TARGET_NETWORK_FREQUENCY': 99},
        ]
        make_results_db(db_path, metric_rows, metadata_rows)

        r = Result(str(FIXTURE_EXP), exp, metrics=['return'])
        df = r.load()
        assert df is not None
        assert 99 not in df['TARGET_NETWORK_FREQUENCY'].to_list()
        assert set(df['TARGET_NETWORK_FREQUENCY'].to_list()).issubset({1, 8})
