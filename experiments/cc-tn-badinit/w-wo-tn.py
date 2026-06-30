import os
import sys
sys.path.append(os.getcwd() + '/src')

import matplotlib.pyplot as plt
import numpy as np
from experiment.tools import parseCmdLineArgs
from experiment.ExperimentModel import ExperimentModel
from utils.results import ResultCollection

from PyExpPlotting.matplot import save, setDefaultConference
import rlevaluation.hypers as Hypers
from rlevaluation.statistics import Statistic
from rlevaluation.temporal import TimeSummary, extract_learning_curves, curve_percentile_bootstrap_ci
from rlevaluation.config import data_definition
from rlevaluation.interpolation import compute_step_return

setDefaultConference('jmlr')

COLORS = {
    'dqn': {
        1: 'tab:green',
        128: 'tab:blue',
    },
    'ln-dqn': {
        1: 'tab:purple',
        128: 'tab:red',
    },
    'ln-noaff-dqn': {
        1: 'tab:orange',
        128: 'tab:pink',
    },
}

LABELS = {
    'dqn': {
        1: 'DQN w.o. TN',
        128: 'DQN',
    },
    'ln-dqn': {
        1: 'LayerNorm DQN w.o. TN',
        128: 'LayerNorm DQN',
    },
    'ln-noaff-dqn': {
        1: 'No param LayerNorm DQN w.o. TN',
        128: 'No param LayerNorm DQN',
    }
}

WORKING_ENVS = [
    'Acrobot',
    'Cartpole',
    'MountainCar',
]

if __name__ == "__main__":
    path, should_save, save_type = parseCmdLineArgs()

    results = ResultCollection(Model=ExperimentModel, metrics=['return'])
    data_definition(
        hyper_cols=results.get_hyperparameter_columns(),
        seed_col='seed',
        time_col='frame',
        environment_col=None,
        algorithm_col=None,
        make_global=True,
    )

    for env, sub_results in results.groupby_directory(level=2):
        if env not in WORKING_ENVS:
            continue
        fig, ax = plt.subplots(1, 1)
        for alg_result in sub_results:
            alg = alg_result.filename

            df = alg_result.load()
            if df is None:
                continue

            for (tn,), sub_df in df.group_by('TARGET_NETWORK_FREQUENCY'):

                if tn not in [1, 128]:
                    continue

                report = Hypers.select_best_hypers(
                    sub_df,
                    metric='return',
                    prefer=Hypers.Preference.high,
                    time_summary=TimeSummary.mean,
                    statistic=Statistic.mean,
                )

                exp = alg_result.exp
                N_POINTS = 1000
                step_size = exp.total_steps // N_POINTS

                xs, ys = extract_learning_curves(
                    sub_df,
                    hyper_vals=report.best_configuration,
                    metric='return',
                    interpolation=lambda x, y: compute_step_return(x, y, exp.total_steps),
                )

                xs = np.asarray(xs)[:, ::step_size]
                ys = np.asarray(ys)[:, ::step_size]
                assert np.all(np.isclose(xs[0], xs))

                res = curve_percentile_bootstrap_ci(
                    rng=np.random.default_rng(0),
                    y=ys,
                    statistic=Statistic.mean,
                    iterations=500,
                )

                ax.plot(xs[0], res.sample_stat, label=LABELS[alg][tn], color=COLORS[alg][tn], linewidth=1.5)
                ax.fill_between(xs[0], res.ci[0], res.ci[1], color=COLORS[alg][tn], alpha=0.2)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Return')

        if env == 'MountainCar':
            ax.set_ylim(-400, -100)
        elif env == 'Acrobot':
            ax.set_ylim(-200, -50)
        elif env == 'Cartpole':
            ax.set_ylim(200, 500)
        ax.set_title(env)
        ax.legend(loc = 'lower right')

        path = os.path.sep.join(os.path.relpath(__file__).split(os.path.sep)[:-1])
        save(
            save_path=f'{path}/plots',
            plot_name=env,
            f=fig,
            height_ratio=2/3,
        )
