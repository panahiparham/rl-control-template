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

setDefaultConference('jmlr')

COLORS = {
    'dqn': 'tab:blue',
    'ln-dqn': 'tab:red',
}

LABELS = {
    'dqn': 'DQN [MLP]',
    'ln-dqn': 'DQN [Layer Norm + MLP]',
}

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
        fig, ax = plt.subplots(1, 1)
        for alg_result in sub_results:
            alg = alg_result.filename

            df = alg_result.load()
            if df is None:
                continue

            rets = {}
            for (tn_refresh,), sub_df in df.group_by('TARGET_NETWORK_FREQUENCY'):

                report = Hypers.select_best_hypers(
                    sub_df,
                    metric='return',
                    prefer=Hypers.Preference.low,
                    time_summary=TimeSummary.mean,
                    statistic=Statistic.mean,
                )

                exp = alg_result.exp

                _, ys = extract_learning_curves(
                    sub_df,
                    hyper_vals=report.best_configuration,
                    metric='return',
                )

                rets[tn_refresh] = np.array([np.mean(y) for y in ys])

            # Draw sensitivity curve for target network refresh frequency
            x = np.array(list(rets.keys()))
            ys = np.array([v[:] for v in rets.values()]).T # use all seeds!
            order = np.argsort(x)
            x = x[order]
            ys = ys[:, order]

            for y in ys:
                ax.plot(x, y, color=COLORS[alg], alpha=0.2, linewidth=0.4)

            res = curve_percentile_bootstrap_ci(
                rng=np.random.default_rng(0),
                y=ys,
                statistic=Statistic.mean,
                iterations=500,
            )


            ax.plot(x, res.sample_stat, label=LABELS[alg], color=COLORS[alg], linewidth=1.5)
            ax.fill_between(x, res.ci[0], res.ci[1], color=COLORS[alg], alpha=0.3)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xlabel('Target Network Refresh Frequency')
        ax.set_xscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels(x)
        ax.set_ylabel('Average Lifetime \n Return')
        ax.set_title(env)

        path = os.path.sep.join(os.path.relpath(__file__).split(os.path.sep)[:-1])
        save(
            save_path=f'{path}/plots',
            plot_name=env,
            f=fig,
            height_ratio=2/3,
        )
