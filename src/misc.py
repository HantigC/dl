from collections import defaultdict

import matplotlib.pyplot as plt


def transpose_history(fit_history):
    epoch_sumarries = defaultdict(lambda: defaultdict(list))

    def _transpose(stage, epoch_sumarry):
        for name, val in epoch_sumarry[stage].items():
            epoch_sumarries[name][stage].append(val)

    for epoch_summary in fit_history:
        _transpose("train", epoch_summary)
        _transpose("eval", epoch_summary)
    return {k: dict(v) for k, v in epoch_sumarries.items()}


def plot_summary(summary):
    for name, values_mapp in summary.items():
        fig, ax = plt.subplots()
        ax.set_title(name)
        ax.set_xlabel("epoch")
        for stage_name, values in values_mapp.items():
            ax.plot(values, label=stage_name)

        ax.legend()
        plt.show()
