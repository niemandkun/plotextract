import numpy as np
from scipy.interpolate import interp1d


def find_plots_for_arrows(plots, arrows):
    _arrows = {}

    arrows = sorted(arrows, key=lambda x: x[0])

    for arrow in arrows:
        for index, (plot_name, plot) in enumerate(plots):
            arrow_y, arrow_x = arrow[0], arrow[1]
            plot_y = interp1d(*plot)(arrow_x)
            if 0 < plot_y - arrow_y < 5:
                _arrows[index] = np.array([plot_y, arrow_x])

    return _arrows


def __interpolate(x, y):
    f = interp1d(x, y, kind='linear')
    return f, x.min(), x.max()
