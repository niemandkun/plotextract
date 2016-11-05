import numpy as np
from skimage.segmentation import active_contour

from plotvision.arrows import find_plots_for_arrows

POLYNOMIAL_FIT_DEGREE = 10
MINIMAL_PLOT_AREA = 0.0006


def get_series(plot_labels, label_names, arrows=None, grid_transform=None,
               min_area=MINIMAL_PLOT_AREA):
    """
    yields (color, (X, Y)) -- collection of points for each series
    """
    plots = list(__get_plots(plot_labels, label_names, min_area))
    arrows = find_plots_for_arrows(plots, arrows)

    for index, (plot_name, plot) in enumerate(plots):
        plot_minimum = __get_minimum(arrows, index, plot, grid_transform)
        if grid_transform is not None:
            plot = grid_transform.apply(*plot)
        yield plot_name, plot, plot_minimum


def __get_plots(plot_labels, label_names, min_area):
    area_threshold = __get_min_area_absolute(plot_labels, min_area)

    for label in range(1, plot_labels.max() + 1):
        plot_pixels = (plot_labels == label)

        if len(np.argwhere(plot_pixels)) > area_threshold:
            plot = __get_series(plot_pixels)
            plot_name = label_names[label]
            yield plot_name, plot


def __get_minimum(arrows, label, plot, transform=None):
    x, y = __find_minimum(arrows, label, plot)
    if transform is not None:
        x, y = transform.apply(x, y)
    return x, y


def __find_minimum(arrows, label, plot):
    if label in arrows:
        minimum = arrows[label]
        return minimum[1], minimum[0]
    else:
        x, y = plot

        argmin = np.where(y == y.min())[0]
        # argmin = np.where(y == y.max())[0]  # USE FOR VALIDATION PURPOSES!

        argmin = argmin[len(argmin) // 2]
        return x[argmin], y[argmin]


def __get_min_area_absolute(image, min_area_percentage):
    image_area = image.shape[0] * image.shape[1]
    return image_area * min_area_percentage


def __get_series(plot_mask):
    xy = np.argwhere(plot_mask.T)
    x, y = xy[:, 0], xy[:, 1]

    estimate_function = __polyfit(x, y)
    snake_domain = np.arange(x[0], x[-1], 3)

    init = np.array([snake_domain, estimate_function(snake_domain)]).T

    snake = active_contour(plot_mask, init, bc='fixed',
                           w_line=1, w_edge=0,
                           alpha=0.5,
                           beta=2,
                           gamma=1)

    x, y = snake[:,0], snake[:,1]

    return x, y


def __polyfit(x, y):
    return np.poly1d(np.polyfit(x, y, POLYNOMIAL_FIT_DEGREE))