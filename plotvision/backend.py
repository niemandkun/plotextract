from traceback import print_exc

from plotvision.disassemble import disassemble
from plotvision.gridextract import parse_grid, Transform
from plotvision.plotextract import get_series


DEFAULT_TRANSFORM = Transform(bias=[0, 0], scale=[1, -1])
DEFAULT_CLUSTERING_THRESHOLD = 0.0015
DEFAULT_LABELING_THRESHOLD = 0.0015


def try_get_transform(default_transform, grid_mask, verbose):
    try:
        return parse_grid(grid_mask)
    except Exception:
        if verbose:
            print_exc()
        print "WARNING: cannot parse grid, will use default transform instead"
        return default_transform


def __main(plot_image, clustering_threshold, labeling_threshold, verbose, apply_transform):
    plot_label, label_names, grid_mask, arrows = disassemble(plot_image, clustering_threshold, labeling_threshold)

    if apply_transform:
        transform = try_get_transform(DEFAULT_TRANSFORM, grid_mask, verbose)
    else:
        transform = DEFAULT_TRANSFORM

    if verbose:
        print "Built transform:"
        print 'bias:', transform.bias, '\nscale:', transform.scale

    return plot_label, label_names, transform, arrows


def img_to_data(plot_image, clustering_threshold=DEFAULT_CLUSTERING_THRESHOLD,
                labeling_threshold=DEFAULT_LABELING_THRESHOLD, verbose=False,
                apply_transform=True):
    """ yields (color, (X, Y)) -- collection of points for each series
        if interp_step is None, no interpolation is performed
        if interp_step is int, interpolation is performed with specified step
    """
    plot_label, label_names, transform, arrows = __main(plot_image, clustering_threshold,
                                                labeling_threshold, verbose, apply_transform)

    return list(get_series(plot_label, label_names, arrows, transform))
