import numpy as np
from skimage.color import rgb2hsv, hsv2rgb
from skimage.measure import label
from skimage.measure import regionprops
from skimage.morphology import closing
from skimage.morphology import dilation
from skimage.morphology import rectangle
from skimage.morphology import remove_small_objects
from skimage.morphology import skeletonize
from skimage.exposure import rescale_intensity
from sklearn.cluster import DBSCAN

from plotvision.utils import to_rgb


def disassemble(plot_image, clustering_threshold, labeling_threshold):

    if len(plot_image.shape) > 2:
        if plot_image.shape[2] == 4:
            plot_image = plot_image[:,:,:3]

        if is_black_on_white(plot_image):
            plot_image = 1 - plot_image
            bow = True
        else:
            bow = False

    if len(plot_image.shape) == 2:
        hue, saturation, volume = plot_image, plot_image, plot_image
    else:
        hue, saturation, volume = hsv_decompose(plot_image)

    hue = rescale_intensity(hue)
    clusters = clusterize_hue(hue, clustering_threshold)
    grid_mask = get_grid_mask(saturation, volume)
    cluster_label, clusters_names, arrows = label_clusters(hue, saturation, clusters,
                                                           labeling_threshold, bow)
    return cluster_label, clusters_names, grid_mask, arrows


def is_black_on_white(img):
    return np.mean(rgb2hsv(img)[2]) > 0.2


def hsv_decompose(plot_image, volume_threshold=0.25):
    plot_image = rgb2hsv(plot_image)
    volume = plot_image[:, :, 2] > volume_threshold
    saturation = (plot_image[:, :, 1] > 0) * volume
    hue = plot_image[:, :, 0] * saturation
    return hue, saturation, volume


def get_grid_mask(saturation_image, volume_image):
    return volume_image * (1 - saturation_image)


def clusterize_hue(hue_image, threshold):
    samples = hue_image[::25].flatten()
    db = DBSCAN(eps=threshold).fit(samples.reshape(-1, 1))
    return calculate_means_of_clusters(samples, db.labels_)


def calculate_means_of_clusters(samples, labels):
    clusters = {}

    for hue_value, label in zip(samples, labels):
        if label == -1:
            continue

        if label in clusters:
            clusters[label] = (clusters[label] + hue_value) / 2
        else:
            clusters[label] = hue_value

    return clusters


def label_clusters(hue_image, saturation_image, clusters, threshold, is_black_on_white):

    def distance(a, b):
        return abs(a - b)

    clusters_img = np.zeros(hue_image.shape, dtype=int)
    clusters_names = {}
    arrows = []

    for label, mean_value in clusters.items():
        plot_mask = (distance(hue_image, mean_value) < threshold) * (saturation_image > 0)
        plot_mask = clear_noise(plot_mask)

        clusters_img[plot_mask] = (plot_mask * (label + 1))[plot_mask]
        clusters_names[label + 1] = to_rgb(hue=mean_value if not is_black_on_white else 255 - mean_value)

    return clusters_img, clusters_names, arrows


def clear_noise(plot_mask):
    return remove_small_objects(skeletonize(plot_mask),
                                min_size=5, connectivity=2)


def cut_round_objects(img):

    img = closing(img, rectangle(3, 1))

    regions = label(img)
    dilated = dilation(regions)

    result = np.zeros(img.shape)
    blobs = []

    for p, q in zip(regionprops(regions), regionprops(dilated)):
        if not (q.perimeter / p.area < 1.8 and p.area > 10):
            result += (regions == p.label) * p.label
        elif p.area > 10:
            blobs.append(p.centroid)

    return result > 0, blobs
