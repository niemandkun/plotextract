import numpy as np
from skimage.color import hsv2rgb


def try_cast(_type):
    def cast_func(_object, default=None):
        try:
            return _type(_object)
        except ValueError:
            return default
    return cast_func


def crop_zeros(npa):
    white = np.argwhere(npa)
    (ystart, xstart), (ystop, xstop) = white.min(0), white.max(0) + 1
    return npa[ystart:ystop, xstart:xstop]


def get_projection(image):
    return (np.sum(image, axis=1), np.sum(image, axis=0))


def to_rgb(hue, saturation=1, volume=1):
    return format_rgb_string(hsv2rgb(np.array([[[hue, saturation, volume]]])).flatten())


def format_rgb_string(rgb):
    return "#" + ("%02X" * 3) % tuple(map(lambda x: round(x * 255), rgb))
