from itertools import product

import numpy as np
from PIL import Image
from pytesseract import image_to_string
from scipy.stats import mode
from skimage.filters import threshold_otsu
from skimage.transform import pyramid_expand
from sklearn.linear_model import RANSACRegressor, LinearRegression

from plotvision.utils import crop_zeros, get_projection
from plotvision.utils import try_cast


class GridLine:
    def __init__(self, min, max):
        self.min = min
        self.max = max


class GridPart:
    def __init__(self, inner, outer):
        self.inner = inner
        self.outer = outer


class Grid:
    def __init__(self, vert, horiz):
        self.vert = vert
        self.horiz = horiz


MY_GRID = Grid(
    vert=GridPart(
        inner=GridLine(50, 100),
        outer=GridLine(100, 240)),
    horiz=GridPart(
        inner=GridLine(200, 500),
        outer=GridLine(500, 1000)))


class Transform:
    def __init__(self, bias, scale):
        self.bias = bias
        self.scale = scale

    def apply(self, x, y):
        x = (x * self.scale[0]) + self.bias[0]
        y = (y * self.scale[1]) + self.bias[1]
        return x, y

    @staticmethod
    def from_(coordinate_system):
        return TransformBuilder(coordinate_system)


class TransformBuilder:
    def __init__(self, _from):
        self._from = _from

    def to(self, coordinate_system):
        return TransformBuilder.__create_transform(self._from, coordinate_system)

    @staticmethod
    def __create_transform(_from, _to):
        scale = _to.step / _from.step
        to_x, from_x = TransformBuilder.first_not_none(_to.x_ticks, _from.x_ticks)
        to_y, from_y = TransformBuilder.first_not_none(_to.y_ticks, _from.y_ticks)
        bias = np.array([to_x - scale[0] * from_x, to_y - scale[1] * from_y])
        return Transform(bias, scale)

    @staticmethod
    def first_not_none(*iterables):
        for vector in product(*iterables):
            for i in range(len(vector)):
                if vector[i] is None:
                    break
            else:
                return vector
        return None


class VertexValidator:
    def __init__(self, regression_model, ):
        self.regressor = regression_model

    def is_data_valid(self, x, y, epsilon=1):
        return abs(self.regressor.predict(x) - y) < epsilon

    def try_correct(self, x, y):
        new_y = self.regressor.predict(y)
        if (self.is_data_valid(x, new_y)):
            return x, new_y
        else:
            return None, None


class CoordinateSystem(object):
    def __init__(self, x_ticks, y_ticks):
        self.x_ticks, self.y_ticks = self.validate(x_ticks, y_ticks)
        self.step = np.array([self.get_step(t) for t in [self.x_ticks, self.y_ticks]])

    def validate(self, x_ticks, y_ticks):
        x_ticks, y_ticks = self.remove_none(x_ticks, y_ticks)
        validator = self.make_regression(x_ticks, y_ticks)

        new_x_ticks = []
        new_y_ticks = []

        for x, y in zip(x_ticks, y_ticks):
            if validator.is_data_valid(x, y):
                new_x_ticks.append(x)
                new_y_ticks.append(y)
            else:
                x, y = validator.try_correct(x, y)
                new_x_ticks.append(x)
                new_y_ticks.append(y)

        return new_x_ticks, new_y_ticks

    def make_regression(self, x_ticks, y_ticks):
        model = RANSACRegressor(LinearRegression())
        model.fit(np.array(x_ticks).reshape(-1, 1), np.array(y_ticks).reshape(-1, 1))
        return VertexValidator(model)

    def remove_none(self, iter1, iter2):
        new_iter1 = []
        new_iter2 = []
        for x, y in zip(iter1, iter2):
            if x is not None and y is not None:
                new_iter1.append(x)
                new_iter2.append(y)
        return new_iter1, new_iter2

    def get_step(self, ticks_labels):
        step = []
        for i in xrange(len(ticks_labels) - 1):
            if ticks_labels[i+1] is not None and ticks_labels[i] is not None:
                step.append(round(ticks_labels[i+1]-ticks_labels[i], 4))
        return mode(step).mode[0]


class FigureCoordinateSystem(CoordinateSystem):
    def __init__(self, grid_projection, grid_image):
        axes = self.__get_axes(*grid_projection)
        left_labels, bottom_labels = self.__crop_axes_labels(grid_image, axes)
        left_coords = self.__get_numbers(left_labels)
        bottom_coords = self.__get_numbers(bottom_labels)
        super(FigureCoordinateSystem, self).__init__(bottom_coords, left_coords)

    def __get_axes(self, sum_horiz, sum_vert):
        bottom_axis = np.argwhere(sum_horiz > MY_GRID.horiz.inner.min).max()
        left_axis = np.argwhere(sum_vert > MY_GRID.vert.inner.min).min()
        return left_axis, bottom_axis

    def __crop_axes_labels(self, grid_image, axes):
        left_axis, bottom_axis = axes
        bottom_labels = grid_image[bottom_axis + 1:, left_axis + 1:]
        left_labels = grid_image[:, :left_axis]
        return left_labels, bottom_labels

    def __get_numbers(self, ticks_image):
        text = self.__get_plain_text(ticks_image)
        return [try_cast(float)(s) for s in text.split()]

    def __get_plain_text(self, ticks_image):
        grd = ticks_image
        grd = pyramid_expand(np.uint8(grd), upscale=2)
        thr = threshold_otsu(grd)
        grd = grd > thr + 0.0002
        grd = crop_zeros(grd)
        return image_to_string(Image.fromarray(np.uint8(grd)), config='-psm 6')


class PictureCoordinateSystem(CoordinateSystem):
    def __init__(self, projection):
        left_coords, bottom_coords = self.__get_grids_coordinates(*projection)
        super(PictureCoordinateSystem, self).__init__(bottom_coords, left_coords)

    def __get_grids_coordinates(self, sum_horiz, sum_vert):
        horiz_grids = np.argwhere((sum_horiz > MY_GRID.horiz.inner.min) *
                                  (sum_horiz < MY_GRID.horiz.inner.max)).flatten()

        vert_grids = np.argwhere((sum_vert > MY_GRID.vert.inner.min) *
                                 (sum_vert < MY_GRID.vert.inner.max)).flatten()
        return horiz_grids, vert_grids


def parse_grid(grid_image):
    projection = get_projection(grid_image)
    return Transform.from_(PictureCoordinateSystem(projection))\
        .to(FigureCoordinateSystem(projection, grid_image))