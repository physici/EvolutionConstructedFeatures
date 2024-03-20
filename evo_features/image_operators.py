# -*- coding: utf-8 -*-
"""
Created on Thu May 11 14:48:27 2023

@author: rainer.jacob

Provides the different filters as objects that have a method to mutate the
individual parameters

Module applies gpu versions of numpy/scipy (cupy) and skimage (cucim)
"""

import skimage.draw as draw
from numpy.typing import NDArray
from typing import Tuple
from random import uniform, randrange, choice, random

import skimage.filters as filters
import skimage.morphology as morphs
import skimage.feature as feature
import skimage.transform as transform
import skimage.segmentation as segmentation
from skimage.transform import hough_line, hough_line_peaks
from skimage.transform import hough_circle, hough_circle_peaks
from skimage import exposure, util

from skimage.util import img_as_float

from scipy import ndimage as ndi
from scipy.fft import fft2, fftshift
import numpy as np
from skimage.feature import hog, local_binary_pattern


def transform_uint8(img: NDArray[np.float16]) -> NDArray[np.uint8]:
    """
    Convert to uint8

    Parameters
    ----------
    img : NDArray[np.float16]
        Input image.

    Returns
    -------
    img_out : NDArray[np.uint8]
        Output image.

    """
    try:
        assert img.size > 0
        assert len(img.shape) == 2
    except AssertionError:
        img_out: NDArray[np.uint8] = np.zeros(shape=(2, 2), dtype="uint8")
        return img_out

    imin = img.min()
    imax = img.max()

    tmp = imax - imin
    if tmp == 0:
        tmp = 1
    a = 255 / tmp
    b = 255 - a * imax
    img_out = (a * img + b).astype(np.uint8)

    return img_out


class Sobel:
    def __init__(self) -> None:
        # initization
        pass

    def mutate(self, mutation_probability: float = 0.3) -> None:
        # mutation
        pass

    def apply(self, img: NDArray[np.float16]) -> NDArray[np.uint8]:
        # apply filter
        try:
            assert img.size > 0
            assert len(img.shape) == 2
        except AssertionError:
            image: NDArray[np.uint8] = np.zeros(shape=(2, 2), dtype="uint8")
            return image

        img_out: NDArray[np.float16] = filters.sobel(img)
        return transform_uint8(img_out)


class Gabor:
    def __init__(self) -> None:
        # initization
        self._frequency = uniform(1, 32)
        self._theta = uniform(0, 2 * np.pi)
        self._sigmax = uniform(0, 1)
        self._sigmay = uniform(0, 1)

    def mutate(self, mutation_probability: float = 0.3) -> None:
        # mutation
        if random() < mutation_probability:
            self._frequency = uniform(1, 32)
        if random() < mutation_probability:
            self._theta = uniform(0, 2 * np.pi)
        if random() < mutation_probability:
            self._sigmax = uniform(0, 1)
        if random() < mutation_probability:
            self._sigmay = uniform(0, 1)

    def apply(self, img: NDArray[np.float16]) -> NDArray[np.uint8]:
        # apply filter
        try:
            assert img.size > 0
            assert len(img.shape) == 2
        except AssertionError:
            image: NDArray[np.uint8] = np.zeros(shape=(2, 2), dtype="uint8")
            return image

        kernel = filters.gabor_kernel(
            self._frequency, self._theta, self._sigmax, self._sigmay
        )
        tmp: NDArray[np.float16] = img_as_float(img)
        val: float = tmp.std()
        if val == 0:
            val = 1
        tmp = (tmp - tmp.mean()) / val  # TODO fix
        img_out: NDArray[np.float16] = np.sqrt(
            ndi.convolve(tmp, np.real(kernel), mode="wrap") ** 2
            + ndi.convolve(tmp, np.imag(kernel), mode="wrap") ** 2
        )

        return transform_uint8(img_out)


class GaussianBlur:
    def __init__(self) -> None:
        # initization
        self._sigma = uniform(0, 50)

    def mutate(self, mutation_probability: float = 0.3) -> None:
        # mutation
        if random() < mutation_probability:
            self._sigma = uniform(0, 50)

    def apply(self, img: NDArray[np.float16]) -> NDArray[np.uint8]:
        # apply filter
        try:
            assert img.size > 0
            assert len(img.shape) == 2
        except AssertionError:
            image: NDArray[np.uint8] = np.zeros(shape=(2, 2), dtype="uint8")
            return image

        img_out: NDArray[np.float16] = filters.gaussian(img, sigma=self._sigma)
        return transform_uint8(img_out)


class Erode:
    def __init__(self) -> None:
        # initization
        self._iterations = int(uniform(1, 10))

    def mutate(self, mutation_probability: float = 0.3) -> None:
        # mutation
        if random() < mutation_probability:
            self._iterations = int(uniform(1, 10))

    def apply(self, img: NDArray[np.float16]) -> NDArray[np.uint8]:
        # apply filter
        try:
            assert img.size > 0
            assert len(img.shape) == 2
        except AssertionError:
            image: NDArray[np.uint8] = np.zeros(shape=(2, 2), dtype="uint8")
            return image

        img_out: NDArray[np.float16] = img
        for idx in range(self._iterations):
            img_out = morphs.erosion(img_out)
        return transform_uint8(img_out)


class Dilate:
    def __init__(self) -> None:
        # initization
        self._iterations = int(uniform(1, 10))

    def mutate(self, mutation_probability: float = 0.3) -> None:
        # mutation
        if random() < mutation_probability:
            self._iterations = int(uniform(1, 10))

    def apply(self, img: NDArray[np.float16]) -> NDArray[np.uint8]:
        # apply filter
        try:
            assert img.size > 0
            assert len(img.shape) == 2
        except AssertionError:
            image: NDArray[np.uint8] = np.zeros(shape=(2, 2), dtype="uint8")
            return image

        img_out: NDArray[np.float16] = img
        for idx in range(self._iterations):
            img_out = morphs.dilation(img_out)
        return transform_uint8(img_out)


class Canny:
    def __init__(self) -> None:
        # initization
        self._sigma = uniform(0, 10)
        self._lwthr = uniform(0, 210)
        self._hgthr = uniform(self._lwthr, 254)

    def mutate(self, mutation_probability: float = 0.3) -> None:
        # mutation
        if random() < mutation_probability:
            self._sigma = uniform(0, 10)
        if random() < mutation_probability:
            self._lwthr = uniform(0, 210)
        if random() < mutation_probability:
            self._hgthr = uniform(self._lwthr, 254)

    def apply(self, img: NDArray[np.float16]) -> NDArray[np.uint8]:
        try:
            assert img.size > 0
            assert len(img.shape) == 2
        except AssertionError:
            image: NDArray[np.uint8] = np.zeros(shape=(2, 2), dtype="uint8")
            return image

        # apply filter
        img_out: NDArray[np.float16] = feature.canny(
            img,
            sigma=self._sigma,
            low_threshold=self._lwthr,
            high_threshold=self._hgthr,
        )
        img_out = img_out.astype(np.float16)
        return transform_uint8(img_out)


class AdaptThreshold:
    def __init__(self) -> None:
        # initization
        self._blocksize = randrange(1, 31, 2)
        self._offset = uniform(1, 254)

    def mutate(self, mutation_probability: float = 0.3) -> None:
        # mutation
        if random() < mutation_probability:
            self._blocksize = randrange(1, 31, 2)
        if random() < mutation_probability:
            self._offset = uniform(1, 254)

    def apply(self, img: NDArray[np.float16]) -> NDArray[np.uint8]:
        try:
            assert img.size > 0
            assert len(img.shape) == 2
        except AssertionError:
            image: NDArray[np.uint8] = np.zeros(shape=(2, 2), dtype="uint8")
            return image

        # apply filter
        mask: NDArray[np.float16] = filters.threshold_local(
            img, block_size=self._blocksize, offset=self._offset
        )
        img_out: NDArray[np.float16] = (img > mask).astype(np.float16)
        return transform_uint8(img_out)


class Normalize:
    def __init__(self) -> None:
        # initization
        pass

    def mutate(self, mutation_probability: float = 0.3) -> None:
        # mutation
        pass

    def apply(self, img: NDArray[np.float16]) -> NDArray[np.uint8]:
        # apply filter
        try:
            assert img.size > 0
            assert len(img.shape) == 2
        except AssertionError:
            image: NDArray[np.uint8] = np.zeros(shape=(2, 2), dtype="uint8")
            return image

        img_min = img.min()
        img_max = img.max()
        if img_max == 0:
            img_max = 1
        img_out: NDArray[np.float16] = (img - img_min) / img_max * 255
        return transform_uint8(img_out)


class HoughLines:
    def __init__(self) -> None:
        # initization
        pass

    def mutate(self, mutation_probability: float = 0.3) -> None:
        # mutation
        pass

    def apply(self, img: NDArray[np.float16]) -> NDArray[np.uint8]:
        # apply filter
        try:
            assert img.size > 0
            assert len(img.shape) == 2
        except AssertionError:
            image: NDArray[np.uint8] = np.zeros(shape=(2, 2), dtype="uint8")
            return image

        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
        h, theta, d = hough_line(img, tested_angles)
        h, theta, d = hough_line_peaks(h, theta, d)

        img_out: NDArray[np.float16] = np.zeros(img.shape, dtype=np.float16)
        for angle, dist in zip(theta, d):
            # horizontal
            if angle == 0:
                x0, y0 = 0, int(dist)
                x1, y1 = img_out.shape[0] - 1, int(dist)
                coords = [x0, y0, x1, y1]
                coords = [int(x) for x in coords]
            # vertical
            if angle % np.pi == 0:
                x0, y0 = int(dist), 0
                x1, y1 = int(dist), img_out.shape[1] - 1
                coords = [x0, y0, x1, y1]
                coords = [int(x) for x in coords]
            else:
                xr, yr = dist * np.array([np.cos(angle), np.sin(angle)])
                slope = -1 / np.tan(angle)
                b = yr - slope * xr

                x0 = round(-b / slope)
                y0 = int(0)

                x1 = int(0)
                y1 = round(b)

                x2 = img_out.shape[1] - 1
                y2 = round(slope * x2 + b)

                y3 = img_out.shape[0] - 1
                x3 = round((y3 - b) / slope)

                points = [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]
                if points[0] == points[1]:
                    points[1] = (-1, -1)
                if points[2] == points[3]:
                    points[3] = (-1, -1)

                points = list(set(points))

                coords = []
                for elem in points:
                    x, y = elem
                    if (0 <= x < img_out.shape[1]) and (
                        0 <= y < img_out.shape[0]
                    ):
                        coords.append(int(y))
                        coords.append(int(x))

            if len(coords) == 4:
                rr, cc, val = draw.line_aa(*coords)
                try:
                    img_out[rr, cc] = val * 255
                except IndexError:
                    pass

        return transform_uint8(img_out)


class HoughCircles:
    def __init__(self) -> None:
        # initization
        self._lwradius = int(uniform(0, 200))
        self._hgradius = int(uniform(self._lwradius, 254))

    def mutate(self, mutation_probability: float = 0.3) -> None:
        # mutation
        if random() < mutation_probability:
            self._lwradius = int(uniform(0, 200))
        if random() < mutation_probability:
            self._hgradius = int(uniform(self._lwradius, 254))

    def apply(self, img: NDArray[np.float16]) -> NDArray[np.uint8]:
        # apply filter
        try:
            assert img.size > 0
            assert len(img.shape) == 2
        except AssertionError:
            image: NDArray[np.uint8] = np.zeros(shape=(2, 2), dtype="uint8")
            return image

        hough_radii = np.arange(self._lwradius, self._hgradius, 2)
        hough_res = hough_circle(img, hough_radii)
        accums, cx, cy, radii = hough_circle_peaks(
            hough_res, hough_radii, total_num_peaks=3
        )

        img_out: NDArray[np.float16] = np.zeros(img.shape, dtype=np.float16)
        for center_y, center_x, radius in zip(cy, cx, radii):
            circy, circx = draw.circle_perimeter(
                center_y, center_x, radius, shape=img_out.shape
            )
            img_out[circy, circx] = 255
        return transform_uint8(img_out)


class MedianBlur:
    def __init__(self) -> None:
        # initization
        self._disksize = int(uniform(0, 50))

    def mutate(self, mutation_probability: float = 0.3) -> None:
        # mutation
        if random() < mutation_probability:
            self._disksize = int(uniform(0, 50))

    def apply(self, img: NDArray[np.float16]) -> NDArray[np.uint8]:
        # apply filter
        try:
            assert img.size > 0
            assert len(img.shape) == 2
        except AssertionError:
            image: NDArray[np.uint8] = np.zeros(shape=(2, 2), dtype="uint8")
            return image

        img_out: NDArray[np.float16] = np.zeros(img.shape, dtype=np.float16)
        img_out = filters.median(img, morphs.disk(self._disksize))
        return transform_uint8(img_out)


class CensusTransform:
    """
    Census tranformation according to
    https://www.appsloveworld.com/opencv/100/13/census-transform-in-python-opencv

    https://en.wikipedia.org/wiki/Census_transform
    """

    def __init__(self) -> None:
        # initization
        pass

    def mutate(self, mutation_probability: float = 0.3) -> None:
        # mutation
        pass

    def apply(self, img: NDArray[np.uint8]) -> NDArray[np.uint8]:
        # apply filter
        try:
            assert len(img.shape) == 2
            assert img.shape[0] >= 4
            assert img.shape[1] >= 4
        except AssertionError:
            image: NDArray[np.uint8] = np.zeros(shape=(2, 2), dtype="uint8")
            return image

        h, w = img.shape

        img_out: NDArray[np.uint8] = np.zeros((h - 2, w - 2), dtype=np.uint8)
        # center pixels
        cp = img[1 : h - 1, 1 : w - 1]
        # offsets
        offsets = [
            (u, v) for v in range(3) for u in range(3) if not u == 1 == v
        ]

        for u, v in offsets:
            img_out = (img_out << 1) | (
                img[v : v + h - 2, u : u + w - 2] >= cp
            )
        return transform_uint8(img_out)


class HarrisCornerStrength:
    def __init__(self) -> None:
        # initization
        self._sensitivity = uniform(0, 0.3)
        self._eps = uniform(1e-8, 1e-5)
        self._sigma = uniform(1e-3, 50)

    def mutate(self, mutation_probability: float = 0.3) -> None:
        # mutation
        if random() < mutation_probability:
            self._sensitivity = uniform(0, 0.3)
        if random() < mutation_probability:
            self._eps = uniform(1e-8, 1e-5)
        if random() < mutation_probability:
            self._sigma = uniform(1e-3, 50)

    def apply(self, img: NDArray[np.float16]) -> NDArray[np.uint8]:
        # apply filter
        try:
            assert len(img.shape) == 2
            assert img.shape[0] >= 4
            assert img.shape[1] >= 4
        except AssertionError:
            image: NDArray[np.uint8] = np.zeros(shape=(2, 2), dtype="uint8")
            return image

        img_out: NDArray[np.float16] = np.zeros(img.shape, dtype=np.float16)

        corners = feature.corner_peaks(
            feature.corner_harris(
                img, self._sensitivity, self._eps, self._sigma
            ),
            min_distance=1,
        )

        for corner in corners:
            img_out[corner[0], corner[1]] = 255

        return transform_uint8(img_out)


class ContrastStretch:
    def __init__(self) -> None:
        # initization
        pass

    def mutate(self, mutation_probability: float = 0.3) -> None:
        # mutation
        pass

    def apply(self, img: NDArray[np.float16]) -> NDArray[np.uint8]:
        # apply filter
        try:
            assert img.size > 0
            assert len(img.shape) == 2
        except AssertionError:
            image: NDArray[np.uint8] = np.zeros(shape=(2, 2), dtype="uint8")
            return image

        img_out: NDArray[np.float16] = np.zeros(img.shape, dtype=np.float16)
        p2, p98 = np.percentile(img, (2, 98))
        img_out = exposure.rescale_intensity(img, in_range=(p2, p98))
        return transform_uint8(img_out)


class HistEqualization:
    def __init__(self) -> None:
        # initization
        self._cliplimit = uniform(0, 1)

    def mutate(self, mutation_probability: float = 0.3) -> None:
        # mutation
        if random() < mutation_probability:
            self._cliplimit = uniform(0, 1)

    def apply(self, img: NDArray[np.float16]) -> NDArray[np.uint8]:
        # apply filter
        try:
            assert img.size > 0
            assert len(img.shape) == 2
        except AssertionError:
            image: NDArray[np.uint8] = np.zeros(shape=(2, 2), dtype="uint8")
            return image

        img_out: NDArray[np.float16] = exposure.equalize_adapthist(
            img, clip_limit=self._cliplimit
        )
        return transform_uint8(img_out)


class GaussianGradient:
    def __init__(self) -> None:
        # initization
        self._alpha = uniform(1, 50)
        self._sigma = uniform(1e-3, 50)

    def mutate(self, mutation_probability: float = 0.3) -> None:
        # mutation
        if random() < mutation_probability:
            self._alpha = uniform(1, 50)
        if random() < mutation_probability:
            self._sigma = uniform(1e-3, 50)

    def apply(self, img: NDArray[np.float16]) -> NDArray[np.uint8]:
        # apply filter
        try:
            assert img.size > 0
            assert len(img.shape) == 2
        except AssertionError:
            image: NDArray[np.uint8] = np.zeros(shape=(2, 2), dtype="uint8")
            return image

        img_out: NDArray[np.float16] = np.zeros(img.shape, dtype=np.float16)
        img_out = segmentation.inverse_gaussian_gradient(
            img, self._alpha, self._sigma
        )

        return transform_uint8(img_out)


class IntegralImage:
    def __init__(self) -> None:
        # initization
        pass

    def mutate(self, mutation_probability: float = 0.3) -> None:
        # mutation
        pass

    def apply(self, img: NDArray[np.float16]) -> NDArray[np.uint8]:
        # apply filter
        try:
            assert img.size > 0
            assert len(img.shape) == 2
        except AssertionError:
            image: NDArray[np.uint8] = np.zeros(shape=(2, 2), dtype="uint8")
            return image

        img_out: NDArray[np.float16] = np.zeros(img.shape, dtype=np.float16)
        img_out = transform.integral_image(img)
        return transform_uint8(img_out)


class LaplaceEdge:
    def __init__(self) -> None:
        # initization
        self._ksize = round(uniform(1, 50))
        self._ksize = 3

    def mutate(self, mutation_probability: float = 0.3) -> None:
        # mutation
        if random() < mutation_probability:
            self._ksize = round(uniform(1, 50))
        if random() < mutation_probability:
            self._ksize = round(uniform(1, 50))

    def apply(self, img: NDArray[np.float16]) -> NDArray[np.uint8]:
        # apply filter
        try:
            assert img.size > 0
            assert len(img.shape) == 2
        except AssertionError:
            image: NDArray[np.uint8] = np.zeros(shape=(2, 2), dtype="uint8")
            return image

        img_out: NDArray[np.float16] = np.zeros(img.shape, dtype=np.float16)
        try:
            img_out = filters.laplace(
                img, self._ksize
            )  # TODO fix kernel size limitation
        except ValueError:
            return transform_uint8(img_out)
        return transform_uint8(img_out)


class FourierTransform:
    def __init__(self) -> None:
        # initization
        pass

    def mutate(self, mutation_probability: float = 0.3) -> None:
        # mutation
        pass

    def apply(self, img: NDArray[np.float16]) -> NDArray[np.uint8]:
        # apply filter
        try:
            assert img.size > 0
            assert len(img.shape) == 2
        except AssertionError:
            image: NDArray[np.uint8] = np.zeros(shape=(2, 2), dtype="uint8")
            return image

        img_out: NDArray[np.float16] = np.zeros(img.shape, dtype=np.float16)
        img_out = np.abs(fftshift(fft2(img)))

        return transform_uint8(img_out)


class Scale:
    def __init__(self) -> None:
        # initization
        self._factor_width = uniform(0.3, 3)
        self._factor_height = uniform(0.3, 3)

    def mutate(self, mutation_probability: float = 0.3) -> None:
        # mutation
        if random() < mutation_probability:
            self._factor_width = uniform(0.3, 3)
        if random() < mutation_probability:
            self._factor_height = uniform(0.3, 3)

    def apply(self, img: NDArray[np.float16]) -> NDArray[np.float16]:
        # apply filter
        try:
            assert img.size > 0
            assert len(img.shape) == 2
        except AssertionError:
            image: NDArray[np.uint8] = np.zeros(shape=(2, 2), dtype="uint8")
            return image

        # check resulting image size to prevent out of memory error
        height, width = img.shape
        width *= self._factor_width
        height *= self._factor_height
        if width * height > 1000000:
            return transform_uint8(img)

        img_out: NDArray[np.float16] = transform.rescale(
            img,
            (self._factor_height, self._factor_width),
            anti_aliasing=True,
            preserve_range=True,
        )
        return transform_uint8(img_out)


class InvertGray:
    def __init(self) -> None:
        pass

    def mutate(self, mutation_probability: float = 0.3) -> None:
        pass

    def apply(self, img: NDArray[np.float16]) -> NDArray[np.float16]:
        # apply filter
        try:
            assert img.size > 0
            assert len(img.shape) == 2
        except AssertionError:
            image: NDArray[np.uint8] = np.zeros(shape=(2, 2), dtype="uint8")
            return image

        img_out: NDArray[np.float16] = util.invert(img)
        return transform_uint8(img_out)


class HOG_descriptor:
    def __init__(self, max_size: Tuple[int, int] = (50, 50)) -> None:
        # initization
        self._max_size = max_size
        self._orientations = randrange(1, 10)
        self._ppc_w = randrange(1, max_size[0] // 2 - 1)
        self._ppc_h = randrange(1, max_size[1] // 2 - 1)
        self._cpb_w = randrange(1, max_size[0] // self._ppc_w)
        self._cpb_h = randrange(1, max_size[1] // self._ppc_h)
        self._norm = choice([True, False])

    def mutate(self, mutation_probability: float = 0.3) -> None:
        # mutation
        if random() < mutation_probability:
            self._orientations = randrange(1, 10)
        if random() < mutation_probability:
            self._ppc_w = randrange(1, self._max_size[0] // 2 - 1)
        if random() < mutation_probability:
            self._ppc_h = randrange(1, self._max_size[1] // 2 - 1)
        if random() < mutation_probability:
            self._cpb_w = randrange(1, self._max_size[0] // self._ppc_w)
        if random() < mutation_probability:
            self._cpb_h = randrange(1, self._max_size[1] // self._ppc_h)
        if random() < mutation_probability:
            self._norm = choice([True, False])

    def apply(self, img: NDArray[np.float16]) -> NDArray[np.float16]:
        # apply filter
        try:
            assert img.size > 0
            assert len(img.shape) == 2
        except AssertionError:
            image: NDArray[np.float16] = np.zeros(
                shape=(2, 2), dtype="float16"
            )
            return image

        try:
            feature_vector = hog(
                img,
                orientations=self._orientations,
                pixels_per_cell=(self._ppc_w, self._ppc_h),
                cells_per_block=(self._cpb_w, self._cpb_h),
                transform_sqrt=self._norm,
            )
        except ValueError:
            # feature_vector = np.zeros(shape=(1,), dtype="float")
            feature_vector = np.array([])

        return feature_vector


class LBP_descriptor:
    def __init__(self, max_size: int = 50) -> None:
        # initization
        self._max_size = max_size
        self._r = randrange(1, max_size // 2)
        self._p = randrange(1, max_size // 2)
        self._bins = randrange(1, max_size)

    def mutate(self, mutation_probability: float = 0.3) -> None:
        # mutation
        if random() < mutation_probability:
            self._r = randrange(1, self._max_size // 2)
        if random() < mutation_probability:
            self._p = randrange(1, self._max_size // 2)
        if random() < mutation_probability:
            self._bins = randrange(1, self._max_size)

    def apply(self, img: NDArray[np.float16]) -> NDArray[np.float16]:
        # apply filter
        try:
            assert img.size > 0
            assert len(img.shape) == 2
        except AssertionError:
            image: NDArray[np.float16] = np.zeros(
                shape=(2, 2), dtype="float16"
            )
            return image

        try:
            with np.errstate(invalid="raise"):
                lbp = local_binary_pattern(img, self._p, self._r)
                hist, _ = np.histogram(
                    lbp, density=True, bins=self._bins, range=(0, self._bins)
                )
        except FloatingPointError:
            hist = np.array([])

        return hist


class RidgeDetector:
    def __init__(self, max_stop: int = 10) -> None:
        # initization
        self._max_stop = max_stop
        self._stop = randrange(1, max_stop)

    def mutate(self, mutation_probability: float = 0.3) -> None:
        # mutation
        if random() < mutation_probability:
            self._stop = randrange(1, self._max_stop)

    def apply(self, img: NDArray[np.float16]) -> NDArray[np.float16]:
        # apply filter
        try:
            assert img.size > 0
            assert len(img.shape) == 2
        except AssertionError:
            image: NDArray[np.float16] = np.zeros(
                shape=(2, 2), dtype="float16"
            )
            return image

        try:
            img_out = filters.meijering(img, sigmas=range(1, self._stop, 1))
        except ValueError:
            return transform_uint8(img_out)
        return transform_uint8(img_out)
