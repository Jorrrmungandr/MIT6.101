#!/usr/bin/env python3

import math
import pytest
from PIL import Image as Image

# NO ADDITIONAL IMPORTS ALLOWED!


def get_pixel(image, x, y):
    w = image['width']
    return image['pixels'][x*w + y]


def set_pixel(image, x, y, c):
    w = image['width']
    image['pixels'][x*w + y] = c


def apply_per_pixel(image, func):
    h, w = image['height'], image['width']
    result = {
        'height': h,
        'width': w,
        'pixels': [0 for _ in range(h * w)],
    }
    for x in range(h):
        for y in range(w):
            color = get_pixel(image, x, y)
            newcolor = func(color)
            set_pixel(result, x, y, newcolor)
    return result


def inverted(image):
    return apply_per_pixel(image, lambda c: 255-c)


# HELPER FUNCTIONS

def correlate(image, kernel, boundary_behavior):
    """
    Compute the result of correlating the given image with the given kernel.
    `boundary_behavior` will one of the strings 'zero', 'extend', or 'wrap',
    and this function will treat out-of-bounds pixels as having the value zero,
    the value of the nearest edge, or the value wrapped around the other edge
    of the image, respectively.

    if boundary_behavior is not one of 'zero', 'extend', or 'wrap', return
    None.

    Otherwise, the output of this function should have the same form as a 6.101
    image (a dictionary with 'height', 'width', and 'pixels' keys), but its
    pixel values do not necessarily need to be in the range [0,255], nor do
    they need to be integers (they should not be clipped or rounded at all).

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.

    DESCRIBE YOUR KERNEL REPRESENTATION HERE

    kernel is a N*N matrix
    """
    if boundary_behavior not in ('zero', 'extend', 'wrap'):
        return

    w, h, n = image['width'], image['height'], len(kernel)
    new_h, new_w = h + n - 1, w + n - 1
    padding = (n-1) // 2

    new_image = {
        'height': new_h,
        'width': new_w,
        'pixels': [0 for _ in range(new_h * new_w)]
    }

    for i, new_i in enumerate(range(padding, padding+h)):
        for j, new_j in enumerate(range(padding, padding+w)):
            c = get_pixel(image, i, j)
            set_pixel(new_image, new_i, new_j, c)

    if boundary_behavior == 'zero':
        pass

    elif boundary_behavior == 'extend':
        # corner
        # left top
        for i in range(padding):
            for j in range(padding):
                set_pixel(new_image, i, j, get_pixel(image, 0, 0))
        # right top
        for i in range(padding):
            for j in range(padding+w, new_w):
                set_pixel(new_image, i, j, get_pixel(image, 0, w-1))
        # left bottom
        for i in range(padding+h, new_h):
            for j in range(padding):
                set_pixel(new_image, i, j, get_pixel(image, h-1, 0))
        # right bottom
        for i in range(padding+h, new_h):
            for j in range(padding+w, new_w):
                set_pixel(new_image, i, j, get_pixel(image, h-1, w-1))

        # side
        # top
        for i in range(padding):
            for j in range(padding, padding+w):
                set_pixel(new_image, i, j, get_pixel(image, 0, j-padding))

        # left
        for i in range(padding, padding+h):
            for j in range(padding):
                set_pixel(new_image, i, j, get_pixel(image, i-padding, 0))
        # right
        for i in range(padding, padding+h):
            for j in range(padding+w, new_w):
                set_pixel(new_image, i, j, get_pixel(image, i-padding, w-1))
        # bottom
        for i in range(padding+h, new_h):
            for j in range(padding, padding+w):
                set_pixel(new_image, i, j, get_pixel(image, h-1, j-padding))

    elif boundary_behavior == 'wrap':
        pass

    res = {
        'height': h,
        'width': w,
        'pixels': [0 for _ in range(h * w)]
    }

    for i in range(padding, padding+h):
        for j in range(padding, padding+w):
            new_c = 0
            for x, xx in enumerate(range(i-padding, i+padding+1)):
                for y, yy in enumerate(range(j-padding, j+padding+1)):
                    new_c += kernel[x][y] * get_pixel(new_image, xx, yy)
            set_pixel(res, i-padding, j-padding, new_c)
    # no need to clip
    # round_and_clip_image(new_image)
    return res



def round_and_clip_image(image):
    """
    Given a dictionary, ensure that the values in the 'pixels' list are all
    integers in the range [0, 255].

    All values should be converted to integers using Python's `round` function.

    Any locations with values higher than 255 in the input should have value
    255 in the output; and any locations with values lower than 0 in the input
    should have value 0 in the output.
    """
    w, h = image['width'], image['height']

    for i in range(h):
        for j in range(w):
            c = get_pixel(image, i, j)
            if not isinstance(c, int):
                set_pixel(image, i, j, round(c))
            if c < 0:
                set_pixel(image, i, j, 0)
            elif c > 255:
                set_pixel(image, i, j, 255)


# FILTERS
def blur_kernel(n):
    val = 1 / n ** 2
    return [[val for _ in range(n)]
            for __ in range(n)]


def blurred(image, n):
    """
    Return a new image representing the result of applying a box blur (with
    kernel size n) to the given input image.

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.
    """
    # first, create a representation for the appropriate n-by-n kernel (you may
    # wish to define another helper function for this)
    kernel = blur_kernel(n)

    # then compute the correlation of the input image with that kernel
    res = correlate(image, kernel, 'extend')

    # and, finally, make sure that the output is a valid image (using the
    # helper function from above) before returning it.
    round_and_clip_image(res)
    return res

def sharpen_kernel(n):
    val = 1 / n ** 2
    res = [[-val for _ in range(n)]
            for __ in range(n)]
    res[n//2][n//2] += 2

    return res


def sharpened(image, n):
    # create the kernel used for sharpening
    kernel = sharpen_kernel(n)

    # then compute the correlation of the input image with that kernel
    res = correlate(image, kernel, 'extend')

    # and, finally, make sure that the output is a valid image (using the
    # helper function from above) before returning it.
    round_and_clip_image(res)
    return res

def edges(image):
    kernel_x = [[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]]

    kernel_y = [[-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]]

    res1 = correlate(image, kernel_x, 'extend')
    res2 = correlate(image, kernel_y, 'extend')

    w, h = image['width'], image['height']
    res = {
        'height': h,
        'width': w,
        'pixels': [0 for _ in range(h * w)]
    }


    for i in range(h):
        for j in range(w):
            c1 = get_pixel(res1, i, j)
            c2 = get_pixel(res2, i, j)
            set_pixel(res, i, j, math.sqrt(c1**2 + c2**2))

    round_and_clip_image(res)
    return res


# HELPER FUNCTIONS FOR LOADING AND SAVING IMAGES

def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns a dictionary
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_image('test_images/cat.png')
    """
    with open(filename, 'rb') as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith('RGB'):
            pixels = [round(.299 * p[0] + .587 * p[1] + .114 * p[2])
                      for p in img_data]
        elif img.mode == 'LA':
            pixels = [p[0] for p in img_data]
        elif img.mode == 'L':
            pixels = list(img_data)
        else:
            raise ValueError('Unsupported image mode: %r' % img.mode)
        w, h = img.size
        return {'height': h, 'width': w, 'pixels': pixels}


def save_greyscale_image(image, filename, mode='PNG'):
    """
    Saves the given image to disk or to a file-like object.  If filename is
    given as a string, the file type will be inferred from the given name.  If
    filename is given as a file-like object, the file type will be determined
    by the 'mode' parameter.
    """
    out = Image.new(mode='L', size=(image['width'], image['height']))
    out.putdata(image['pixels'])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


# custom test
def test_correlate_extend():
    img = {
        'height': 3,
        'width': 3,
        'pixels': [0, 1, 2,
                   3, 4, 5,
                   6, 7, 8],
    }

    kernel = [[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0]]
    res = blur_kernel(5)


def test_blur():
    img = load_greyscale_image('test_images/cat.png')
    res = blurred(img, 3)
    save_greyscale_image(res, 'blurred_cat.png')

def test_sharpen():
    img = load_greyscale_image('test_images/cat.png')
    res = sharpened(img, 3)
    save_greyscale_image(res, 'sharpened_cat.png')

def test_edge():
    image = load_greyscale_image('test_images/cat.png')

    kernel_x = [[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]]

    kernel_y = [[-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]]

    res1 = correlate(image, kernel_x, 'extend')
    res2 = correlate(image, kernel_y, 'extend')

    res = edges(image)

    save_greyscale_image(res1, 'edge_centered_pixel1.png')
    save_greyscale_image(res2, 'edge_centered_pixel2.png')

    save_greyscale_image(res, 'edge_cat.png')


if __name__ == '__main__':
    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place for
    # generating images, etc.

    pass
