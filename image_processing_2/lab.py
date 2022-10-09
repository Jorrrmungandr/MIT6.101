#!/usr/bin/env python3

# NO ADDITIONAL IMPORTS!
# (except in the last part of the lab; see the lab writeup for details)
import copy
import math
from PIL import Image


# VARIOUS FILTERS
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



def color_filter_from_greyscale_filter(filt):
    """
    Given a filter that takes a greyscale image as input and produces a
    greyscale image as output, returns a function that takes a color image as
    input and produces the filtered color image.
    """
    def color_filter(img):
        h, w = img['height'], img['width']

        img_r = {
            'height': h,
            'width': w,
            'pixels': [img['pixels'][i][0] for i in range(h*w)]
        }
        img_g = {
            'height': h,
            'width': w,
            'pixels': [img['pixels'][i][1] for i in range(h * w)]
        }
        img_b = {
            'height': h,
            'width': w,
            'pixels': [img['pixels'][i][2] for i in range(h * w)]
        }

        res_r, res_g, res_b = filt(img_r), filt(img_g), filt(img_b)

        return {
            'height': h,
            'width': w,
            'pixels': [(res_r['pixels'][i], res_g['pixels'][i], res_b['pixels'][i]) for i in range(h*w)]
        }

    return color_filter


def make_blur_filter(n):
    def blur_filter(img):
        return blurred(img, n)

    return blur_filter


def make_sharpen_filter(n):
    def sharpen_filter(img):
        return sharpened(img, n)

    return sharpen_filter


def filter_cascade(filters):
    """
    Given a list of filters (implemented as functions on images), returns a new
    single filter such that applying that filter to an image produces the same
    output as applying each of the individual ones in turn.
    """
    def cascade_filter(img):
        cur = img
        for filt in filters:
            cur = filt(cur)
        return cur

    return cascade_filter


# SEAM CARVING

# Main Seam Carving Implementation


def seam_carving(image, ncols):
    """
    Starting from the given image, use the seam carving technique to remove
    ncols (an integer) columns from the image. Returns a new image.
    """
    res = copy.deepcopy(image)
    for _ in range(ncols):
        grey = greyscale_image_from_color_image(res)
        energy = compute_energy(grey)
        cem = cumulative_energy_map(energy)
        seam = minimum_energy_seam(cem)
        res = image_without_seam(res, seam)

    return res


# Optional Helper Functions for Seam Carving


def greyscale_image_from_color_image(image):
    """
    Given a color image, computes and returns a corresponding greyscale image.

    Returns a greyscale image (represented as a dictionary).
    """
    h, w = image['height'], image['width']

    return {
        'height': h,
        'width': w,
        'pixels': [round(image['pixels'][i][0] * .299 +
                   image['pixels'][i][1] * .587 +
                   image['pixels'][i][2] * .114)
                   for i in range(h*w)]
    }


def compute_energy(grey):
    """
    Given a greyscale image, computes a measure of "energy", in our case using
    the edges function from last week.

    Returns a greyscale image (represented as a dictionary).
    """
    return edges(grey)


def cumulative_energy_map(energy):
    """
    Given a measure of energy (e.g., the output of the compute_energy
    function), computes a "cumulative energy map" as described in the lab 2
    writeup.

    Returns a dictionary with 'height', 'width', and 'pixels' keys (but where
    the values in the 'pixels' array may not necessarily be in the range [0,
    255].
    """
    h, w = energy['height'], energy['width']
    cem = copy.deepcopy(energy)

    for i in range(1, h):
        for j in range(w):
            cem['pixels'][i*w+j] += min(cem['pixels'][(i-1)*w+(j-1 if j>0 else j)],
                                        cem['pixels'][(i-1)*w+j],
                                        cem['pixels'][(i-1)*w+(j+1 if j<w-1 else j)])

    return cem



def minimum_energy_seam(cem):
    """
    Given a cumulative energy map, returns a list of the indices into the
    'pixels' list that correspond to pixels contained in the minimum-energy
    seam (computed as described in the lab 2 writeup).
    """
    h, w = cem['height'], cem['width']
    seam = []

    last_row = cem['pixels'][(h-1)*w : h*w]
    min_index = last_row.index(min(last_row))
    seam.append(min_index + (h-1)*w)

    for i in reversed(range(h-1)):
        cur_row = cem['pixels'][i*w : (i+1)*w]
        min_value = min(cur_row[min_index-1 if min_index > 0 else min_index],
                        cur_row[min_index],
                        cur_row[min_index+1 if min_index < w-1 else min_index])

        for j in range(min_index-1, min_index+2):
            if 0 <= j <= w-1 and cur_row[j] == min_value:
                min_index = j
                break

        seam.append(min_index + w*i)

    return seam


def image_without_seam(image, seam):
    """
    Given a (color) image and a list of indices to be removed from the image,
    return a new image (without modifying the original) that contains all the
    pixels from the original image except those corresponding to the locations
    in the given list.
    """
    res = copy.deepcopy(image)
    for x in seam:
        res['pixels'].pop(x)

    res['width'] -= 1
    return res


# HELPER FUNCTIONS FOR LOADING AND SAVING COLOR IMAGES


def load_color_image(filename):
    """
    Loads a color image from the given file and returns a dictionary
    representing that image.

    Invoked as, for example:
       i = load_color_image('test_images/cat.png')
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img = img.convert("RGB")  # in case we were given a greyscale image
        img_data = img.getdata()
        pixels = list(img_data)
        w, h = img.size
        return {"height": h, "width": w, "pixels": pixels}


def save_color_image(image, filename, mode="PNG"):
    """
    Saves the given color image to disk or to a file-like object.  If filename
    is given as a string, the file type will be inferred from the given name.
    If filename is given as a file-like object, the file type will be
    determined by the 'mode' parameter.
    """
    out = Image.new(mode="RGB", size=(image["width"], image["height"]))
    out.putdata(image["pixels"])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns an instance of this class
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_greyscale_image('test_images/cat.png')
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith("RGB"):
            pixels = [
                round(0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]) for p in img_data
            ]
        elif img.mode == "LA":
            pixels = [p[0] for p in img_data]
        elif img.mode == "L":
            pixels = list(img_data)
        else:
            raise ValueError("Unsupported image mode: %r" % img.mode)
        w, h = img.size
        return {"height": h, "width": w, "pixels": pixels}


def save_greyscale_image(image, filename, mode="PNG"):
    """
    Saves the given image to disk or to a file-like object.  If filename is
    given as a string, the file type will be inferred from the given name.  If
    filename is given as a file-like object, the file type will be determined
    by the 'mode' parameter.
    """
    out = Image.new(mode="L", size=(image["width"], image["height"]))
    out.putdata(image["pixels"])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()

# MY OWN FUNCTION
def custom_feature():
    pass

def circle(img, rgb, x, y, radius):
    res = copy.deepcopy(img)

    w = res['width']
    for i in range(radius):
        j = round(math.sqrt(radius**2 - i**2))
        res['pixels'][(x+i) * w + y+j] = rgb
        res['pixels'][(x-i) * w + y+j] = rgb
        res['pixels'][(x+i) * w + y-j] = rgb
        res['pixels'][(x-i) * w + y-j] = rgb
        res['pixels'][(x+j) * w + y+i] = rgb
        res['pixels'][(x-j) * w + y+i] = rgb
        res['pixels'][(x+j) * w + y-i] = rgb
        res['pixels'][(x-j) * w + y-i] = rgb
        if j < i:
            break

    return res


if __name__ == "__main__":
    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place for
    # generating images, etc.
    img = load_color_image('test_images/frog.png')

    test = circle(img, (255,255,255), 100, 100, 30)

    save_color_image(test, 'test.png')

    circle_img = load_color_image('test.png')

