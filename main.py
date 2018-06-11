import argparse
import cv2
import numpy as np
import math


def centered_crop(img, new_height, new_width):
    width = np.size(img, 1)
    height = np.size(img, 0)

    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2
    c_img = img[top:bottom, left:right, :]
    return c_img


def nearest_neighbor(img_orig, x, y):
    if round(x) == height:
        x = x - 0.5
    if round(y) == width:
        y = y - 0.5
    return img_orig[round(x)][round(y)]


def bilinear(img_orig, x, y):
    base_x = math.floor(x)
    base_y = math.floor(y)
    if base_x < height - 1 and base_y < width - 1:
        dx = x - base_x
        dy = y - base_y
        acc = (1 - dx) * (1 - dy) * img_orig[base_x][base_y]
        acc = acc + dx * (1 - dy) * img_orig[base_x + 1][base_y]
        acc = acc + (1 - dx) * dy * img_orig[base_x][base_y + 1]
        acc = acc + dx * dy * img_orig[base_x][base_y + 1]
        return acc
    else:
        return nearest_neighbor(img_orig, x, y)


def bicubic(img_orig, x, y):
    def R(s):
        def P(t):
            if t > 0:
                return t
            else:
                return 0

        return (P(s + 2) ** 3 - 4 * P(s + 1) ** 3 + 6 * P(s) ** 3 - 4 * P(s - 1) ** 3) / 6

    base_x = math.floor(x)
    base_y = math.floor(y)
    if base_x < height - 2 and base_y < width - 2:
        acc = 0
        dx = x - base_x
        dy = y - base_y
        for m in range(-1, 3):
            for n in range(-1, 3):
                acc = acc + img_orig[base_x + n][base_y + m] * R(m - dy) * R(dx - n)
        return acc
    else:
        return bilinear(img_orig, x, y)


def lagrange(img_orig, x, y):
    def L(n):
        return (
                (-dx * (dx - 1) * (dx - 2) * img_orig[base_x - 1][base_y + n - 2]) / 6 +
                ((dx + 1) * (dx - 1) * (dx - 2) * img_orig[base_x][base_y + n - 2]) / 2 +
                (-dx * (dx + 1) * (dx - 2) * img_orig[base_x + 1][base_y + n - 2]) / 2 +
                (dx * (dx + 1) * (dx - 1) * img_orig[base_x + 2][base_y + n - 2]) / 6
        )

    base_x = math.floor(x)
    base_y = math.floor(y)
    if base_x < height - 2 and base_y < width - 2:
        acc = 0
        dx = x - base_x
        dy = y - base_y
        return (
                (-dy * (dy - 1) * (dy - 2) * L(1)) / 6 +
                ((dy + 1) * (dy - 1) * (dy - 2) * L(2)) / 2 +
                (-dy * (dy + 1) * (dy - 2) * L(3)) / 2 +
                (dy * (dy + 1) * (dy - 1) * L(4)) / 6
        )

    else:
        return bilinear(img_orig, x, y)


modes = ("nearest_neighbor", "bilinear", "bicubic", "lagrange")
parser = argparse.ArgumentParser(description='Fix tilted images')
group1 = parser.add_mutually_exclusive_group(required=True)
group1.add_argument('-a', '--angle', '-r', type=float, help='Rotation angle counter-clockwise in degrees')
group1.add_argument('-e', '--escala', '-s', type=float, help='Scaling factor')
parser.add_argument('-d', '--dimensions', type=int, nargs=2, help='Pair of values, height and width, of output image',
                    default=[-1, -1])
parser.add_argument('-m', '--mode', type=str.lower , help='Interpolation mode',
                    default=modes[0], choices=modes)
parser.add_argument('-i', '--input', type=str, help='Input image path', required=True)
parser.add_argument('-o', '--output', type=str, help='Output image path')

args = vars(parser.parse_args())
print(args)
INPUT = args["input"]
OUTPUT = args["output"]
MODE = args["mode"]
DIM = args["dimensions"]
SCALE = args["escala"]
ANGLE = args["angle"]

img_orig = cv2.imread(INPUT, cv2.IMREAD_ANYCOLOR)

# calculate new dimensions
width = np.size(img_orig, 1)
height = np.size(img_orig, 0)
if ANGLE:
    # ANGLE become positive and less than 360
    ANGLE = ANGLE % 360
    # ANGLE between 0 and 180
    simple_angle = math.fabs(ANGLE % 180 - ANGLE // 180 * 180)
    if ANGLE > 90:
        height = width
        width = height
        simple_angle = simple_angle - 90
    new_width = math.ceil(width * math.cos(math.radians(simple_angle)) + height * math.sin(math.radians(simple_angle)))
    new_height = math.ceil(width * math.sin(math.radians(simple_angle)) + height * math.cos(math.radians(simple_angle)))

elif SCALE:
    new_width = math.ceil(width * SCALE)
    new_height = math.ceil(height * SCALE)
if len(img_orig.shape) > 2:
    img_out = np.zeros((new_height, new_width, img_orig.shape[2]), dtype=img_orig.dtype)
else:
    img_out = np.zeros((new_height, new_width), dtype=img_orig.dtype)
width = np.size(img_orig, 1)
height = np.size(img_orig, 0)


def ceil(value, max):
    if value > max:
        return max
    else:
        return value


def floor(value, min):
    if value < min:
        return min
    else:
        return value


for x in range(new_height):
    for y in range(new_width):
        if ANGLE:
            new_cx, new_cy = new_height / 2.0, new_width / 2.0
            cx, cy = height / 2.0, width / 2.0
            x_orig = math.cos(math.radians(-ANGLE)) * (x - new_cx) - math.sin(math.radians(-ANGLE)) * (y - new_cy) + cx
            y_orig = math.sin(math.radians(-ANGLE)) * (x - new_cx) + math.cos(math.radians(-ANGLE)) * (y - new_cy) + cy
        if SCALE:
            x_orig = x / SCALE
            y_orig = y / SCALE
        if 0 < x_orig < height and 0 < y_orig < width:
            if MODE == modes[0]:
                img_out[x][y] = nearest_neighbor(img_orig, x_orig, y_orig)
            elif MODE == modes[1]:
                img_out[x][y] = bilinear(img_orig, x_orig, y_orig)
            elif MODE == modes[2]:
                img_out[x][y] = bicubic(img_orig, x_orig, y_orig)
            elif MODE == modes[3]:
                pixel = np.around(lagrange(img_orig, x_orig, y_orig), decimals=0)
                if len(pixel) > 1:
                    for i in range(len(pixel)):
                        pixel[i] = ceil(pixel[i], 255)
                        pixel[i] = floor(pixel[i], 0)

                else:
                    pixel = ceil(pixel, 255)
                    pixel = floor(pixel, 0)
                img_out[x][y] = pixel


if DIM[0] > 0 and DIM[1] > 0:
    img_out = centered_crop(img_out, DIM[0], DIM[1])

if OUTPUT and not OUTPUT.endswith(".png"):
    OUTPUT = OUTPUT + ".png"
    cv2.imwrite(OUTPUT, img_out)

cv2.imshow("Original", img_orig)
cv2.imshow("Transformed using %s interpolation" % MODE, img_out)
cv2.waitKey(0)
