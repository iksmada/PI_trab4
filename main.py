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



modes = ("nearest_neighbor", "bilinear", "bicubic", "lagrange" )
parser = argparse.ArgumentParser(description='Fix tilted images')
group1 = parser.add_mutually_exclusive_group(required=True)
group1.add_argument('-a', '--angle', type=float, help='Rotation angle counter-clockwise in degrees')
group1.add_argument('-e', '--escala', type=float, help='Scaling factor')
parser.add_argument('-d', '--dimensions', type=int, nargs=2, help='Pair of values, height and width, of output image',
                    default=[-1, -1])
parser.add_argument('-m', '--mode', type=str, help='Interpolation mode',
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

img_orig = cv2.imread(INPUT)
img_gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)

# calculate new dimensions
width = np.size(img_gray, 1)
height = np.size(img_gray, 0)
if ANGLE:
    # ANGLE become positive and less than 360
    ANGLE = ANGLE % 360
    # ANGLE between 0 and 180
    simple_angle = math.fabs(ANGLE % 180 - ANGLE//180*180)
    if ANGLE > 90:
        height = width
        width = height
        simple_angle = simple_angle - 90
    new_width = math.ceil(width * math.cos(math.radians(simple_angle)) + height * math.sin(math.radians(simple_angle)))
    new_height = math.ceil(width * math.sin(math.radians(simple_angle)) + height * math.cos(math.radians(simple_angle)))

elif SCALE:
    new_width = math.ceil(width * SCALE)
    new_height = math.ceil(height * SCALE)

img_out = np.zeros((new_height, new_width), dtype=img_gray.dtype)
width = np.size(img_gray, 1)
height = np.size(img_gray, 0)
for x in range(new_height):
    for y in range(new_width):
        if ANGLE:
            x_orig = x
            y_otig = y
        if SCALE:
            x_orig = x/SCALE
            y_orig = y/SCALE
        if x_orig > 0 and x_orig < height and y_orig > 0 and y_orig < width:
            if MODE == modes[0]:
                img_out[x][y] = nearest_neighbor(img_gray, x_orig, y_orig)
            if MODE == modes[2]:
                img_out[x][y] = img_gray[x][y]




if OUTPUT and not OUTPUT.endswith(".png"):
    OUTPUT = OUTPUT + ".png"

cv2.imshow("Original", img_orig)
cv2.imshow("Transformed", img_out)
cv2.waitKey(0)
