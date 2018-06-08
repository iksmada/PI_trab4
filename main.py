import argparse
import cv2
import numpy as np

modes = ("projection", "hough")
pre = ("crop", "sobel", "otsu", "contours", "gray")
parser = argparse.ArgumentParser(description='Fix tilted images')
parser.add_argument('-i', '--input', type=str, help='Input image path', required=True)
parser.add_argument('-o', '--output', type=str, help='Output image path')
parser.add_argument('-m', '--mode', type=str, help='Technique for alignment algorithm',
                    default='projection', choices=modes)
parser.add_argument('-p', '--pre', type=str, help='Technique for preprocessing',default='gray', choices=pre, nargs="*")
parser.add_argument('-c', '--crop', type=int, help='Crop window size', default=500)

args = vars(parser.parse_args())
INPUT = args["input"]
OUTPUT = args["output"]
MODE = args["mode"]
PRE = args["pre"]
CROP = args["crop"]

if OUTPUT and not OUTPUT.endswith(".png"):
    OUTPUT = OUTPUT + ".png"

img_orig = cv2.imread(INPUT)
cv2.imshow("Original", img_orig)
