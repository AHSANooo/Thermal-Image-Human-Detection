import gc

import cv2

from pipeline import pipeline as p
import os
import glob

input_dir = "dataset/"

def main():
    image_files = glob.glob(os.path.join(input_dir, "*.png")) + \
                  glob.glob(os.path.join(input_dir, "*.jpg"))

    for i, image_file in enumerate(image_files):
        result = p(image_file)
        cv2.imwrite("output/result" + str(i) + ".png", result)
        gc.collect()

if __name__ == "__main__":
    main()