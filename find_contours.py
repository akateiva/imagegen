import argparse
import pandas as pd
import numpy as np
import os
import cv2

parser = argparse.ArgumentParser(description='finds contours')
parser.add_argument("--item-list", type = str, default="items.pkl")
parser.add_argument("--image-dir", type = str, default="../images")

def load_items(path):
    print("loading item list from", path)
    items = pd.read_pickle(path)
    print("loaded", len(items), "items")
    return items

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
def find_item_contour(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    upper = np.array([ 256, 7, 255])
    lower = np.array([ 0, 0, 245])
    ranged = 255 - cv2.inRange(hsv, lower, upper)
    th, thresh = cv2.threshold(ranged, 200, 255, cv2.THRESH_BINARY)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    thresh = cv2.erode(thresh, kernel, iterations=2)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key= cv2.contourArea)
    return cnt

def get_contour(path):
    img = cv2.imread(path)
    cnt = find_item_contour(img)
    contour_display = np.zeros(img.shape)
    epsilon = 0.0015*cv2.arcLength(cnt,True)
    # approximate contour to reduce 
    approx = cv2.approxPolyDP(cnt,epsilon,True) 
    return approx

def main():
    args = parser.parse_args()
    items = load_items(args.item_list)
    items = items[1:400]
    
    print("finding contours for items")
    items['contour'] = items['img'].apply(lambda filename: get_contour(os.path.join(args.image_dir, filename)))
    items['contour_area'] = items['contour'].apply(cv2.contourArea)
    print(items.groupby('category_id').describe())

    print("writing ./items_contours.pkl")
    items.to_pickle('./items_contours.pkl')


if __name__ == "__main__":
    # execute only if run as a script
    main()
