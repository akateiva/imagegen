import numpy as np
import pandas as pd
import cv2
import backgrounds
import argparse
import os
import cocoset

parser = argparse.ArgumentParser(description='generates synthetic training data')
parser.add_argument("--item-list", type = str, default="items_contours.pkl")
parser.add_argument("--image-dir", type = str, default="../Data/images")
parser.add_argument("--out-dir", type = str, default="../out")
parser.add_argument("--n-samples", type = int, default=30)
parser.add_argument("--n-items-per-sample", type = int, default=2)
parser.add_argument('--size', type=int, default=512)
parser.add_argument("--bg", type = str, default='plain')
parser.add_argument("--bg-img-dir", type=str, default='/tmp/indoor/*')
parser.add_argument("--debug", type=bool, default=False)
args = parser.parse_args()

np.random.seed(0)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

# real wonk
MAX_TRANSLATE = 0.5
TRANSLATE_OFFSET = -0
MAX_SCALE = 1.5
MIN_SCALE = 0.2

def rotate_item(img, angle):
    M = cv2.getRotationMatrix2D((img.shape[1]/2,img.shape[0]/2),angle,1)
    rotated_img = cv2.warpAffine(img,M,img.shape[:2])
    return rotated_img


def mask_to_poly(mask):
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, contours, _ = cv2.findContours(mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key= cv2.contourArea)
    epsilon = 0.0015*cv2.arcLength(cnt,True)
    # approximate contour to reduce 
    approx = cv2.approxPolyDP(cnt,epsilon,True) 
    return approx

# pastes the item into the image and then returns the image and an annotation
def paste_item_into_image(item, image, angle_range=15):
    # first we define the bbox where we going to paste the item
    x, y = ((np.random.rand(2) * MAX_TRANSLATE + TRANSLATE_OFFSET ) * image.shape[:-1]).astype(int)
    #x, y = (np.random.normal(loc=0.3, scale=0.10, size=2)* image.shape[:-1]).astype(int)
    w = h = int(max(MIN_SCALE, np.random.rand()*MAX_SCALE) * min(image.shape[:-1]))

    # we rescale the image and contour to the desired bbox dimensions
    item_image = cv2.imread(os.path.join(args.image_dir, item.img))       # item image
    cnt = item['contour']                                            # the contour
    cnt = (cnt * (w / item_image.shape[0])).astype(int)              # resize contour to desired dimensions
    item_image = cv2.resize(item_image, (w, h))    

    # prepare the mask
    mask = np.zeros(item_image.shape, dtype=np.uint8)
    cv2.drawContours(mask, [cnt], 0, (255, 255, 255), -1)
    
    angle = np.random.randint(-angle_range, angle_range+1)
    item_image = rotate_item(item_image, angle)
    mask = rotate_item(mask, angle)
    mask = cv2.erode(mask, kernel, iterations=2)
    
    overhang_x = image.shape[0] - x - w
    if overhang_x >= 0:
        overhang_x = None
    overhang_y = image.shape[1] - y - h
    if overhang_y >= 0:
        overhang_y = None
    
    # paste the item into the sample
    np.copyto(image[x: x+w, y:y+h, :],
            item_image[:overhang_x, :overhang_y, :],
            where=mask[:overhang_x, :overhang_y, :]>0)
    subregion = image[x: x+w, y:y+h, :]
    subregion = cv2.illuminationChange(subregion, mask[:overhang_x, :overhang_y, :], alpha=1, beta=0.5)
    
    # get the segmentation polygon of the mask ( + offset )
    segmentation = mask_to_poly(mask) + [x,w]

    return image, segmentation

# takes the given items and prepares the image and annotations
def generate_training_sample(items,
        bg_generator,
        angle_range=25):
    img = next(bg_generator)
    img_data = dataset.allocate_image(img.shape[0:-1])
    for _, item in items.iterrows():
        img, segmentation = paste_item_into_image(item, img)
        dataset.add_annotation({
            "segmentation": [segmentation.ravel().tolist()],
            "area": cv2.contourArea(segmentation),
            "iscrowd": 0,
            "image_id": img_data['id'],
            "bbox": list(cv2.boundingRect(segmentation)),
            "category_id": item['category_id']
            })
    img = cv2.blur(img,(3,3))
    return img, img_data

IM_SIZE = (args.size, args.size)
## 1. Prepare background generator
if args.bg == 'plain':
    bg_gen = backgrounds.random_plain_bg_gen(IM_SIZE)
elif args.bg == 'noise':
    bg_gen = backgrounds.random_noise_bg_gen(IM_SIZE)
elif args.bg == 'indoor':
    bg_gen = backgrounds.indoor_scene_bg_gen(IM_SIZE, args.bg_img_dir)
else:
    raise ValueError("No such background generator: {}".format(args.bg))

## 2. Load item list with contours and etc., compute weights for uniform sampling
items = pd.read_pickle(args.item_list)
print("Loaded", len(items), "with contours")
category_item_weights = items.groupby('category_id')['img'].count().apply(lambda x: 1/x).rename('item_weight')
items = items.join(category_item_weights, on='category_id')

## 3. Build dataset
dataset = cocoset.COCOSet()
os.makedirs(os.path.join(args.out_dir, 'images/'), exist_ok=True)
for i in range(0, args.n_samples):
    print("{} / {} ".format(i, args.n_samples))
    sample_items = items.sample(args.n_items_per_sample, weights=items['item_weight'])
    img, img_data = generate_training_sample(sample_items, bg_gen)
    cv2.imwrite('{}/images/{}'.format(args.out_dir, img_data['file_name']), img)
    if args.debug:
        cv2.imshow('generated', img)
        cv2.waitKey(1000)
print("writing annotations.json")
dataset.write_annotations('{}/annotations.json'.format(args.out_dir))
