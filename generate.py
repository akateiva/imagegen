import numpy as np
import pandas as pd
import cv2
import backgrounds
import argparse
import os
import cocoset
import elasticdeform

parser = argparse.ArgumentParser(description='generates synthetic training data')
parser.add_argument("--item-list", type = str, default="items_contours.pkl")
parser.add_argument("--image-dir", type = str, default="../Data/images")
parser.add_argument("--out-dir", type = str, default="../out")
parser.add_argument("--n-samples", type = int, default=30)
parser.add_argument("--n-items-per-sample", type = int, default=2)
parser.add_argument('--size', type=int, default=512)
parser.add_argument("--bg", type = str, default='plain')
parser.add_argument("--bg-img-dir", type=str, default='/tmp/indoor/*')
parser.add_argument("--debug", action='store_true')
parser.add_argument("--hue-shift", action='store_true')
parser.add_argument("--elastic-transform", action='store_true')
parser.add_argument("--ps-blend", action='store_true')
args = parser.parse_args()

np.random.seed(0)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))

# real wonk
MAX_TRANSLATE = 0.5
TRANSLATE_OFFSET = -0
MAX_SCALE = 0.7
MIN_SCALE = 0.3

def rotate_item(img, angle):
    M = cv2.getRotationMatrix2D((img.shape[1]/2,img.shape[0]/2),angle,1)
    rotated_img = cv2.warpAffine(img,M,img.shape[:2])
    return rotated_img

def hue_shift(img, angle):
    #angle must be uint8
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    shift_h = (h + angle) % 180
    shift_hsv = cv2.merge((shift_h, s, v))
    shift_img = cv2.cvtColor(shift_hsv, cv2.COLOR_HSV2BGR)
    return shift_img


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
    ## SCALING AND TRANSLATION
    item_image = cv2.imread(os.path.join(args.image_dir, item.img))  # item image
    cnt = item['contour']                                            # the contour
    cnt = (cnt * (w / item_image.shape[0])).astype(int)              # resize contour to desired dimensions
    item_image = cv2.resize(item_image, (w, h))    
    ## MASK 
    mask = np.zeros(item_image.shape, dtype=np.uint8) 
    #mask = np.full(item_image.shape, fill_value=128,dtype=np.uint8)
    cv2.drawContours(mask, [cnt], 0, (255, 255, 255), -1)
    ## ROTATION
    angle = np.random.randint(-angle_range, angle_range+1)
    item_image = rotate_item(item_image, angle)
    mask = rotate_item(mask, angle)
    # MASK EROSION
    mask = cv2.erode(mask, kernel, iterations=2)
    #item_image = cv2.bitwise_and(item_image, mask)                  # put image on black bg for poisson blending
    tmp = np.full(item_image.shape, fill_value=64, dtype=np.uint8)
    np.copyto(tmp, item_image, where=mask>0)
    item_image = tmp
    #item_image = cv2.bitwise_and(item_image, mask)                  # put image on grey bg for poisson
    mask = cv2.dilate(mask, kernel, iterations=1)

    mask_area = (mask > 1).sum() / 3
    if mask_area < 32*32:
        print('dropping', item, 'because mask area=', mask_area)
        return image, None

    if args.debug:
        cv2.imshow('mask', mask)
        cv2.imshow('item image', item_image)
    ## ELASTIC DEFORMATION
    if args.elastic_transform:
        [item_image, mask] = elasticdeform.deform_random_grid([item_image, mask], points=5, sigma=1, axis=[(0, 1), (0, 1)])
    ## HUE SHIFT
    if args.hue_shift:
        hue_shift_amount = np.random.randint(0, 180+1, dtype=np.uint8)
        item_image = hue_shift(item_image, hue_shift_amount)

    overhang_x = image.shape[0] - x - w
    if overhang_x >= 0:
        overhang_x = None
    overhang_y = image.shape[1] - y - h
    if overhang_y >= 0:
        overhang_y = None

    min_x_position = int(item_image.shape[1]/2)
    max_x_position = int(image.shape[1] - item_image.shape[1]/2)
    min_y_position = int(item_image.shape[0]/2)
    max_y_position = int(image.shape[0] - item_image.shape[0]/2)


    ## POISSON BLENDING
    if args.ps_blend:
        center_x = np.random.randint(min_x_position, max_x_position)
        center_y = np.random.randint(min_y_position, max_y_position)
        print('Image Shape: {}, Item Shape {}, Center: {} {}'.format(image.shape, item_image.shape, center_x, center_y))
        image = cv2.seamlessClone(item_image, image, mask, (center_x,center_y), cv2.NORMAL_CLONE)
    else:
    ## RUFF COPY 
        # paste the item into the sample
        np.copyto(image[x: x+w, y:y+h, :],
                item_image[:overhang_x, :overhang_y, :],
                where=mask[:overhang_x, :overhang_y, :]>0)
    
    # get the segmentation polygon of the mask ( + offset )
    segmentation = mask_to_poly(mask) + [y,x]

    return image, segmentation

# takes the given items and prepares the image and annotations
def generate_training_sample(items,
        bg_generator,
        angle_range=25):
    img = next(bg_generator)
    img_data = dataset.allocate_image(img.shape[0:-1])
    for _, item in items.iterrows():
        img, segmentation = paste_item_into_image(item, img)
        if segmentation is not None:
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
        cv2.waitKey(10000)
print("writing annotations.json")
dataset.write_annotations('{}/annotations.json'.format(args.out_dir))
