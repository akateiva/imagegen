import cv2
import pandas as pd
import numpy as np
import json
import argparse
import itertools
import datetime
import random
import os
from elastic_transform import elastic_transform
parser = argparse.ArgumentParser(description='finds contours')
parser.add_argument("--item-list", type = str, default="items.pkl")
parser.add_argument("--image-dir", type = str, default="../Data/images")
parser.add_argument("--out-dir", type = str, default="./out")
parser.add_argument("--n-samples", type = int, default=30)
parser.add_argument("--n-items-per-sample", type = int, default=2)
parser.add_argument("--use-elastic-transform", type = bool, default=False)
parser.add_argument("--use-noise-bg", type = bool, default=False)


categories = [
    {'id': 1,'supercategory': 'clothing', 'name': 'bag'}, 
    {'id': 2,'supercategory': 'clothing', 'name': 'boots'},
    {'id': 3,'supercategory': 'clothing', 'name': 'footwear'},
    {'id': 4,'supercategory': 'clothing', 'name': 'outer'},
    {'id': 5,'supercategory': 'clothing', 'name': 'sunglasses'},
    {'id': 6,'supercategory': 'clothing', 'name': 'pants'},
    {'id': 7,'supercategory': 'clothing', 'name': 'top'},
    {'id': 8,'supercategory': 'clothing', 'name': 'shorts'},
    {'id': 9,'supercategory': 'clothing', 'name': 'headwear'},
    {'id': 10,'supercategory': 'clothing', 'name': 'scarf&tie'}
]

generated_count = 0;
annotation_count = 0;
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

# rotates an image by angle in degrees
def rotate_item(img, angle):
    M = cv2.getRotationMatrix2D((img.shape[1]/2,img.shape[0]/2),angle,1)
    rotated_img = cv2.warpAffine(img,M,img.shape[:2])
    return rotated_img

# scales an item by scale (0 to 1)
def scale_item(img, scale):
    M = np.float32([[scale, 0, ((1-scale)*img.shape[0])/2], [0, scale, ((1-scale)*img.shape[0])/2]])
    scaled_img = cv2.warpAffine(img,M,img.shape[:2])
    return scaled_img

def translate_item(img, tx, ty):
    M = np.float32([[1, 0, img.shape[0]*tx], [0, 1, img.shape[1]*ty]])
    shifted_img = cv2.warpAffine(img, M, img.shape[:2])
    return shifted_img

def apply_random_transforms(img, mask, angle_range):
    # scale
    scale = random.uniform(0.2, 0.9)
    img = scale_item(img, scale)
    mask = scale_item(mask, scale)
    # rotate
    angle = random.randint(-angle_range, angle_range)
    img = rotate_item(img, angle)
    mask = rotate_item(mask, angle)
    # translate
    tx = random.uniform(-0.35, 0.35)
    ty = random.uniform(-0.35, 0.35)
    img = translate_item(img, tx, ty)
    mask = translate_item(mask, tx, ty)
    return img, mask

    # elastic transform
    #img = elastic_transform(img, 666, 0.2)


# generate_images stiches the provided items together into a single image with annotations
def generate_image(items, image_dir, out_dir, angle_range = 25, use_elastic_transform = False, use_noise_bg = False):
    global generated_count, annotation_count

    out_image = np.full((512, 512, 3), np.random.randint(255, size=3, dtype=np.uint8), np.uint8)
    if use_noise_bg:
        out_image = np.random.randint(0, 255, size=(512, 512, 3), dtype=np.uint8)

    image_data = { "id": generated_count, "file_name": str(generated_count) + '.jpg', "width": 512, "height": 512} # coco images section
    annotations_data = []
    for _, item in items.iterrows():
        # place each of the item images in out_image
        # TODO: Cache
        img = cv2.imread(os.path.join(image_dir, item.img))
        cnt = item['contour']
        # rescale into 512 x 512
        cnt = (cnt * (out_image.shape[0] / img.shape[0])).astype(int)
        img = cv2.resize(img, (512, 512))
        mask = np.zeros((512,512,3), dtype=np.uint8)
        cv2.drawContours(mask, [cnt], 0, (255, 255, 255), -1)
        img, mask = apply_random_transforms(img, mask, angle_range)
    
        # elastic transform
        if use_elastic_transform:
            img, mask = elastic_transform(img, mask, img.shape[1]*1.5, img.shape[1]*0.08, img.shape[1]*0.08)
        mask = cv2.erode(mask, kernel, iterations=2)
    
        # after random transforms, masks no longer match contours
        # so they have to be found again
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, contours, _ = cv2.findContours(mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key= cv2.contourArea)
        epsilon = 0.0015*cv2.arcLength(cnt,True)
        # approximate contour to reduce 
        approx = cv2.approxPolyDP(cnt,epsilon,True) 
        np.copyto(out_image, img, where=mask>0)

        annotations_data.append({
            "segmentation": [approx.ravel().tolist()],
            "area": cv2.contourArea(cnt),
            "iscrowd": 0,
            "image_id": image_data['id'],
            "bbox": list(cv2.boundingRect(cnt)), 
            "category_id": item['category_id'],
            "id": annotation_count
            })
        annotation_count = annotation_count + 1 

    cv2.imwrite(os.path.join(out_dir, 'images', image_data['file_name']), out_image)
    generated_count = generated_count + 1
    return image_data, annotations_data

def main():
    args = parser.parse_args()
    print("loading item contours")
    items = pd.read_pickle(args.item_list)
    print("loaded", len(items), "with contours")

    # compute the weights per category so that each item can be sampled from a uniform category distribution
    category_item_weights = items.groupby('category_id')['img'].count().apply(lambda x: 1/x).rename('item_weight')
    items = items.join(category_item_weights, on='category_id')
    print("computed category item weights", category_item_weights)
    print(items.describe())

    generate_dataset(args.out_dir, items, args.image_dir, args.n_samples, args.n_items_per_sample, args.use_elastic_transform, args.use_noise_bg)

    
def generate_dataset(out_dir, items, image_dir, n_samples, n_items_per_sample, use_elastic_transform, use_noise_bg):
    print('> preparing directories')
    os.makedirs(os.path.join(out_dir, 'images/'))

    dataset = {
            "info": {
                "description": "generated dataset",
                "date_created": str(datetime.datetime.now())
                },
            "images": [],
            "annotations": [],
            "categories": categories,
            }

    for i in range(0, n_samples):
        sample_items = items.sample(n_items_per_sample, weights=items['item_weight'])
        try:
            image, annotations = generate_image(sample_items, image_dir, out_dir, use_elastic_transform=use_elastic_transform, use_noise_bg=use_noise_bg)
            dataset['images'].append(image)
            dataset['annotations'] = dataset['annotations'] + annotations
        except:
            print("sample", str(i), "failed. maybe the countour was not good enough?")
    
    print('> writing annotations')
    with open(os.path.join(out_dir, 'annotations.json'), 'w') as f:
        json.dump(dataset, f)
    print('> done')

if __name__ == '__main__':
    main()
