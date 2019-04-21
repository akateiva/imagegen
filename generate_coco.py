import cv2
import pandas as pd
import numpy as np
import json
import argparse
import itertools
import datetime
import random
import os

parser = argparse.ArgumentParser(description='finds contours')
parser.add_argument("--item-list", type = str, default="items.pkl")
parser.add_argument("--image-dir", type = str, default="../Data/images")
parser.add_argument("--out-dir", type = str, default="./out")


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
    scale = random.uniform(0, 1)
    print("scaling item to ", scale)
    img = scale_item(img, scale)
    mask = scale_item(mask, scale)
    # rotate
    angle = random.randint(-angle_range, angle_range)
    print("rotating item", angle, "degrees")
    img = rotate_item(img, angle)
    mask = rotate_item(mask, angle)
    # translate
    tx = random.uniform(-0.25, 0.25)
    ty = random.uniform(-0.25, 0.25)
    img = translate_item(img, tx, ty)
    mask = translate_item(mask, tx, ty)
    return img, mask

    # elastic transform
    #img = elastic_transform(img, 666, 0.2)


def generate_image(items, image_dir, out_dir, angle_range = 25):
    global generated_count, annotation_count
    out_image = np.full((512, 512, 3), np.random.randint(255, size=3, dtype=np.uint8), np.uint8)
    image_data = { "id": generated_count, "file_name": str(generated_count) + '.jpg', "width": 512, "height": 512} # coco images section
    annotations_data = []
    for item in items:
        # place each of the item images in out_image
        img = cv2.imread(os.path.join(image_dir, item.img))
        cnt = item['contour']
        # rescale into 512 x 512
        cnt = (cnt * (out_image.shape[0] / img.shape[0])).astype(int)
        img = cv2.resize(img, (512, 512))
        mask = np.zeros((512,512,3), dtype=np.uint8)
        cv2.drawContours(mask, [cnt], 0, (255, 255, 255), -1)

        img, mask = apply_random_transforms(img, mask, angle_range)

        np.copyto(out_image, img, where=mask>0)

        annotations_data.append({
            "segmentation": [cnt.ravel().tolist()],
            "area": cv2.contourArea(cnt),
            "iscrowd": 0,
            "image_id": image_data['id'],
            "bbox": list(cv2.boundingRect(cnt)), 
            "category_id": item['category_id'],
            "id": annotation_count
            })
        annotation_count = annotation_count + 1 

        cv2.imshow('image', out_image)
        cv2.imshow('mask', mask)
        cv2.waitKey(200) 
    cv2.imwrite(os.path.join(out_dir, image_data['file_name']), out_image)
    generated_count = generated_count + 1
    return image_data, annotations_data

def main():
    args = parser.parse_args()
    print("loading item contours")
    items = pd.read_pickle(args.item_list)
    print("loaded", len(items), "with contours")
    print(items)

    dataset = {
            "info": {
                "description": "generated dataset",
                "date_created": str(datetime.datetime.now())
                },
            "images": [],
            "annotations": [],
            "categories": categories,
            }

    for _, item in items.iterrows():
        image, annotations = generate_image([item], args.image_dir, args.out_dir)
        dataset['images'].append(image)
        dataset['annotations'] = dataset['annotations'] + annotations

        
        with open('annotations.json', 'w') as f:
            json.dump(dataset, f)

    # find distribution of data
    #distribution = items.groupby('category_id').count() / len(items)
    #distribution = items['category_id'].value_counts() / len(items)
    #print(distribution)
    #return
    

    #for i in range(0, 100):
    #    sample = items.sample(1)
    #    print(sample)
    #    image_data, annotations_data = generate_image(sample, args.image_dir, args.out_dir)
    #    print(image_data, annotations_data)

    
if __name__ == '__main__':
    main()
