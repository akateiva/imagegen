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

MAX_SCALE = 0.7
MIN_SCALE = 0.3

ROI_RNG = np.random.RandomState(seed=0)
ROTATION_RNG = np.random.RandomState(seed=0)
HUE_SHIFT_RNG = np.random.RandomState(seed=0)
SAMPLING_RNG = np.random.RandomState(seed=0)


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

def get_roi_iou(roi_a, roi_b):
    xA = int(max(roi_a[0] - roi_a[2] / 2 , roi_b[0] - roi_b[2] / 2))
    yA = int(max(roi_a[1] - roi_a[3] / 2, roi_b[1] - roi_b[3] / 2 ))
    xB = int(min(roi_a[0] + roi_a[2] / 2 , roi_b[0] + roi_b[2] / 2))
    yB = int(min(roi_a[1] + roi_a[3] / 2, roi_b[1] + roi_b[3] / 2 ))

    intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    roi_a_area = roi_a[2] * roi_a[3]
    roi_b_area = roi_b[2] * roi_b[3]
    iou = intersection_area / float(roi_a_area + roi_b_area - intersection_area)
    return iou

# returns the center x and y, as well as width and height for a random ROI in an image
def get_roi(target_im_shape):
    max_size = min(target_im_shape[:-1]) * MAX_SCALE
    min_size = min(target_im_shape[:-1]) * MIN_SCALE
    w = h = ROI_RNG.randint(min_size, max_size)
    min_x_position = int(w/2) + 1
    max_x_position = int(target_im_shape[1] - w/2)-1
    min_y_position = int(h /2) + 1
    max_y_position = int(target_im_shape[0] - h/2)-1
    x = ROI_RNG.randint(min_x_position, max_x_position)
    y = ROI_RNG.randint(min_y_position, max_y_position)
    return x, y, w, h

def get_rois(im_shape, n_rois, max_iou=0.25):
    rois = []
    while len(rois) < n_rois:
        candidate_roi = get_roi(im_shape)
        candidate_doesnt_overlap = True
        candidate_placement_attempts = 0
        for roi in rois:
            iou = get_roi_iou(roi, candidate_roi)
            print(iou)
            if iou > max_iou:
                candidate_doesnt_overlap = False
        if candidate_doesnt_overlap or candidate_placement_attempts > 5:
            rois.append(candidate_roi)
    return rois

def replace_background(target, mask, background_scalar):
    result = np.full(target.shape, fill_value=background_scalar, dtype=np.uint8)
    np.copyto(result, target, where=mask>0)
    return result

def paste_item(item, target, roi=None):
    x, y, w, h = roi
    
    # 1. read image, then rescale it and its polygon to the ROI
    src = cv2.imread(os.path.join(args.image_dir, item.img))
    if (src is None or target is None):
        raise ValueError("Image {} could not be read or target image is none!.".format(item.img))

    cnt = (item['contour'] * (w / src.shape[0])).astype(int)
    src = cv2.resize(src, (w, h))    

    # 2. prepare mask matrix
    mask = np.zeros(src.shape, dtype=np.uint8)
    cv2.drawContours(mask, [cnt], 0, (255, 255, 255), -1)

    # 3. perform rotation
    ANGLE_RANGE = 25
    angle = ROTATION_RNG.randint(-ANGLE_RANGE, ANGLE_RANGE+1)
    src = rotate_item(src, angle)
    mask = rotate_item(mask, angle)

    # 4. Mask Fine-Tuning
    mask = cv2.erode(mask, kernel, iterations=2)
    src = replace_background(src, mask, 150 )   # replace background to grey for poisson blending 

    mask_area = mask.sum()/255/3
    assert mask_area > 32*32, 'Mask area under 32x32'
    print('mask area: {}'.format(mask.sum()/255/3))

    # 5. Elastic Transform
    if args.elastic_transform:
        # convert to float64 before transforming to avoid uint8 artifacts
        [src, mask] = elasticdeform.deform_random_grid([src.astype('float64'), mask.astype('float64')], points=3, sigma=w/13, axis=[(0, 1), (0, 1)])
        src = src.clip(0, 255).astype('uint8')
        mask = mask.clip(0, 255).astype('uint8')

    # 6. Hue Shift
    if args.hue_shift:
        hue_shift_amount = HUE_SHIFT_RNG.randint(0, 180, dtype=np.uint8)
        src = hue_shift(src, hue_shift_amount)

    # 7. Poisson Blending ( Seamless Cloning )
    if args.ps_blend:
        mask = cv2.dilate(mask, kernel, iterations=2)
        target = cv2.seamlessClone(src, target, mask, (x,y), cv2.NORMAL_CLONE)
    else:
        # Simple copy
        np.copyto(target[int(y - h/2):int(y + h/2), int(x - w/2):int(x + w/2), :], src, where=mask>0)

    # 8. Debug Display

    # 9. Convert mask back to polygon
    segmentation = mask_to_poly(mask) + [int(x - w/2),int(y - h/2)]
    #cv2.drawContours(target, [segmentation], 0, (255, 0, 255), 5) # for debugging
    return target, segmentation


# takes the given items and prepares the image and annotations
def generate_training_sample(items,
        bg_generator,
        angle_range=25):
    img = next(bg_generator)
    rois = get_rois(img.shape, len(items))
    img_data = dataset.allocate_image(img.shape[0:-1])
    for roi, (_, item) in zip(rois, items.iterrows()):
        try:
            img, segmentation = paste_item(item, img, roi)
            if segmentation is not None:
                dataset.add_annotation({
                    "segmentation": [segmentation.ravel().tolist()],
                    "area": cv2.contourArea(segmentation),
                    "iscrowd": 0,
                    "image_id": img_data['id'],
                    "bbox": list(cv2.boundingRect(segmentation)),
                    "category_id": item['category_id']
                    })
        except AssertionError as error:
            print('sample generation failed ', item, error)
    #img = cv2.blur(img,(5,5))
    return img, img_data
print(args)
IM_SIZE = (args.size, args.size)
## 1. Prepare background generator
if args.bg == 'plain':
    bg_gen = backgrounds.random_plain_bg_gen(IM_SIZE)
elif args.bg == 'noise':
    bg_gen = backgrounds.random_noise_bg_gen(IM_SIZE)
elif args.bg == 'indoor' or args.bg == 'image':
    bg_gen = backgrounds.indoor_scene_bg_gen(IM_SIZE, args.bg_img_dir)
else:
    raise ValueError("No such background generator: {}".format(args.bg))

## 2. Load item list with contours and etc., compute weights for uniform sampling
items = pd.read_pickle(args.item_list)
print("Loaded", len(items), "with contours")
category_item_weights = items.groupby('category_id')['img'].count().apply(lambda x: 1/x).rename('item_weight')
items = items.join(category_item_weights, on='category_id')
print("Args", args)

## 3. Build dataset
dataset = cocoset.COCOSet(parameters_info=vars(args))
os.makedirs(os.path.join(args.out_dir, 'images/'), exist_ok=True)
for i in range(0, args.n_samples):
    print("{} / {} ".format(i, args.n_samples))
    sample_items = items.sample(args.n_items_per_sample, weights=items['item_weight'], random_state=SAMPLING_RNG)
    img, img_data = generate_training_sample(sample_items, bg_gen)
    cv2.imwrite('{}/images/{}'.format(args.out_dir, img_data['file_name']), img)
    if args.debug:
        cv2.imshow('generated', img)
        cv2.waitKey(10000)
print("writing annotations.json")
dataset.write_annotations('{}/annotations.json'.format(args.out_dir))
