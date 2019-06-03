import numpy as np
import glob
import cv2

# prepares image backgrounds. im_size 
def random_noise_bg_gen(im_size):
    while True:
        yield np.full(im_size+(3,), np.random.randint(255, size=3, dtype=np.uint8), np.uint8)

def random_plain_bg_gen(im_size):
    while True:
        yield np.full(im_size+(3,), np.random.randint(255, size=3, dtype=np.uint8), np.uint8)

def indoor_scene_bg_gen(im_size, bg_img_dir, enforce_size=True):
    print('looking for backgrounds in {}'.format(bg_img_dir))
    image_files = glob.glob(bg_img_dir)
    print('backgrounds.py', len(image_files), 'background images found')
    for image_file in image_files:
        img = cv2.imread(image_file)
        if enforce_size:
            scale = min(im_size)/min(img.shape[:-1])
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        yield img

