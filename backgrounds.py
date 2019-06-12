import numpy as np
import glob
import cv2

BG_RNG = np.random.RandomState(seed=0)

# prepares image backgrounds. im_size 
def random_noise_bg_gen(im_size):
    while True:
        extra_width = BG_RNG.randint(im_size[0]*0.25, im_size[0]*0.5)
        im_new_size = (im_size[0] , im_size[1] + extra_width)
        yield BG_RNG.randint(0, 255, size=im_new_size+(3,), dtype=np.uint8)

def random_plain_bg_gen(im_size):
    while True:
        extra_width = BG_RNG.randint(im_size[0]*0.25, im_size[0]*0.5)
        im_new_size = (im_size[0] , im_size[1] + extra_width)
        yield np.full(im_new_size+(3,), BG_RNG.randint(255, size=3, dtype=np.uint8), np.uint8)

def indoor_scene_bg_gen(im_size, bg_img_dir, enforce_size=True):
    print('looking for backgrounds in {}'.format(bg_img_dir))
    image_files = glob.glob(bg_img_dir)
    print('backgrounds.py', len(image_files), 'background images found')
    for image_file in image_files:
        img = cv2.imread(image_file)
        if img is None:
            print('failed to read {}'.format(image_file))
            continue;
        if enforce_size:
            scale = min(im_size)/min(img.shape[:-1])
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        yield img

