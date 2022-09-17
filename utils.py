import numpy as np
import cv2
from numpy import random

def do_random_flip(image):
    if np.random.rand()>0.5:
        image = cv2.flip(image,0)
    if np.random.rand()>0.5:
        image = cv2.flip(image,1)
    if np.random.rand()>0.5:
        image = image.transpose(1,0,2) # y=x 대칭 #(0,1,2)->(1,0,2)
    image = np.ascontiguousarray(image)
    return image

def do_random_rot90(image):
    r = np.random.choice([
        0,
        cv2.ROTATE_90_CLOCKWISE,
        cv2.ROTATE_90_COUNTERCLOCKWISE,
        cv2.ROTATE_180,
    ])
    if r==0:
        return image
    else:
        image = cv2.rotate(image, r)
        return image
        
def do_random_contast(image, mag=0.3):
    alpha = 1 + random.uniform(-1,1)*mag # -0.3 ~ 0.3 -> 0.7~1.3 
    image = image * alpha
    image = np.clip(image,0,1)
    return image

def do_random_hsv(image, mag=[0.15,0.25,0.25]):
    image = (image*255).astype(np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0].astype(np.float32)  # hue
    s = hsv[:, :, 1].astype(np.float32)  # saturation
    v = hsv[:, :, 2].astype(np.float32)  # value
    h = (h*(1 + random.uniform(-1,1)*mag[0]))%180
    s =  s*(1 + random.uniform(-1,1)*mag[1])
    v =  v*(1 + random.uniform(-1,1)*mag[2])

    hsv[:, :, 0] = np.clip(h,0,180).astype(np.uint8)
    hsv[:, :, 1] = np.clip(s,0,255).astype(np.uint8)
    hsv[:, :, 2] = np.clip(v,0,255).astype(np.uint8)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    image = image.astype(np.float32)/255
    return image

def do_random_noise(image, mag=0.1):
    height, width = image.shape[:2]
    noise = np.random.uniform(-1,1, (height, width,1))*mag
    image = image + noise
    image = np.clip(image,0,1)
    return image

def do_random_rotate_scale(image, angle=30, scale=[0.8,1.2] ):
    angle = np.random.uniform(-angle, angle)
    scale = np.random.uniform(*scale) if scale is not None else 1
    
    height, width = image.shape[:2]
    center = (height // 2, width // 2)
    
    transform = cv2.getRotationMatrix2D(center, angle, scale)
    image = cv2.warpAffine( image, transform, (width, height), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    return image

def train_augment5b(image):
    image = do_random_flip(image)
    image = do_random_rot90(image)

    for fn in np.random.choice([
        lambda image: (image),
        lambda image: do_random_noise(image, mag=0.1),
        lambda image: do_random_contast(image, mag=0.40),
        lambda image: do_random_hsv(image, mag=[0.40, 0.40, 0])
    ], 2): image = fn(image)

    for fn in np.random.choice([
        lambda image: (image),
        lambda image: do_random_rotate_scale(image, angle=45, scale=[0.50, 2.0]),
    ], 1): image = fn(image)

    return image