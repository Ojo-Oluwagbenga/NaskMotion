import cv2
from .interpolator import stripcut, trunc_img_arr, stack
import numpy as np
import urllib
from PIL import Image, ImageDraw
import requests
from io import BytesIO

def createblank(w, h):
    img = np.zeros((int(h), int(w), 4), np.uint8)
    img.fill(255)
    return img

def tocv2(pilimage):
    return cv2.cvtColor(np.array(pilimage), cv2.COLOR_RGBA2BGRA)

def makeequal(imgs):

    maxh, maxw = imgs[0].shape[:2]

    for img in imgs:
        h,w = img.shape[:2]
        maxh = h if (h > maxh) else maxh
        maxw = w if (w > maxw) else maxw

    filled = []
    fixedheight = -1
    for img in imgs:
        h,w = img.shape[:2]
        defh = maxh - h
        defw = maxw - w
        
        
        if (defw//2 != 0):
            imfillside = createblank(defw//2, h)
            img = cv2.hconcat([imfillside, img, imfillside])

        if (defh//2 != 0):
            imfilltops = createblank(w + defw, defh//2)
            img = cv2.vconcat([imfilltops, img, imfilltops])
        
        h,w = img.shape[:2]
        img = img[0:maxh-2, 0:w]

        filled.append(img)

    return filled          


def cv2readimgonline(link):
    response = requests.get(link)
    img = Image.open(BytesIO(response.content))

    return tocv2(img)

def twinview(im1urls, width):
    # flip im1 image to left
    im1 = cv2readimgonline(im1urls[0])
    im2 = cv2readimgonline(im1urls[1])

    im1 = cv2.rotate(im1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # flip im2 image to right
    im2 = cv2.rotate(im2, cv2.ROTATE_90_CLOCKWISE)


    print(im1.shape, im2.shape)

    if (im1.shape != im2.shape):
        imgs = makeequal([im1,im2])
        im1 = imgs[0]
        im2 = imgs[1]

    

    tarr = trunc_img_arr(im1, width, 0, 0)
    tarr2 = trunc_img_arr(im2, width, 0, 0)

    count = len(tarr) if (len(tarr) < len (tarr2)) else len (tarr2)

    truncall = []
    for i in range(count):
        truncall.append(tarr[i])
        truncall.append(tarr2[i])

    interpolImage = stack(truncall)


    return interpolImage

def test():
    pass


 