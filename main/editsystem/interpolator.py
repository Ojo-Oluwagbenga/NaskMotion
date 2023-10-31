import cv2
import os
import numpy as np
from PIL import Image
import requests
from io import BytesIO
# from .c_interpolator import makeequal


def createblank(w, h):
    img = np.zeros((int(h), int(w), 4), np.uint8)
    img.fill(255)
    return img

def tocv2(pilimage):
    return cv2.cvtColor(np.array(pilimage), cv2.COLOR_RGBA2BGRA)

def generateMask(img2, viswidth, darkwidth):
    height, width = img2.shape[:2]

    imgD = Image.new(mode="RGBA", size=(darkwidth, height))
    imgV = Image.new(mode="RGBA", size=(viswidth, height))
    

    band = viswidth+darkwidth
    dark = []
    vis = []
    for i in range(height):
        for v in range(viswidth): 
            vis.append((0, 0, 0, 0))
        for d in range(darkwidth):
            dark.append((20, 20, 20)) 

    imgD.putdata(dark)
    imgD.save("transD.png")

    imgV.putdata(vis)
    imgV.save("transV.png", 'PNG')

    imgV = cv2.imread("transD.png", -1)
    imgD = cv2.imread("transV.png", -1) 

    os.remove("transD.png")
    os.remove("transV.png")

    packarray = []
    for i in range((width//band)+1):
        packarray.append(imgV)
        packarray.append(imgD)

        
    return cv2.hconcat(packarray)

def pp(vr, vane="Def" ):
    print(vane+ " ----| " + str(vr) + " |-----")

def stack(slicedpics):
    for i in slicedpics:
        print(i.shape)
    return cv2.hconcat(slicedpics)

def stripcut(img, start, width):
    h,w = img.shape[:2]
    offset = start + width
    if (offset < w):
        return img[0:h, start:offset]
    else:
        return img[0:h, start:(w-offset)]
        print('passed') 

def getsnaps(vid, count):
    length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cdiff = length//count

    ins = 0
    fr = 0
    ret = []

    while(True):
        success, frame = vid.read()

        if (success):
            if (fr == (ins)*cdiff):
                ins += 1
                ret.append(frame)    
        
            fr += 1
        else:
            break
        if (ins == count):
            break

    vid.release()
    
    return ret

def trunc_img_arr(img, viswidth, dwidth, offset):
    bandwidth = viswidth + dwidth
    cutcount = img.shape[1]//(bandwidth)

    fr = offset

    ret = []

    for band in range(cutcount):
        ret.append(stripcut(img, fr, viswidth))
        fr += bandwidth
        
    return ret

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
        img = img[0:maxh-1, 0:w]

        filled.append(img)

    return filled          


def cv2readimgonline(link):
    response = requests.get(link)
    img = Image.open(BytesIO(response.content))

    return tocv2(img)

def motionimagery(snaps, darkwidth):
    # snaps are sent as an online link to images

    retsnaps = []
    for i in snaps:
        retsnaps.append(cv2readimgonline(i))

    snaps = retsnaps
    filmcount = len(snaps)
    snaps = makeequal(snaps)
    
    if (filmcount > 1):
        viswidth = darkwidth//(filmcount-1)
    elif(filmcount > 0):
        viswidth = darkwidth
        darkwidth = 0
    else: 
        return []

    ind = 0
    truncpack = []
    truncall = []
    minlen = -1 
    darkwidth = viswidth*(filmcount-1)


    for img in snaps:

        tarr = trunc_img_arr(img, viswidth, darkwidth, ind*viswidth)

        if (minlen > len(tarr) or minlen == -1):
            minlen = len(tarr) 

        truncpack.append(tarr)
        ind += 1
    
    
    for x in range(minlen):
        for pack in truncpack:
            truncall.append(pack[x])

    interpolImage = stack(truncall)
    gmask = generateMask(interpolImage, viswidth, darkwidth)


    return([interpolImage, gmask])
    
# interpolPack = interpolate('rotate.mp4')
 
try:
    pass
    
    # cv2.imwrite(path+"/ipimage.png", interpolPack[0])
    # cv2.imwrite(path+"/ipmask.png", interpolPack[1])

except Exception as e:
    print(e)


cv2.waitKey(0)
cv2.destroyAllWindows() 