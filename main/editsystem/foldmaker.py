import cv2
import numpy
from PIL import Image, ImageDraw
import time
# import imutils
import math
import requests
from io import BytesIO
import base64
import cv2
import numpy as np

from .c_interpolator import createblank
# from .c_interpolator import createblank


def pilreadimgonline(link):
    response = requests.get(link)
    img = Image.open(BytesIO(response.content))

    return img

def p2linetrim(img, sv):
    # sv meaning shape vertex is a tuple collection of points
    pass
def calc_dist(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    return pow( (pow(dx,2) + pow(dy,2)), 0.5)
def eqn_gen(p1, p2):
    x1 = p1[0]
    x2 = p2[0]
    y1 = p1[1]
    y2 = p2[1]
    k = (y1 - y2)/(x1 - x2)
    dist = calc_dist(p1, p2)

    return lambda x: {(k*(x - x1) + y1), p1, p2, dist}
def is_on_line(p3, linefunc):
    # line func is the lamda return from eqn_gen
    x = p3[0]
    y = p3[1]
    lp = linefunc(x)
    if (int(lp[0]) == y):
        p3 = (x,lp[0])
        return (lp(3) == (calc_dist(p3, lp[1]) + calc_dist(p3, lp[2])))
    else:
        return False


def vect(A, B):
    return (B[0]-A[0], B[1]-A[1])
def vectmag(vX):
    return pow(pow(vX[0], 2) + pow(vX[1], 2), 0.5)
def vectarg(vX):
    # mag = vectmag(vX)
    return getang(vX)
def addvect(vA, vB):
    return (vB[0]+vA[0], vB[1]+vA[1])
def subvect(vB,vA):
    # subs A from B
    return (vB[0] - vA[0], vB[1] - vA[1])
def rotvect(vX, ang):
    mag = vectmag(vX)
    arg = vectarg(vX)
    nx = mag*math.cos(ang + arg)
    ny = mag*math.sin(ang + arg)
    return (nx, ny)
def getquad(point):
    ret = 1
    x = point[0]
    y = point[1]
    # print(point)
    if (point[0] <= 0 and point[1] >= 0):
        ret = 2
    if (point[0] <= 0 and point[1] <= 0):
        ret = 3
    if (point[0] >= 0 and point[1] <= 0):
        ret = 4

    return ret
def getang(p):
    ang = math.atan(abs(p[1]/p[0])) if (p[0] != 0) else 90
    q = getquad(p)

    # print(q)
    if (q == 2):
        ang = torad(180) - ang
    if (q == 3):
        ang += torad(180)
    if (q == 4):
        ang = torad(360) - ang

    return ang

def rotate_about_p(p1, p2, ang):
    # Rotates p2 around p1 with angle ang
    # inverse the y pos to mean negative
    p1 = (p1[0], -p1[1])
    p2 = (p2[0], -p2[1])
    mag = vectmag(vect(p1,p2))
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    arg = getang((dx, dy))

    nx = mag*math.cos(ang+arg)
    ny = mag*math.sin(ang+arg)

    return (nx, -ny)
def tocv2(pilimage):
    return cv2.cvtColor(numpy.array(pilimage), cv2.COLOR_RGBA2BGRA)
def topil(cv2image):
    img = cv2.cvtColor(cv2image, cv2.COLOR_BGRA2RGBA)
    return Image.fromarray(img)
def todeg(arcval):
    return math.degrees(arcval)
def torad(arcval):
    return math.radians(arcval)
def resize(img, scale):
    # Image should be parsed as cv2 Read Image
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)

    return cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)
def crop(img, sd, width, height):
    # Start cord in array of x-axis, y-axis
    # Height is taken downward from start
    # Width is taken to right from start
    # Image should be parsed as cv2 Read Image

    x=sd[0]
    y=sd[1]
    return img[y:y+height, x:x+width]
def pilcrop(img, xsf, ysf):
    # xsf is in form (a,b) where a is x start annd b is x end
    return img.crop((xsf[0], ysf[0], xsf[1],  ysf[1]))
def pilboxscale(img, scale):
    w,h = img.size

    return img.resize((int(w*scale), int(h*scale)))
def equationgen(p1, p2):
    x1 = p1[0]
    x2 = p2[0]
    y1 = p1[1]
    y2 = p2[1]
    k = 10000 if (x1 - x2 == 0) else (y1 - y2)/(x1 - x2)

    return [lambda x: int(k*(x - x1) + y1), k]

def draw_dash(p1, p2, thickness, img):
    # dlen is the dash length
    # space is the space length
    dlen = 60
    space = 50
    draw = ImageDraw.Draw(img)

    st,end = (p2[0], p1[0]) if (p2[0] < p1[0]) else (p1[0], p2[0])
    if (abs(end - st) > 30):
        funcpack = equationgen(p1, p2)
        func = funcpack[0]
        k = funcpack[1]
        xlen = dlen * pow(1/(1+pow(k,2)), 0.5)
        slen = (space/dlen)*xlen

        # print("xlen", xlen)

        xlen = int (xlen) if xlen < 2 else int(xlen - 1)
        slen = int(slen) + xlen + 2



        for x in range(int(st), int(end), slen):
            y = int(func(x))
            y2 = int(func(x + xlen))
            # draws line on the passed image
            draw.line((x,y, x+xlen,y2), fill=(0,0,0), width= thickness)

    elif (abs(end - st) == 0):
        st,end = (p2[1], p1[1]) if (p2[1] < p1[1]) else (p1[1], p2[1])
        for y in range(int(st), int(end), space+dlen):
            x = p2[0]
            y2 = y+space
            draw.line((x,y, x,y2), fill=(0,0,0), width= thickness)
    else:
        draw.line((p1[0],p1[1], p2[0],p2[1]), fill='grey', width= 1)
        # st,end = (p2[1], p1[1]) if (p2[1] < p1[1]) else (p1[1], p2[1])
        # bunds = space+dlen
        # ranger = abs(end - st)/bunds
        # prevx = 0
        # p = 0
        #

    pass

def draw_line(p1, p2, thickness, img):
    # dlen is the dash length
    # space is the space length
    draw = ImageDraw.Draw(img)
    draw.line((p1[0],p1[1], p2[0],p2[1]), fill=(0,0,0), width= thickness)

def cuttoshape(img, polygon):
    # polygon in form of the array of point vertices
    # convert to numpy (for convenience)
    # Polygon in form of [(y, x), (y2, x2), (y3, x3)]

    w,h = img.size

    po = polygon[0]
    xmin = po[0]
    ymin = po[1]

    xmax = xmin
    ymax = ymin
    for p in polygon:
        x = p[0]
        y = p[1]

        if (x < xmin):
            xmin = x

        if (x > xmax):
            xmax = x

        if (y < ymin):
            ymin = y

        if (y > ymax):
            ymax = y

    maxw = xmax - xmin
    maxh = ymax - ymin


    # Select Scale of the image respecting the one that needs zoom the most
    scaledata = [maxw/w, 'w'] if (maxw/w > maxh/h) else [maxh/h, 'h']
    scale = scaledata[0]

    h2 = h*scale
    w2 = w*scale


    img = pilboxscale(img, scale)

    # I need to trim the excess on due to scaling
    # offset is the distance from the edge of the picture
    if (scaledata[1] == 'w'):
        # meaning w is exact and h is bigger than required
        # We'll have to equally crop top and bottom
        offset = (h2-maxh)//2
        img = pilcrop(img, (0, w2), (offset, offset + maxh))

    if (scaledata[1] == 'h'):
        # meaning h is exact and w is bigger than required
        # We'll have to equally crop left and right
        offset = (w2-maxw)//2
        img = pilcrop(img, (offset, offset+maxw), (0, h2))


    imArray = numpy.asarray(img)
    # create mask
    maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
    ImageDraw.Draw(maskIm).polygon(polygon, outline ="blue", fill='blue')
    mask = numpy.array(maskIm)

    # assemble new image (uint8: 0-255)
    newImArray = numpy.empty(imArray.shape,dtype='uint8')

    # colors (three first columns, RGB)
    newImArray[:,:,:3] = imArray[:,:,:3]

    # transparency (4th column)
    newImArray[:,:,3] = mask*255

    # back to Image from numpy
    return Image.fromarray(newImArray, "RGBA")

def cuttoshape_noscale(img, polygon):
    # polygon in form of the array of point vertices
    # convert to numpy (for convenience)
    # Polygon in form of [(y, x), (y2, x2), (y3, x3)]

    po = polygon[0]
    xmin = po[0]
    ymin = po[1]

    xmax = xmin
    ymax = ymin
    for p in polygon:
        x = p[0]
        y = p[1]

        if (x < xmin):
            xmin = x

        if (x > xmax):
            xmax = x

        if (y < ymin):
            ymin = y

        if (y > ymax):
            ymax = y

    imArray = numpy.asarray(img)
    # create mask
    maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
    ImageDraw.Draw(maskIm).polygon(polygon, outline ="blue", fill='blue')
    mask = numpy.array(maskIm)

    # assemble new image (uint8: 0-255)
    newImArray = numpy.empty(imArray.shape,dtype='uint8')

    # colors (three first columns, RGB)
    newImArray[:,:,:3] = imArray[:,:,:3]

    # transparency (4th column)
    newImArray[:,:,3] = mask*255

    # back to Image from numpy
    img = Image.fromarray(newImArray, "RGBA")


    return pilcrop(img, (xmin, xmax), (ymin, ymax))


def _64tocv2(b64):
    b64 = b64.replace('data:image/jpeg;base64,', '')

    im_bytes = base64.b64decode(b64)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    img_cv2 = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    return img_cv2

def storyfold(b64s, **kwargs):
    retsnaps = []
    img = ''

    sub_imgdata = {
        'A':img,
        'B':img,
        'C':img,
        'D':img,
        'E':img,
    }
    imgdata = {}
    ic = 0
    for key, val in sub_imgdata.items():
        imgdata[key] = topil(_64tocv2(b64s[ic]))
        ic += 1

    w = kwargs.get('width') #This is the width that will be taken by the origami
    l = kwargs.get('gap') #This is the l in the fold vertices

    t = w/3
    tt = torad(22.5)

    # E face param/
    root2 = pow(2,0.5)
    a = (w/root2)*math.sin(tt)
    x = (w/root2)*math.cos(tt)
    k = w/2+t
    y = (k/math.cos(tt)) - x
    m = (k*math.tan(tt))
    c = w/2 - m
    p = w*math.tan(tt)/2

    data = {
        'A':[
            [(0,0), (w,0), (w,w/2), (0, w/2)],
            [
                [
                    [(0,0), (w,0), (w,w/2), (0, w/2)],
                    0,(0,0),(0,0),'f'
                ]
            ]
        ],
        'B':[
            [(w/2,0), (w,w/2), (w,w/2+2*l), (w/2+l,w+l), (w/2-l,w+l), (0,w/2+2*l), (0,w/2)],
            [
                [
                    [(w/2,0), (w/2,w/2), (0,w/2)],
                    90,(w/2,w/2),(w,0),'b'
                ],
                [
                    [(w/2,0), (w/2,w/2), (w,w/2)],
                    -90,(0,w/2),(0,0),'b'
                ],
                [
                    [(0,w/2),(w, w/2), (w,w/2+2*l), (w/2+l,w+l), (w/2-l,w+l), (0,w/2+2*l),],
                    0,(0,0),(0,w/2),'f'
                ]
            ]
        ],
        'C':[
            [(0,0), (w,0), (w,w/2), (0,w/2)],
            [
                [
                    [(0,0), (w,0), (w,l), (w/2+l,w/2), (w/2-l,w/2), (0,l)],
                    180,(w,0),(0,w/2+l),'b'
                ],
                [
                    [(0,l), (w/2-l,w/2), (0,w/2)],
                    0,(0,w/2-l),(0,w+l),'f'
                ],
                [
                    [(w,l), (w/2+l,w/2), (w,w/2)],
                    0,(w/2-l,w/2-l),(w,w+l),'f'
                ]
            ]
        ],
        'D':[
            [(w/2,0), (w,w/2), (w,w/2+t), (0,w/2+t), (0,w/2)],
            [
                [
                    [(w/2,0), (w/2,w/2), (0,w/2)],
                    90,(0,w/2),(w,w+l),'b'
                ],
                [
                    [(w/2,0), (w/2,w/2), (w,w/2)],
                    -90,(w/2,w/2),(0,w+l),'b'
                ],
                [
                    [(0,w/2), (w,w/2), (w,w/2+t), (0,w/2+t)],
                    0,(0,0),(0,w+l),'f'
                ],
                [
                    [(w/2-l,w/2), (w/2+l,w/2), (w/2,w/2+l)],
                    180,(0,0),(w/2+l,l),'b'
                ]
            ]
        ],
        'E':[
            [(a,0), (2*a,x), (a+c,x+y), (a-c,x+y), (0,x)],
            [
                [
                    [(a,0), (0,x), (a-c,x+y), (a,x+y)],
                    -22.5,(a,0),(w/2,w/2+l),'b'
                ],
                [
                    [(a,0), (2*a,x), (a+c,x+y), (a,x+y)],
                    22.5,(0,0),(w/2,w/2+l),'b'
                ],
            ]
        ],
    }

    dashvectors_front = [
        [(w/2, 0), (0,w/2)],
        [(w/2, 0), (w,w/2)],
        [(0, w/2+l), (w,w/2+l)],
        [((w/4-l/2), 3*w/4 + 3*l/2), (0,w+l)],
        [((3*w/4+l/2), 3*w/4 + 3*l/2), (w,w+l)],
        [(w/2, w+l), (w/2,w+l+t)],
        [(w/2-p, w+l), (w/2-m,w+l+t)],
        [(w/2+p, w+l), (w/2+m,w+l+t)],
    ]
    dashvectors_back = [
        [(3*w/4+l/2, w/4+l/2), (w/2,w/2+l)],
        [(w/4-l/2, w/4+l/2), (w/2,w/2+l)],
        [(w/2,w/2+l), (w,w/2+p)],
        [(w/2,w/2+l), (0,w/2+p)],
        [(w/2-l,l), (w/2+l,l)],
    ]
    linevectors_front = [
        [(0, w/2), (w,w/2)],
        [(0, w/2+2*l), (w/2-l,w+l)],
        [(w, w/2+2*l), (w/2+l,w+l)],
        [(0, w+l), (w/2-l, w+l)],
        [(w/2+l, w+l), (w, w+l)],
    ]

    parfront = topil(createblank(w, w+l+t))
    parback = topil(createblank(w, w+l+t))


    for key, wholeimg in imgdata.items():
        packet = data[key]
        fragpack = fragment(wholeimg, packet)

        for ind, frag in enumerate(fragpack):
            sub_im = frag[0]
            cd = frag[1]
            if frag[2] == 'f':
                parfront.paste(sub_im, (cd[0], cd[1]), sub_im)
            else:
                parback.paste(sub_im, (cd[0], cd[1]), sub_im)

    dlen = 10
    space = 5

    for dashline in dashvectors_front:
        draw_dash(dashline[0], dashline[1], 10, parfront)

    for dashline in dashvectors_back:
        draw_dash(dashline[0], dashline[1], 10, parback)

    for line in linevectors_front:
        draw_line(line[0], line[1], 20, parfront)

    return (parfront, parback)

def fragment(img, param):
    # param is an Dict that contains the major cut
        # and the minor cut with transformations of the minor
        # in form of
        # [
        #     [(1,2), (10,7), (20,9)],
        #     [ #This will contain info about refragmentation
        #         [
        #             [(10, 10), (12, 11)], # ['--The sub cut---'],
        #             20, #degree of rot
        #             (10,10), #fixture point of child. Note, transformation is done on this coord as well
        #             (10,10), #fixture point on parent. Note, transformation is done on this coord as well
        #              1 #This is the fixture page, 1 for front and 0 for back
        #         ],
        #         [
        #             [(10, 10), (12, 11)], # ['--The sub cut---'],
        #             20, #degree of rot
        #             (10,10), #fixture point. Note, transformation is done on this coord as well
        #             (10,10), #fixture point of parent. Note, transformation is done on this coord as well
        #              1 #This is the fixture page, 1 for front and 0 for back
        #         ],
        #     ]
    # ]

    majcut = cuttoshape(img, param[0])

    majcut.save('out1.png')

    fragpack = []

    for subcuts in param[1]:
        subimg = cuttoshape_noscale(majcut, subcuts[0])



        offdata = getpos_onroll(subimg, subcuts[1], subcuts[2])

        transx = subcuts[3][0] + offdata[1] #This is calculating the resultant append position-x on the parent
        transy = subcuts[3][1] + offdata[2] #This is calculating the resultant append position-y on the parent

        fragpack.append([offdata[0], (int(transx), int(transy)), subcuts[4]])

    return fragpack

def getpos_onroll(img, angle, axis):
    # axis has to be in the form of 2-tuple
    # axis is the fixture point on the child also where the rotation will be performed
    angle = torad(angle)
    w,h = img.size

    O = (0, 0) #Fixed point

    ixc = w//2
    iyc = h//2
    iC = (ixc, iyc) #Initial center before transform

    iR = axis

    rav = addvect(vect(O,iC), vect(iC,iR)) #RAV as in Rotation Axis Vector
    img = img.rotate(todeg(angle), expand = True, resample=Image.BICUBIC)

    w,h = img.size
    fxc = w//2
    fyc = h//2
    fC = (fxc, fyc) #Final center after transform


    rotvec = rotate_about_p(iC, iR, angle)

    fR = (fC[0] + rotvec[0], fC[1] + rotvec[1])


    nrav = addvect(vect(O,fC), vect(fC,fR))

    offset = subvect((0,0), nrav)

    offsetx = offset[0]
    offsety = offset[1]


    # offsetx is the extra distance right we want move from specified origin(Top left)
    # offsety is the extra distance down we want move

    return (img, offsetx, offsety)

def test(rot):

    img = Image.open("name.jpg").convert("RGBA")
    data = interpolateFold({
        'A':img,
        'B':img,
        'C':img,
        'D':img,
        'E':img,
    }, width=400, gap=40)
    data[0].save('front.png')
    data[1].save('back.png')
