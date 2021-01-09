import numpy as np
import cv2 as cv
import imutils
from NeuralNet import SavedNeuralNetMLP

def getSymb(x):
    na = np.array(x)
    na = na.astype('float32')
    na = na - 45
    na = na / 210
    symb = nn.predict([na])
    return symb

def getSubImage(rect, src):
    center, size, theta = rect
    center, size = tuple(map(int, center)), tuple(map(int, size))
    M = cv.getRotationMatrix2D( center, theta, 1)
    dst = cv.warpAffine(src, M, src.shape[:2])
    out = cv.getRectSubPix(dst, size, center)
    return out

def getCard(img):
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    thresh = cv.threshold(gray, 190, 255, cv.THRESH_BINARY)[1]
    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)[0]
    mask = 255*np.ones((img.shape), np.uint8)
    mask = cv.drawContours(mask, [cnts], -1, 0, cv.FILLED)
    img1 = cv.bitwise_or(mask, img)
    lab = cv.cvtColor(img1, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv.merge((cl,a,b))
    final = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
    gray = cv.cvtColor(final, cv.COLOR_RGB2GRAY)
    thresh = cv.threshold(gray, 195, 255, cv.THRESH_BINARY)[1]
    contours, hierarchy = cv.findContours(thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = []
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv.boundingRect(contour)
        if hierarchy[0][idx][3] == 0 and w > 20 and h > 20:
                    cnts.append(contour)
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)
    sizes = []
    symbs = []
    ks = []
    for s, cnt in enumerate(cnts):
        mask = 255*np.ones((img.shape), np.uint8)
        mask = cv.drawContours(mask, [cnt], -1, 0, cv.FILLED)
        img2 = cv.bitwise_or(mask, img)
        rect = cv.minAreaRect(cnt)
        area = abs(cv.contourArea(cnt))
        x,y = rect[0]
        w,h = rect[1]
        dense = int(255*area/w/h)
        prop = int(255*h/w) if w > h else int(255*w/h)
        out = getSubImage(rect, img2)
        b,g,r,_ = np.int0(cv.mean(out))
        symbs.append([prop, dense, b, g])
        sizes.append([s, max(w,h), cnt, int(x), int(y)])
    for x in list(x[0] for x in sorted(list(sorted(sizes,key=lambda l:l[1], reverse=True)))):
        n = getSymb(symbs[x])
        if n < 57 and not n in ks:
            ks.append(n)
        if len(ks) > 7:
            break
    return ks

nn = SavedNeuralNetMLP(filename = 'symbols.sav')

for i in range(1, 56):
    img1 = cv.imread('card%s.jpg' % (i))
    card1 = set(getCard(img1))
    for j in range(i+1, 56):
        img2 = cv.imread('card%s.jpg' % (j))
        card2 = set(getCard(img2))
        s = card1.intersection(card2)
        print(i, j, s.pop())
