import numpy as np
import cv2 as cv
import imutils
import os

def getSubImage(rect, src):
    # https://stackoverflow.com/questions/11627362/how-to-straighten-a-rotated-rectangle-area-of-an-image-using-opencv-in-python
    # Get center, size, and angle from rect
    center, size, theta = rect
    # Convert to int 
    center, size = tuple(map(int, center)), tuple(map(int, size))
    # Get rotation matrix for rectangle
    M = cv.getRotationMatrix2D( center, theta, 1)
    # Perform rotation on src image
    dst = cv.warpAffine(src, M, src.shape[:2])
    out = cv.getRectSubPix(dst, size, center)
    return out

for i in range(1, 56):
    img = cv.imread('card%s.jpg' % (i))
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    thresh = cv.threshold(gray, 190, 255, cv.THRESH_BINARY)[1]
    # find contours
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
    for cnt in cnts:
        mask = np.zeros((img.shape), np.uint8)
        mask.fill(255)
        mask = cv.drawContours(mask, [cnt], -1, 0, cv.FILLED)
        img2 = cv.bitwise_or(mask, img)
        rect = cv.minAreaRect(cnt)
        area = abs(cv.contourArea(cnt))
        x,y = rect[0]
        w,h = rect[1]
        dense = int(255*area/w/h)
        prop = int(255*h/w) if w > h else int(255*w/h)
        sizes.append([prop, max(w,h)])
        out = getSubImage(rect, img2)
        b,g,r,_ = np.int0(cv.mean(out))
        print(prop, dense, b, g)
        if not os.path.exists(str(prop)):
            os.makedirs(str(prop))
        cv.imwrite('%s\\%s_%s_%s_%s_card%s.bmp' % (prop, prop, dense, b, g, i), out)
        # cv.imshow("Symbol", out)
        # cv.waitKey(0)
