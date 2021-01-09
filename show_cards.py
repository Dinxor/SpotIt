import numpy as np
import cv2 as cv
import imutils
from NeuralNet import SavedNeuralNetMLP

names = {0: 'Anchor', 1: 'Apple', 2: 'Baby bottle', 3: 'Bomb', 4: 'Cactus', 5: 'Candle', 6: 'Taxi car', 7: 'Carrot', 8: 'Chess knight', 9: 'Clock', 10: 'Clown', 11: 'Diasy flower', 12: 'Dinosaur', 13: 'Dolphin', 14: 'Dragon', 15: 'Exclamation point', 16: 'Eye', 17: 'Fire', 18: 'Four leaf clover', 19: 'Ghost', 20: 'Green splats', 21: 'Hammer', 22: 'Heart', 23: 'Ice cube', 24: 'Igloo', 25: 'Key', 26: 'Ladybird (Ladybug)', 27: 'Light bulb', 28: 'Lightning bolt', 29: 'Lock', 30: 'Maple leaf', 31: 'Moon', 32: 'No Entry sign', 33: 'Orange scarecrow man', 34: 'Pencil', 35: 'Purple bird', 36: 'Purple cat', 37: 'Purple dobble hand man', 38: 'Red lips', 39: 'Scissors', 40: 'Skull and crossbones', 41: 'Snowflake', 42: 'Snowman', 43: 'Spider', 44: 'Spiders web', 45: 'Sun', 46: 'Sunglasses', 47: 'Target', 48: 'Tortoise', 49: 'Treble clef', 50: 'Tree', 51: 'Water drip', 52: 'Dog', 53: 'Yin and Yang', 54: 'Zebra', 55: 'Question mark', 56: 'Cheese', 57: 'Green Dot', 58: 'Yellow Dot'}

cards = {1: [3, 5, 11, 16, 28, 35, 39, 56], 2: [9, 10, 13, 23, 29, 39, 40, 53], 3: [2, 7, 26, 33, 36, 38, 39, 47], 4: [0, 5, 18, 20, 33, 37, 40, 41], 5: [12, 14, 31, 36, 40, 46, 49, 56], 6: [2, 6, 13, 17, 20, 24, 42, 56], 7: [3, 13, 21, 33, 34, 43, 45, 49], 8: [0, 4, 14, 17, 27, 39, 45, 50], 9: [8, 14, 24, 28, 29, 33, 48, 52], 10: [3, 17, 30, 32, 38, 40, 48, 51], 11: [17, 19, 21, 23, 25, 28, 36, 41], 12: [20, 26, 28, 30, 49, 50, 53, 54], 13: [0, 1, 12, 13, 19, 26, 35, 48], 14: [5, 10, 12, 21, 22, 24, 38, 50], 15: [2, 5, 19, 27, 29, 30, 34, 46], 16: [2, 9, 12, 15, 28, 37, 45, 51], 17: [1, 5, 9, 17, 47, 49, 52, 55], 18: [2, 4, 10, 16, 41, 44, 48, 49], 19: [11, 19, 24, 40, 44, 45, 47, 54], 20: [12, 20, 25, 32, 34, 39, 44, 52], 21: [4, 5, 8, 13, 15, 32, 36, 54], 22: [2, 8, 25, 35, 40, 43, 50, 55], 23: [6, 9, 16, 19, 31, 32, 33, 50], 24: [7, 11, 13, 41, 46, 50, 51, 52], 25: [4, 19, 37, 38, 43, 52, 53, 56], 26: [4, 20, 21, 29, 31, 35, 47, 51], 27: [1, 4, 6, 7, 22, 28, 34, 40], 28: [6, 10, 18, 30, 35, 36, 45, 52], 29: [6, 21, 37, 39, 46, 48, 54, 55], 30: [15, 17, 22, 33, 35, 44, 46, 53], 31: [13, 14, 16, 22, 25, 30, 37, 47], 32: [0, 10, 28, 32, 42, 43, 46, 47], 33: [15, 18, 23, 34, 47, 48, 50, 56], 34: [3, 7, 10, 14, 15, 19, 20, 55], 35: [8, 10, 11, 17, 26, 31, 34, 37], 36: [0, 16, 24, 34, 36, 51, 53, 55], 37: [9, 11, 20, 22, 27, 36, 43, 48], 38: [1, 2, 11, 14, 18, 21, 32, 53], 39: [5, 7, 25, 31, 42, 45, 48, 53], 40: [7, 12, 16, 17, 18, 29, 43, 54], 41: [13, 18, 27, 28, 31, 38, 44, 55], 42: [1, 15, 24, 30, 31, 39, 41, 43], 43: [0, 7, 8, 9, 21, 30, 44, 56], 44: [3, 6, 8, 12, 27, 41, 47, 53], 45: [7, 23, 24, 27, 32, 35, 37, 49], 46: [0, 2, 3, 22, 23, 31, 52, 54], 47: [5, 6, 14, 23, 26, 43, 44, 51], 48: [8, 18, 19, 22, 39, 42, 49, 51], 49: [9, 14, 34, 35, 38, 41, 42, 54], 50: [0, 6, 11, 15, 25, 29, 38, 49], 51: [22, 26, 29, 32, 41, 45, 55, 56], 52: [1, 8, 16, 20, 23, 38, 45, 46], 53: [1, 3, 29, 36, 37, 42, 44, 50], 54: [1, 10, 25, 27, 33, 51, 54, 56], 55: [3, 4, 9, 18, 24, 25, 26, 46]}

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

nn = SavedNeuralNetMLP(filename = 'symbols.sav')
font = cv.FONT_HERSHEY_SIMPLEX
for i in range(1, 56):
    print(i)
    img = cv.imread('card%s.jpg' % (i))
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
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
        if hierarchy[0][idx][3] == 0 and w > 10 and h > 10:
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
        out = getSubImage(rect, img2)
        prop = int(255*h/w) if w > h else int(255*w/h)
        dense = int(255*area/w/h)
        b,g,r,_ = np.int0(cv.mean(out))
        symbs.append([prop, dense, b, g])
        sizes.append([s, max(w,h), cnt, int(x), int(y)])
    for x in list(x[0] for x in sorted(list(sorted(sizes,key=lambda l:l[1], reverse=True)))):
        n = getSymb(symbs[x])
        if n < 57 and not n in ks:
            cv.drawContours(img, [sizes[x][2]], 0, (255, 0, 0), 3)
            cv.putText(img, names[n], (sizes[x][3]+60, sizes[x][4]-70), font, 1, (255, 0, 0), 2, cv.LINE_AA)
            ks.append(n)
        if len(ks) > 7:
            break
    if cards[i] != sorted(ks):
        print(i, cards[i], sorted(ks))
    cv.imshow("Card", img)
    cv.waitKey(0)
