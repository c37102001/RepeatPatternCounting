import numpy as np
import cv2
from ipdb import set_trace as pdb
from itertools import cycle
from utils import get_centroid


class ContourDrawer:
    def __init__(self, color_img, output_path, img_name, do_mark=False):
        self.color_img = color_img
        self.output_path = output_path
        self.img_name = img_name
        self.switchColor = cycle(
            [(0, 255, 255), (128, 255, 255), (0, 0, 255), (0, 255, 0),
             (0, 128, 0), (128, 128, 0), (255, 128, 0), (128, 255, 0)]
        )
        self.do_mark = do_mark

    def blank_img(self):
        return np.zeros(self.color_img.shape, np.uint8)
    
    def draw(self, contours, img=None):
        if img is None:
            img = self.blank_img()
        for index, c in enumerate(contours):
            cv2.drawContours(img, [c], -1, next(self.switchColor), 2)
            if self.do_mark:
                img = self._do_mark(contours, img)
        return img
    
    def draw_same_color(self, contours, img=None, color=None):
        if img is None:
            img = self.blank_img()
        if color is None:
            color = next(self.switchColor)
        img = cv2.drawContours(img, contours, -1, color, 2)
        if self.do_mark:
            img = self._do_mark(contours, img)
        return img

    def _do_mark(self, contours, img):
        for index, c in enumerate(contours):
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.01 * peri, True)
            (x, y, w, h) = cv2.boundingRect(approx)
            img = cv2.putText(img, str(index), (int(x+(w/2)), int(y+(h/2))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (4, 4, 253), 2)
        return img
    
    def save(self, img, desc):
        img_path = '{}{}_{}.jpg'.format(self.output_path, self.img_name, desc)
        cv2.imwrite(img_path, img)