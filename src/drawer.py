import numpy as np
import cv2
from ipdb import set_trace as pdb
from itertools import cycle


class ContourDrawer:
    def __init__(self, color_img, output_path, img_name):
        self.color_img = color_img
        self.output_path = output_path
        self.img_name = img_name
        self.switchColor = cycle(
            [(0, 255, 255), (128, 255, 255), (0, 0, 255), (0, 255, 0),
             (0, 128, 0), (128, 128, 0), (255, 128, 0), (128, 255, 0)]
        )

    def blank_img(self):
        return np.zeros(self.color_img.shape, np.uint8)
    
    def draw(self, contours, given_img=None):
        if given_img is None:
            given_img = self.blank_img()
        for c in contours:
            cv2.drawContours(given_img, [c], -1, next(self.switchColor), 2)
        return given_img
    
    def draw_same_color(self, contours, given_img=None, color=None):
        if given_img is None:
            given_img = self.blank_img()
        if color is None:
            color = next(self.switchColor)
        given_img = cv2.drawContours(given_img, contours, -1, color, 2)
        return given_img

    

    def save(self, img, desc):
        img_path = '{}{}_{}.jpg'.format(self.output_path, self.img_name, desc)
        cv2.imwrite(img_path, img)