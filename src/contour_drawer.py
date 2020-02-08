import numpy as np
import cv2
from ipdb import set_trace as pdb


class ContourDrawer:
    def __init__(self, image, output_path, img_name):
        self.image = image
        self.output_path = output_path
        self.img_name = img_name
        self.switchColor = \
            [(128, 255, 0), (255, 128, 255), (0, 255, 255), (255, 192, 128),
             (128, 192, 64), (128, 255, 160), (128, 128, 128), (128, 0, 128)]
        self.reset()

    def reset(self):
        self.color_index = 0
        self.canvas = np.zeros(self.image.shape, np.uint8)
    
    def draw(self, contour, given_img=None):
        if given_img is not None:   # draw with given contour img
            self.canvas = given_img

        color = self.switchColor[self.color_index % len(self.switchColor)]
        cv2.drawContours(self.canvas, contour, -1, color, 2)
        self.color_index += 1

        if given_img is not None:
            return self.canvas

    def save(self, desc):
        img_path = '{}{}_{}.jpg'.format(self.output_path, self.img_name, desc)
        cv2.imwrite(img_path, self.contour_image)