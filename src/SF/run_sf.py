# model ref: https://github.com/opencv/opencv_extra/blob/master/testdata/cv/ximgproc/model.yml.gz
# code ref: https://github.com/opencv/opencv_contrib/blob/master/modules/ximgproc/samples/edgeboxes_demo.py

import cv2
import numpy as np
from ipdb import set_trace as pdb


def make_single_sf(img_path, input_dir, save_dir):
    img_name = img_path.split('.', 1)[0]
    model = 'SF/model.yml.gz'
    im_path = input_dir + img_path
    im = cv2.imread(im_path)

    edge_detection = cv2.ximgproc.createStructuredEdgeDetection(model)
    rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)
    cv2.imwrite(f'{save_dir}{img_name}_edge.jpg', np.uint8(edges * 255))