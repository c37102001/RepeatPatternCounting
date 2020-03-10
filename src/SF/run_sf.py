import cv2
import numpy as np
from ipdb import set_trace as pdb

# model ref: https://github.com/opencv/opencv_extra/blob/master/testdata/cv/ximgproc/model.yml.gz
# code ref: https://github.com/opencv/opencv_contrib/blob/master/modules/ximgproc/samples/edgeboxes_demo.py


def make_single_sf(img, edge_img_path):
    model = 'SF/model.yml.gz'
    edge_detection = cv2.ximgproc.createStructuredEdgeDetection(model)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    edge_img = edge_detection.detectEdges(np.float32(img) / 255.0)
    edge_img = np.uint8(edge_img * 255)
    cv2.imwrite(edge_img_path, edge_img)