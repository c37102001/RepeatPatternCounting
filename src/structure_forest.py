# ref:
#     code: https://github.com/opencv/opencv_contrib/blob/master/modules/ximgproc/samples/edgeboxes_demo.py
#     model: https://github.com/opencv/opencv_extra/blob/master/testdata/cv/ximgproc/model.yml.gz
#     other: https://answers.opencv.org/question/209660/python-opencv-structured-forests-edge-detection-typeerror/

import cv2
import numpy as np
import ipdb

if __name__ == '__main__':

    fileName = 'IMG_ (61)'
    # fileName = 'brick (1)'

    model = '../model.yml'
    im = cv2.imread('../input/image/%s.jpg' % fileName)

    edge_detection = cv2.ximgproc.createStructuredEdgeDetection(model)
    rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)

    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)
    edges = edges*1000

    # edges = cv2.threshold(edges.astype(np.uint8), 0, 255, cv2.THRESH_OTSU)[1]
    cv2.imwrite('../input/edge_image/%s_edge.jpg' % fileName, edges)

    print('Structure forest done.')
