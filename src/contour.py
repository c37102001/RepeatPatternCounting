import cv2
from utils import is_overlap


def find_contour(edged):
    print('[Contour] Find contour.')

    edged = cv2.threshold(edged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(edged, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[-2]
    contours.sort(key=lambda x: len(x), reverse=False)

    return contours


def check_property(contours, re_height, re_width):
    del_idx_list = []
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        convex_area = cv2.contourArea(cv2.convexHull(c))
        approx = cv2.approxPolyDP(c, 10, True)

        ''' new
        bad_solidity = convex_area == 0 or area / convex_area < 0.3
        small_area = area < 10 or len(c) < 60
        # big_area = len(c) > (re_height + re_width) * 2 / 3.0
        # too_many_edge = len(approx) > 50
        
        if bad_solidity or small_area:
            del_idx_list.append(i)
        '''

        small_and_big = len(c) < 60 or len(c) > (re_height + re_width) * 2 / 3.0
        high_density = area < 4 or area / convex_area < 0.5
        too_many_edge = len(approx) > 50

        if small_and_big or high_density or too_many_edge:
            del_idx_list.append(i)

    while len(del_idx_list) > 0:
        del_idx = del_idx_list.pop()
        del contours[del_idx]
    return contours


def check_simple_overlap(contours):
    tmp_cnt_list = [contours[0]]
    # the first contour
    tmp_cnt = contours[0]
    # Since if is overlap , the order of the overlapped contours will be continuous
    # The goal of the following for-loop is equal to CheckOverlap(.., 'keep inner') and much more easier.
    for c in contours[1:]:
        if not is_overlap(tmp_cnt, c):
            tmp_cnt_list.append(c)
        tmp_cnt = c

    contours = tmp_cnt_list
    return contours
