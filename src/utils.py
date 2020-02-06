import numpy as np
import cv2

switchColor = [(255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 128, 0),
               (255, 0, 128), (128, 0, 255), (128, 255, 0), (0, 128, 255), (0, 255, 128), (128, 128, 0), (128, 0, 128),
               (0, 128, 128), (255, 64, 0), (255, 0, 64), (64, 255, 0), (64, 0, 255), (0, 255, 64), (0, 64, 255)]


def is_overlap(cnt1, cnt2):
    """
    Determine that if one contour contains another one.
    """

    if cnt1 == [] or cnt2 == []:
        return False

    c1M = get_centroid(cnt1)
    c2M = get_centroid(cnt2)
    c1_min_d = min_distance(cnt1)
    c2_min_d = min_distance(cnt2)
    moment_d = eucl_distance(c1M, c2M)

    if min(c1_min_d, c2_min_d) == 0:
        return False

    # TODO why or? why ratio = 3?
    return (moment_d < c1_min_d or moment_d < c2_min_d) and max(c1_min_d, c2_min_d) / min(c1_min_d, c2_min_d) <= 3
    # return (moment_d < c1_min_d and moment_d < c2_min_d) and min(c1_min_d, c2_min_d) / max(c1_min_d, c2_min_d) > 0.6


def get_centroid(cnt):
    """
    Calculate the average coordinate as centroid.
    """
    if len(cnt) == 1:
        return cnt
    elif len(cnt) == 2:
        return (cnt[0] + cnt[1]) / 2

    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return cx, cy


def min_distance(cnt):
    '''
    Calculate the minimum distance between centroid to the contour.
    '''

    cM = get_centroid(cnt)
    if len(cnt[0][0]) == 1:
        cnt = cnt[0]
    min_d = eucl_distance((cnt[0][0][0], cnt[0][0][1]), cM)
    for c in cnt:
        d = eucl_distance((c[0][0], c[0][1]), cM)
        if d < min_d:
            min_d = d

    return min_d


def eucl_distance(a, b):
    if type(a) != np.ndarray:
        a = np.array(a)
    if type(b) != np.ndarray:
        b = np.array(b)

    return np.linalg.norm(a - b)
