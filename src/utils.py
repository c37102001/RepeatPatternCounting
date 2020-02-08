import numpy as np
import cv2
import math
from ipdb import set_trace as pdb
from tqdm import tqdm
import itertools


def filter_contours(contours, re_height, re_width):
    MIN_PERIMETER = 60
    MAX_IMG_SIZE = 5
    MIN_IMG_SIZE = 30000
    SOLIDITY_THRE = 0.5
    MAX_EDGE_NUM = 50
    
    accept_contours = []
    for c in contours:
        perimeter = len(c)
        if perimeter < MIN_PERIMETER or perimeter > (re_height + re_width) * 2 / 3.0:
            continue
        
        contour_area = cv2.contourArea(c)
        if contour_area < (re_height * re_width) / MIN_IMG_SIZE or contour_area > (re_height * re_width) / MAX_IMG_SIZE:
            continue

        convex_area = cv2.contourArea(cv2.convexHull(c))
        solidity = contour_area / convex_area
        if solidity < SOLIDITY_THRE:
            continue

        epsilon = 0.01 * cv2.arcLength(c, closed=True)
        edge_num = len(cv2.approxPolyDP(c, epsilon, closed=True))
        if edge_num > MAX_EDGE_NUM:
            continue

        accept_contours.append(c)

    return accept_contours

def remove_overlap(contours):
    overlap_outer_cnt = []
    contours.sort(key=lambda x: len(x), reverse=False)
    for i, cnt1 in enumerate(contours[:-1]):
        for cnt2 in contours[i+1: ]:
            if is_overlap(cnt1, cnt2):
                overlap_outer_cnt.append(cnt2)
    for cnt in overlap_outer_cnt:
        contours.remove(cnt)
    return contours


def is_overlap(cnt1, cnt2):
    """
    Determine that if one contour contains another one.
    """

    c1M = get_centroid(cnt1)
    c2M = get_centroid(cnt2)
    c1D = abs(cv2.pointPolygonTest(cnt1, c1M, True))
    c2D = abs(cv2.pointPolygonTest(cnt2, c2M, True))
    c1c2D = eucl_distance(c1M, c2M)

    # check contains and similar size
    if c1c2D < min(c1D, c2D) and min(c1D, c2D) / max(c1D, c2D) > (2 / 3):
        return True
    
    return False

def check_overlap(cnt_dic_list, keep='keep_inner'):
    '''
    @param
    keep = 'keep_inner'(default) :
    Since OpenCV FindContours() will find two contours(inner and outer) of a single closed
    edge component, we choose the inner one to preserve. (The reason is that the outter one
    is easily connected with the surroundings.)

    keep = 'group_weight' :
    Since there's two edge results(Canny and SF), we should decide which contours
    to preserved if they are overlapped.
    check if overlap contours are same contour , if true makes them same label
    '''

    if cnt_dic_list == []:
        return []

    checked_list = []

    if keep == 'group_weight':

        label_list = [x['label'] for x in cnt_dic_list]
        label_change_list = []
        ''''''
        label_group_change = []
        label_change_dic = {}

        '''
        Compare each 2 contours and check if they are overlapped.
        If they are overlapped, change the label of the less group count one to the other's label whose group count is more.
        '''
        for cnt_i in range(len(cnt_dic_list) - 1):
            for cnt_k in range(cnt_i + 1, len(cnt_dic_list)):

                if cnt_dic_list[cnt_i]['group_weight'] > 0 and cnt_dic_list[cnt_k]['group_weight'] > 0:
                    if is_overlap(cnt_dic_list[cnt_i]['cnt'], cnt_dic_list[cnt_k]['cnt']):

                        if cnt_dic_list[cnt_i]['group_weight'] > cnt_dic_list[cnt_k]['group_weight']:
                            cnt_dic_list[cnt_k]['group_weight'] = 0
                            label_change_list.append((cnt_dic_list[cnt_k]['label'], cnt_dic_list[cnt_i]['label']))
                        else:
                            cnt_dic_list[cnt_i]['group_weight'] = 0
                            label_change_list.append((cnt_dic_list[cnt_i]['label'], cnt_dic_list[cnt_k]['label']))

        # check if overlap contours are same contour , if true makes them same label
        for label_change in set(label_change_list):
            '''0.5 Changeable'''
            if label_change_list.count(label_change) >= 0.5 * label_list.count(label_change[0]):
                found = False
                for label_group_i in range(len(label_group_change)):
                    if label_change[0] in label_group_change[label_group_i]:
                        found = True
                        label_group_change[label_group_i].append(label_change[1])
                    elif label_change[1] in label_group_change[label_group_i]:
                        found = True
                        label_group_change[label_group_i].append(label_change[0])

                if not found:
                    label_group_change.append([label_change[0], label_change[1]])

                # label_change_dic[label_change[0]] = label_change[1]

        for label_group in label_group_change:
            for label in label_group:
                label_change_dic[label] = label_group[0]

        for cnt_dic in cnt_dic_list:
            if cnt_dic['group_weight'] > 0:
                if cnt_dic['label'] in label_change_dic:
                    cnt_dic['label'] = label_change_dic[cnt_dic['label']]
                checked_list.append(cnt_dic)
    else:

        if keep == 'keep_inner':
            # sort list from little to large
            cnt_dic_list.sort(key=lambda x: len(x['cnt']), reverse=False)

        elif keep == 'keep_outer':
            cnt_dic_list.sort(key=lambda x: len(x['cnt']), reverse=True)

        for c_dic in cnt_dic_list:
            if is_overlap_all(c_dic, checked_list):
                continue
            checked_list.append(c_dic)

            # end keep if

    return checked_list

def is_overlap_all(cnt_dic, cnt_dic_list):
    '''
    Determine if one contour contains/contained  other contours in a input list.
    '''

    if cnt_dic == [] or len(cnt_dic_list) < 1:
        return False

    for c_dic in cnt_dic_list:
        # if len(c) == len(cnt) and GetCentroid(c) == GetCentroid(cnt):
        ##print 'same one'
        # continue
        if IsOverlap(cnt_dic['cnt'], c_dic['cnt']):
            return True

    return False

def avg_img_gradient(img, model='lab'):
    '''
    Count the average gardient of the whole image, in order to compare with
    the color gradient obviousity.
    There are two uses of the 'avg_gradient'.
    1. Avoid that the image are only two color gradients, one of them will be deleted , even if they are close.
    2. If all the color gradient are less than the avg_gradient, all of them will be discarded
       since they are not obvious enough.
    '''

    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])

    if model == 'lab':

        height, width = img.shape[:2]
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_l = lab[:, :, 0]
        lab_a = lab[:, :, 1]
        lab_b = lab[:, :, 2]

        lab_list = [lab_l, lab_a, lab_b]
        gradient_list = []

        for lab_channel in lab_list:
            gradient = cv2.filter2D(lab_channel, -1, kernel)
            gradient_list.append(gradient)

        avg_gradient = 0.0
        for x in range(height):
            for y in range(width):
                avg_gradient += math.sqrt(
                    pow(gradient_list[0][x, y], 2) + pow(gradient_list[1][x, y], 2) + pow(gradient_list[2][x, y], 2))

        avg_gradient /= (float(height) * float(width))

    return avg_gradient

def get_centroid(cnt):
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return cx, cy

def eucl_distance(a, b):
    if type(a) != np.ndarray:
        a = np.array(a)
    if type(b) != np.ndarray:
        b = np.array(b)

    return np.linalg.norm(a - b)

def evaluate_detection_performance(img, fileName, final_group_cnt, resize_ratio, evaluate_csv_path):
    '''
    Evaluation during run time.
    The evaluation is about if the contours are
    detected correctly.
    The results are compared with the groundtruth.

    @param
    evaluate_csv_path : read the groundtruth data
    '''

    tp = 0
    fp = 0
    fn = 0
    pr = 0.0
    re = 0.0
    # Mix the pr and re
    fm = 0.0
    # Only compare the count
    er = 0.0
    groundtruth_list = []
    translate_list = [['Group', 'Y', 'X']]
    with open(evaluate_csv_path + fileName + '.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # groundtruth_list.append( { 'Group':int(row['Group']), 'X':int(int(row['X'])*resize_ratio), 'Y':int(int(row['Y'])*resize_ratio) } )
            groundtruth_list.append({'Group': int(row['Group']), 'X': int(row['X']), 'Y': int(row['Y'])})

    cnt_area_coordinate = Get_Cnt_Area_Coordinate(img, final_group_cnt)
    cnt_area_coordinate.sort(key=lambda x: len(x), reverse=False)

    groundtruth_count = len(groundtruth_list)
    program_count = len(cnt_area_coordinate)

    # _________The 1st Evaluation and the preprocessing of the 2nd evaluation_____________________________
    '''
    @param
    g_dic : the coordinate of one contour in the groundtruth list (g means groundtruth)
    cnt_dic : one contour(all pixels' coordinate in a contour area) in the cnt_area_coordinate
    cnt_area_coordinate : All contours that the program found in one image 

    If g_dic is in cnt_dic (which means one of the groundtruth contours matches one of the contours that the program found),
    save both label of cnt_dic and the coordinate of g_dic in the translate list.
    '''
    for g_dic in groundtruth_list:
        for cnt_dic in cnt_area_coordinate:
            if [int(g_dic['Y'] * resize_ratio), int(g_dic['X'] * resize_ratio)] in cnt_dic['coordinate']:
                tp += 1
                cnt_area_coordinate.remove(cnt_dic)
                translate_list.append([cnt_dic['label'], g_dic['Y'], g_dic['X']])
                break

    '''Make a csv that save the translate list.'''
    f = open(csv_output + fileName[:-4] + '.csv', "wb")
    w = csv.writer(f)
    w.writerows(translate_list)
    f.close()

    fp = program_count - tp
    fn = groundtruth_count - tp

    if tp + fp > 0:
        pr = tp / float(tp + fp)
    if tp + fn > 0:
        re = tp / float(tp + fn)
    if pr + re > 0:
        fm = 2 * pr * re / (pr + re)
    if groundtruth_count > 0:
        er = abs(program_count - groundtruth_count) / float(groundtruth_count)
    print(program_count, groundtruth_count)
    return tp, fp, fn, pr, re, fm, er
    # _____________________1 st evaluation end__________________________________________________

def Get_Cnt_Area_Coordinate(img, final_group_cnt):
    '''
    Take the contour list (in order) as input ,
    output all the points within the contour.
    In order to check if a point is contained in the contour.
    '''

    cnt_area_coordinate = []
    blank_img = np.zeros(img.shape[:2], np.uint8)

    for cnt_group in final_group_cnt:
        for cnt in cnt_group:
            blank_img[:] = 0
            cv2.drawContours(blank_img, [cnt], -1, 255, -1)
            # cv2.imshow('blank',blank_img)
            # cv2.waitKey(0)
            # use argwhere to find all coordinate which value == 1 ( cnt area )
            cnt_area_coordinate.append((np.argwhere(blank_img == 255)).tolist())

    return cnt_area_coordinate


# def is_overlap_by_charlie(cnt1, cnt2):
    
#     # perimeter:
#     if len(cnt1)/ len(cnt2) < 0.8:
#         return False
    
#     # area
#     c1A = cv2.contourArea(cnt1)
#     c2A = cv2.contourArea(cnt2)
#     if c1A / c2A < 0.75:
#         return False
    
#     # pdb()
#     c1M = get_centroid(cnt1)
#     c1D = abs(cv2.pointPolygonTest(cnt1, c1M, True))
#     c2M = get_centroid(cnt2)
#     c2D = abs(cv2.pointPolygonTest(cnt2, c2M, True))
#     c1c2D = eucl_distance(c1M, c2M)
#     if max(c1D, c2D) == 0:
#         pdb()
#     if c1c2D > min(c1D, c2D) or min(c1D, c2D) / max(c1D, c2D) < (2 / 3):
#         return False

#     # if intersect = not overlap
#     cnt1 = cnt1.reshape(len(cnt1), 2)
#     cnt2 = cnt2.reshape(len(cnt2), 2)
#     sets_cnt1 = set(map(lambda x: frozenset(tuple(x)), cnt1))
#     sets_cnt2 = set(map(lambda x: frozenset(tuple(x)), cnt2))
#     if len(sets_cnt1.intersection(sets_cnt2)) > 0:
#         return False

#     return True


