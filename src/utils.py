import numpy as np
import cv2
import math
from ipdb import set_trace as pdb
import itertools
from tqdm import tqdm


def add_border_edge(edge_img):
    edge_img[0] = 255       # first row
    edge_img[-1] = 255      # last row
    edge_img[:, 0] = 255    # first column
    edge_img[:, -1] = 255   # last column
    return edge_img


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


def is_overlap(cnt1, cnt2):
    c1M = get_centroid(cnt1)
    c2M = get_centroid(cnt2)
    c1D = abs(cv2.pointPolygonTest(cnt1, c1M, True))
    c2D = abs(cv2.pointPolygonTest(cnt2, c2M, True))
    c1c2D = eucl_distance(c1M, c2M)

    # check contains and similar size
    if c1c2D < min(c1D, c2D) and min(c1D, c2D) / max(c1D, c2D) > (2 / 3):
        return True

    cnt1tocnt2 = [cv2.pointPolygonTest(cnt2, tuple(point[0]), False) for point in cnt1]
    if cnt1tocnt2.count(1) > 0:
        return True
    cnt2tocnt1 = [cv2.pointPolygonTest(cnt1, tuple(point[0]), False) for point in cnt2]
    if cnt2tocnt1.count(1) > 0:
        return True
    
    return False

def remove_overlap(contours):
    # sort from min to max
    contours.sort(key=lambda x: cv2.contourArea(x), reverse=False)
    overlap_idx = []

    for i, cnt1 in tqdm(enumerate(contours[:-1]), total=len(contours[:-1]), desc='[Remove overlap]'):
        for j, cnt2 in enumerate(contours[i+1: ], start=i+1):
            if is_overlap(cnt1, cnt2):
                overlap_idx.append(j)
    
    overlap_idx = list(set(overlap_idx))
    keep_idx = [i for i in range(len(contours)) if i not in overlap_idx]
    keep_contours = [contours[idx] for idx in keep_idx]
    
    return keep_contours


def remove_outliers(contours, m=3):
    outlier_idx = []

    sizes = np.array([cv2.contourArea(c) for c in contours])
    mean = np.mean(sizes)
    std = np.std(sizes)
    
    outlier_idx = np.where((abs(sizes - mean)/ std) > m)[0].tolist()
    keep_idx = [i for i in range(len(contours)) if i not in outlier_idx]
    keep_contours = [contours[idx] for idx in keep_idx]

    return keep_contours

def count_avg_gradient(img, model='lab'):
    # Count the average gardient of the whole image

    height, width = img.shape[:2]
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])

    if model == 'lab':
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)    # different if not convert from uint8
        lab_l = lab[:, :, 0]        # sized [736, *]
        lab_a = lab[:, :, 1]
        lab_b = lab[:, :, 2]

        gradient_list = []
        for lab_channel in [lab_l, lab_a, lab_b]:
            gradient = cv2.filter2D(lab_channel, -1, kernel)    # sized [736, *]
            gradient_list.append(gradient)                      # sized [3, 736, *]
        
        gradient_list = [g**2 for g in gradient_list]            # sized [3, 736, *]
        gradient_list = sum(gradient_list)                      # sized [736, *]
        gradient_list = np.sqrt(gradient_list)                  # sized [736, *]
        avg_gradient = np.mean(gradient_list)                   # float, e.g. 30.926353
        
    return avg_gradient


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
