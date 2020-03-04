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

    if c1c2D > (len(cnt1) + len(cnt2)) / 4:
        return False

    # check contains and similar size
    if c1c2D < min(c1D, c2D) and min(c1D, c2D) / max(c1D, c2D) > (2 / 3):
        return True

    small_cnt, large_cnt = sorted((cnt1, cnt2), key=len)
    points_side = [cv2.pointPolygonTest(large_cnt, tuple(point[0]), False) for point in small_cnt]
    if points_side.count(-1) < len(small_cnt) * 0.7:        # if outside points less than 0.7
        return True
    
    return False


def remove_group_overlap(cnt_dicts, labels, drawer, do_draw):
    label_area = {}
    for label in set(labels):
        area = [cv2.contourArea(cnt_dict['cnt']) for cnt_dict in cnt_dicts if cnt_dict['label']==label]
        area = sum(area) / len(area)
        label_area[label] = area
    
    # If 2 contours are overlapped, change the label of the less group to another label.
    for i, dict_i in tqdm(enumerate(cnt_dicts[:-1]), total=len(cnt_dicts[:-1]), desc='[Remove overlap]'):
        for dict_j in cnt_dicts[i+1:]:
            if is_overlap(dict_i['cnt'], dict_j['cnt']):
                
                # if same label, keep larger one
                if dict_i['label'] == dict_j['label']:
                    mean_area = label_area[dict_i['label']]
                    if abs(cv2.contourArea(dict_i['cnt'])-mean_area) < abs(cv2.contourArea(dict_j['cnt'])-mean_area):
                        dict_j['group_weight'] = 0
                    else:
                        dict_i['group_weight'] = 0
                
                # if different label and areas are similar, keep larger weight one.
                # if areas are quite different, keep them both
                elif 0.2 <= (cv2.contourArea(dict_i['cnt']) / cv2.contourArea(dict_j['cnt'])) <= 5:
                    grads_i = [d['color_gradient'] for d in cnt_dicts if d['label']==dict_i['label']]
                    mean_grad_i = sum(grads_i) / len(grads_i)
                    grads_j = [d['color_gradient'] for d in cnt_dicts if d['label']==dict_j['label']]
                    mean_grad_j = sum(grads_j) / len(grads_j)
                    if mean_grad_i > mean_grad_j:
                        dict_j['group_weight'] = 0
                    else:
                        dict_i['group_weight'] = 0
                    # if dict_i['group_weight'] > dict_j['group_weight']:
                        # dict_j['group_weight'] = 0
                    # else:
                        # dict_i['group_weight'] = 0
    # draw
    cnt_dicts = [cnt_dict for cnt_dict in cnt_dicts if cnt_dict['group_weight'] > 0]
    labels = [cnt_dict['label'] for cnt_dict in cnt_dicts]
    print('[Remove overlap results] (label, counts): ', [(label, labels.count(label)) for label in set(labels)])
    if do_draw or True:
        img = drawer.blank_img()
        for label in set(labels):
            cnts = [cnt_dict['cnt'] for cnt_dict in cnt_dicts if cnt_dict['label']==label]
            img = drawer.draw_same_color(cnts, img)
        drawer.save(img, '2-3_RemoveOverlap')
    
    return cnt_dicts, labels


def scale_contour(cnt, type, im_area):
    area = cv2.contourArea(cnt)
    if area / im_area >= 0.01:
        scale = 0.95 if type == 'inner' else 1.05
    elif area / im_area >= 0.001:
        scale = 0.9 if type == 'inner' else 1.1
    else:
        scale = 0.85 if type == 'inner' else 1.17
    
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled






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
