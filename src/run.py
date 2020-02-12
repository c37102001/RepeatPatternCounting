import numpy as np
import cv2
import os
import math
import time
import csv
import argparse
import matplotlib.pyplot as plt
from ipdb import set_trace as pdb
from tqdm import tqdm

from canny import canny_edge_detect
from contour_drawer import ContourDrawer
from utils import get_edge_group
from misc import check_overlap, count_avg_gradient, evaluate_detection_performance


resize_height = 736.0       # 736 will make the colony perfomance the best. (ref to yun-tao colony)
_enhance_edge = True
_evaluate = False   # Decide if excecute 1st evalution

input_dir = '../input/image/'
strct_edge_dir = '../input/edge_image/'  # structure forest output
hed_edge_dir = '../input/hed_edge_image/'  # hed edge
csv_output = '../output_csv_6_8[combine_result_before_filter_obvious]/'
evaluate_csv_path = '../evaluate_data/groundtruth_csv/generalize_csv/'
IMG_LIST = ['IMG_ (39).jpg', 'IMG_ (10).jpg', 'IMG_ (16).jpg' ]
TEST = True

# ================ CHANGABLE ===============
DRAW_PROCESS = True
TEST_IMG = 'IMG_ (39).jpg'
KEEP_OVERLAP = ['inner']     # 'inner', 'outer', 'all'
DO_COMBINE = False

output_dir = '../output/exp/'
_use_canny_edge = False
_use_structure_edge = True
_use_hed_edge = True
# ==========================================


max_time_img = ''
min_time_img = ''
min_time = math.inf
max_time = -math.inf
evaluation_csv = [['Image name', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F_measure', 'Error_rate']]


# TODO IMG_LIST, No contour assert
# for i, img_name in enumerate(tqdm(os.listdir(input_dir))):
for i, img_path in enumerate(tqdm(IMG_LIST)):
    start = time.time()

    if TEST:
        img_path = TEST_IMG
        if i > 0:
            break
    
    #============================ Preprocess =================================================
    # check format
    img_name, img_ext = img_path.rsplit('.', 1)     # 'IMG_ (33),  jpg
    if img_ext not in ['jpg', 'png', 'jpeg']:
        print(f'[Error] Format not supported: {img_path}')
        continue

    print('[Input] %s' % img_path)
    input_img = cv2.imread(input_dir + img_path)
    img_height = input_img.shape[0]               # shape: (1365, 2048, 3)
    
    # resize_height=736, shape: (1365, 2048, 3) -> (736,1104,3)
    resize_factor = resize_height / img_height
    resi_input_img = cv2.resize(input_img, (0, 0), fx=resize_factor, fy=resize_factor)
    drawer = ContourDrawer(resi_input_img, output_dir, img_name)
    if DRAW_PROCESS:
        cv2.imwrite(output_dir + img_name + '_a_original_image.jpg', resi_input_img)


    #=================== Get grouped contours of every edge image ============================
    group_cnt_dict_list = []    # combine edge detection result
    if _use_canny_edge:
        canny_edge = canny_edge_detect(img)
        canny_group = get_edge_group(drawer, canny_edge, edge_type='Canny', do_enhance=False, do_draw=DRAW_PROCESS)
        for edge_group in canny_group:
            group_cnt_dict_list.append(edge_group)
    
    edge_imgs = []
    if _use_structure_edge:
        edge_path = strct_edge_dir + img_name + '_edge.jpg'
        edge_type = 'Structure'
        edge_imgs.append((edge_path, edge_type))
    if _use_hed_edge:
        edge_path = hed_edge_dir + img_name + '_hed.png'
        edge_type = 'HED'
        edge_imgs.append((edge_path, edge_type))
    
    # detect contours, filter, extract features, cluster for edge image
    for edge_path, edge_type in edge_imgs:
        if not os.path.isfile(edge_path):
            print(f'[Error] EDGE FILE {edge_path} does not exist!')
            continue
        edge_img = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
        edge_img = cv2.resize(edge_img, (0, 0), fx=resize_factor, fy=resize_factor)     # shape: (736, *)

        for keep in KEEP_OVERLAP:
            edge_group = get_edge_group(drawer, edge_img, edge_type, keep=keep, do_draw=DRAW_PROCESS)
            for g in edge_group:
                group_cnt_dict_list.append(g)

    #================================ Combine ====================================

    if DO_COMBINE:
        strct_edge_path = strct_edge_dir + img_name + '_edge.jpg'
        hed_edge_path = hed_edge_dir + img_name + '_hed.png'

        if os.path.isfile(strct_edge_path) and os.path.isfile(hed_edge_path):
            strct_edge = cv2.imread(strct_edge_path, cv2.IMREAD_GRAYSCALE)
            hed_edge = cv2.imread(hed_edge_path, cv2.IMREAD_GRAYSCALE)
            edge = (strct_edge + hed_edge)
            edge = cv2.resize(edge, (0, 0), fx=resize_height / img_height, fy=resize_height / img_height)

            for keep in KEEP_OVERLAP:
                edge_group = get_edge_group(drawer, edge, 'Combine', keep=keep, do_draw=DRAW_PROCESS)
            for g in edge_group:
                group_cnt_dict_list.append(g)
        else:
            print('[Error] Lack of edge images for combine')

    #=============================================================================

    '''
    group_cnt_dict_list: (list of list of dict), sized [#total groups, #cnts in group (every cnt is a dict)]
    group_cnt_dict_list[0][0] = {
        'cnt': contours[i],
        'shape': cnt_pixel_distances[i],
        'color': cnt_avg_lab[i],
        'size': cnt_norm_size[i],
        'color_gradient': cnt_color_gradient[i]

        # keys added below
        'label': i,
        'group_weight': len(group_cnt_dict)
    }
    '''
    
    # check two contour overlap
    # TODO group_cnt_dict_list and cnt_dict_list is redundant
    # add label and group weight(num of cnts in the group) to contour dictionary
    for i, group_cnt_dict in enumerate(group_cnt_dict_list):
        for cnt_dict in group_cnt_dict:
            cnt_dict['label'] = i
            cnt_dict['group_weight'] = len(group_cnt_dict)

    # flatten to a list of contour dict
    cnt_dict_list = [cnt_dict for group_cnt_dict in group_cnt_dict_list for cnt_dict in group_cnt_dict]
    # show original labels and counts
    cnt_labels = [cnt_dict['label'] for cnt_dict in cnt_dict_list]
    print('(label, counts): ', [(label, cnt_labels.count(label)) for label in set(cnt_labels)])

    # check overlapped cnts and change their labels or remove them
    cnt_dict_list = check_overlap(cnt_dict_list)
    # show labels and counts after removed overlapped cnts
    cnt_labels = [cnt_dict['label'] for cnt_dict in cnt_dict_list]
    print('(label, counts): ', [(label, cnt_labels.count(label)) for label in set(cnt_labels)])

    if DRAW_PROCESS:
        img = drawer.blank_img()
        for label in set(cnt_labels):
            cnts = [cnt_dict['cnt'] for cnt_dict in cnt_dict_list if cnt_dict['label'] == label]
            img = drawer.draw_same_color(cnts, img)
        drawer.save(img, 'g_RemoveOverlapCombineCnt')

    final_group = []
    for label in set(cnt_labels):
        group_cnt_dict_list = [cnt_dict for cnt_dict in cnt_dict_list if cnt_dict['label'] == label]

        if len(group_cnt_dict_list) < 1:
            continue

        group_cnt = []
        avg_color_gradient = 0.0
        avg_shape_factor = 0.0
        total_area = 0.0

        # for each final group count obvious factor
        for cnt_dict in group_cnt_dict_list:
            cnt = cnt_dict['cnt']
            cnt_area = cv2.contourArea(cnt)
            convex_area = cv2.contourArea(cv2.convexHull(cnt))

            total_area += cnt_area
            avg_shape_factor += cnt_area / convex_area
            avg_color_gradient += cnt_dict['color_gradient']
            group_cnt.append(cnt)

        avg_shape_factor /= float(len(group_cnt_dict_list))
        avg_color_gradient /= float(len(group_cnt_dict_list))

        final_group.append({
            'group_cnt': group_cnt,
            'area': total_area,
            'color_gradient': avg_color_gradient, 
            'shape': avg_shape_factor,
            'votes': 0,
        })

    # obviousity filter
    obvious_list = ['area', 'shape', 'color_gradient']
    for obvious_para in obvious_list:
        final_group.sort(key=lambda x: x[obvious_para], reverse=False)
        para_list = [group[obvious_para] for group in final_group]
        
        # find obvious index
        diff = np.diff(para_list)
        obvious_index = np.where(diff == max(diff))[0][0] + 1

        # check cover_area
        if obvious_para == 'area':
            obvious_area = para_list[obvious_index]
            for i in range(obvious_index-1, -1, -1):
                area = para_list[i]
                # include cnt with cover area close to obvious cover area
                if area * 2 > obvious_area:
                    obvious_area = area
                    obvious_index = i
        
        # check shape factor
        elif obvious_para == 'shape':
            obvious_shape_factor = para_list[obvious_index]
            for i, shape_factor in enumerate(para_list[:obvious_index]):
                # include cnt with shape factor close to obvious shape factor
                if shape_factor > 0.8 * obvious_shape_factor:
                    obvious_index = i
                    break
        
        # check color_gradient
        elif obvious_para == 'color_gradient':
            # count whole image avg color gradient
            avg_gradient = count_avg_gradient(resi_input_img)
            # exclude obvious cnt with color gradient less than avg_gradient
            for i in range(obvious_index, len(para_list)):
                if para_list[i] < avg_gradient:
                    obvious_index += 1

            # skip if all color gradients are less tha avg_gradient
            if obvious_index == len(para_list):
                print('No color_gradient result')
                continue
        
        for group in final_group[obvious_index:]:
            group['votes'] += 1

        if DRAW_PROCESS:
            img = drawer.blank_img()
            for group in final_group[obvious_index:]:
                img = drawer.draw_same_color(group['group_cnt'], img, color=(0, 255, 0))  # green for obvious
            for group in final_group[:obvious_index]:
                img = drawer.draw_same_color(group['group_cnt'], img, color=(0, 0, 255))  # red for others
            drawer.save(img, desc=f'h_Obvious{obvious_para.capitalize()}')

            plt.bar(x=range(len(para_list)), height=para_list)
            plt.title(f'{obvious_para} cut idx: {obvious_index} | value: {para_list[obvious_index]}')
            plt.savefig(f'{output_dir}{img_name}_h_Obvious{obvious_para.capitalize()}_hist.png')
            plt.close()
        

    obvious_group = []
    final_group.sort(key=lambda x: x['votes'], reverse=True)
    max_weight = final_group[0]['votes']

    for group in final_group:
        # TODO can further specify accept 2 when the loss weight is from color
        if group['votes'] >= min(2, max_weight):
            obvious_group.append(group)

    # contours with same label
    group_cnt = [group['group_cnt'] for group in obvious_group]
    
    # draw final result
    img = resi_input_img / 3.0    # darken the image to make the contour visible
    for cnts in group_cnt:
        img = drawer.draw_same_color(cnts, img)
    img = cv2.resize(img, (0, 0), fx=1/resize_factor, fy=1/resize_factor)
    img = np.concatenate((input_img, img), axis=1)
    drawer.save(img, 'l_FinalResult')

    spent_time = time.time() - start
    print(f'Finished in {spent_time} s')

    print('-----------------------------------------------------------')
    if spent_time > max_time:
        max_time = spent_time
        max_time_img = img_name
    if spent_time < min_time:
        min_time = spent_time
        min_time_img = img_name

    if _evaluate:
        resize_ratio = resize_height / float(img_height)
        tp, fp, fn, pr, re, fm, er = evaluate_detection_performance(img, img_name, group_cnt,
                                                                    resize_ratio, evaluate_csv_path)
        evaluation_csv.append([img_name, tp, fp, fn, pr, re, fm, er])


if _evaluate:
    f = open(evaluate_csv_path + 'evaluate-bean.csv', "wb")
    w = csv.writer(f)
    w.writerows(evaluation_csv)
    f.close()

print('img:', max_time_img, ' max_time:', max_time, 's')
print('img:', min_time_img, 'min_time:', min_time, 's')

