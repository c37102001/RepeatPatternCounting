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
from misc import check_overlap, avg_img_gradient, evaluate_detection_performance


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


t_start_time = time.time()
max_time_img = ''
min_time_img = ''
min_time = math.inf
max_time = -math.inf
evaluation_csv = [['Image name', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F_measure', 'Error_rate']]


# TODO IMG_LIST, No contour assert
# for i, img_name in enumerate(tqdm(os.listdir(input_dir))):
for i, img_path in enumerate(tqdm(IMG_LIST)):
    start_time = time.time()

    if TEST:
        img_path = TEST_IMG
        if i > 0:
            break

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

    final_differ_edge_group = []    # combine edge detection result
    if _use_canny_edge:
        canny_edge = canny_edge_detect(img)
        canny_group = get_edge_group(drawer, canny_edge, edge_type='Canny', do_enhance=False, do_draw=DRAW_PROCESS)
        for edge_group in canny_group:
            final_differ_edge_group.append(edge_group)
    
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
                final_differ_edge_group.append(g)

    #=============================================================================

    # combine
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
                final_differ_edge_group.append(g)
        else:
            print('[Error] Lack of edge images for combine')

    #=============================================================================

    # check two edge contour overlap
    compare_overlap_queue = []
    total_group_number = len(final_differ_edge_group)

    for group_index in range(total_group_number):
        cnt_group = final_differ_edge_group[group_index]['group_dic']

        for cnt_dic in cnt_group:
            compare_overlap_queue.append(
                {'cnt': cnt_dic['cnt'], 'label': group_index, 'group_weight': len(cnt_group), 'cnt_dic': cnt_dic})

    _label = [x['label'] for x in compare_overlap_queue]
    print('label_dic:', [(y, _label.count(y)) for y in set(_label)])

    compare_overlap_queue = check_overlap(compare_overlap_queue, keep='group_weight')

    
    # drawer.reset()
    img = drawer.blank_img()
    _label = [x['label'] for x in compare_overlap_queue]
    print('label_dic:', [(y, _label.count(y)) for y in set(_label)])

    final_group = []

    for label_i in range(total_group_number):
        tmp_group = []
        for i in range(len(compare_overlap_queue)):
            if compare_overlap_queue[i]['label'] == label_i:
                tmp_group.append(compare_overlap_queue[i]['cnt_dic'])

        if len(tmp_group) < 1:
            continue

        tmp_cnt_group = []
        avg_color_gradient = 0.0
        avg_shape_factor = 0.0
        cnt_area = 0.0

        # for each final group count obvious factor
        for cnt_dic in tmp_group:
            cnt = cnt_dic['cnt']
            tmp_area = cv2.contourArea(cnt)
            cnt_area += tmp_area
            avg_shape_factor += tmp_area / float(cv2.contourArea(cv2.convexHull(cnt)))
            avg_color_gradient += cnt_dic['color_gradient']
            tmp_cnt_group.append(cnt)

        avg_shape_factor /= float(len(tmp_group))
        avg_color_gradient /= float(len(tmp_group))
        avg_area = cnt_area / float(len(tmp_group))

        if len(tmp_cnt_group) < 2:
            continue
        # drawer.draw(np.array(tmp_cnt_group))
        img = drawer.draw_one_color(np.array(tmp_cnt_group), img)

        final_group.append({'cnt': tmp_cnt_group, 'avg_area': avg_area, 'cover_area': cnt_area,
                            'color_gradient': avg_color_gradient, 'shape_factor': avg_shape_factor,
                            'obvious_weight': 0, 'group_dic': tmp_group})

    if DRAW_PROCESS:
        desc = 'g_RemoveOverlapCombineCnt'
        # drawer.save(desc)
        drawer.save(img, desc)

    # line 637 - line 712 obviousity filter
    # contour_image = drawer.canvas
    contour_image = img
    obvious_list = ['cover_area', 'color_gradient', 'shape_factor']
    # sort final cnt group by cover_area , shape_factor and color_gradient
    for obvious_para in obvious_list:

        if obvious_para == 'color_gradient':
            avg_gradient = avg_img_gradient(img)
            final_group.append({'cnt': [], 'cover_area': [], 'color_gradient': avg_gradient, 'shape_factor': [],
                                'obvious_weight': -1})

        final_group.sort(key=lambda x: x[obvious_para], reverse=True)
        obvious_index = len(final_group) - 1
        max_diff = 0
        area_list = [final_group[0][obvious_para]]

        if obvious_para == 'color_gradient' and final_group[0]['obvious_weight'] < 0:
            final_group.remove({'cnt': [], 'cover_area': [], 'color_gradient': avg_gradient, 'shape_factor': [],
                                'obvious_weight': -1})
            print('No color_gradient result')
            continue

        for i in range(1, len(final_group)):
            area_list.append(final_group[i][obvious_para])
            diff = final_group[i - 1][obvious_para] - final_group[i][obvious_para]

            '''0.5 Changeable'''
            if diff > max_diff:
                if obvious_para == 'cover_area' and 0.5 * final_group[i - 1][obvious_para] < final_group[i][
                    obvious_para]:
                    continue

                max_diff = diff
                obvious_index = i - 1

        print('obvious_index:', obvious_index)

        for i in range(obvious_index + 1):
            if final_group[i]['obvious_weight'] == -1:
                obvious_index = i
                break

            final_group[i]['obvious_weight'] += 1
            cv2.drawContours(contour_image, np.array(final_group[i]['cnt']), -1, (0, 255, 0), 2)    # GREEN

        for i in range(obvious_index + 1, len(final_group)):
            COLOR = (0, 0, 255) # RED
            '''0.8 Changeable'''
            if obvious_para == 'shape_factor' and final_group[i]['shape_factor'] >= 0.8:
                COLOR = (0, 255, 0) # GREEN
                final_group[i]['obvious_weight'] += 1
            cv2.drawContours(contour_image, np.array(final_group[i]['cnt']), -1, COLOR, 2)

        if DRAW_PROCESS:
            cv2.imwrite(output_dir + img_name + '_h_para[' + obvious_para + ']_obvious(Green).jpg', contour_image)

        plt.bar(x=range(len(area_list)), height=area_list)
        plt.title(obvious_para + ' cut_point : ' + str(obvious_index) + '  | value: ' + str(
            final_group[obvious_index][obvious_para]))

        if DRAW_PROCESS:
            plt.savefig(output_dir + img_name + '_h_para[' + obvious_para + ']_obvious_his.png')
        plt.close()

        if obvious_para == 'color_gradient':
            final_group.remove({'cnt': [], 'cover_area': [], 'color_gradient': avg_gradient, 'shape_factor': [],
                                'obvious_weight': -1})

    # end obvious para for
    final_obvious_group = []
    # "vote (0-3) " to decide which groups to remain
    final_group.sort(key=lambda x: x['obvious_weight'], reverse=True)
    weight = final_group[0]['obvious_weight']

    for f_group in final_group:

        # determine obvious if match more than two obvious condition
        if f_group['obvious_weight'] == weight:
            final_obvious_group.append(f_group)
    # end choose obvious way if

    final_nonoverlap_cnt_group = []
    compare_overlap_queue = []
    total_group_number = len(final_obvious_group)
    # get all group cnt and filter overlap
    for group_index in range(total_group_number):
        cnt_group = final_obvious_group[group_index]['group_dic']

        for cnt_dic in cnt_group:
            compare_overlap_queue.append(
                {'cnt': cnt_dic['cnt'], 'label': group_index, 'group_weight': len(cnt_group),
                    'color': cnt_dic['color']})

    for label_i in range(total_group_number):
        tmp_group = []
        avg_color = [0, 0, 0]
        avg_edge_number = 0
        avg_size = 0
        for cnt_dic in compare_overlap_queue:
            if cnt_dic['label'] == label_i:
                tmp_group.append(cnt_dic['cnt'])
                '''10 Changeable'''
                approx = cv2.approxPolyDP(cnt_dic['cnt'], 10, True)
                factor = 4 * np.pi * cv2.contourArea(cnt_dic['cnt']) / float(pow(len(cnt_dic['cnt']), 2))
                if factor < 0.9:
                    avg_edge_number += len(approx)

                for i in range(3):
                    avg_color[i] += cnt_dic['color'][i]

                avg_size += cv2.contourArea(cnt_dic['cnt'])

        # end compare_overlap_queue for
        if len(tmp_group) < 1:
            continue
        count = len(tmp_group)
        avg_edge_number /= count
        avg_size /= count
        for i in range(3):
            avg_color[i] /= count

        final_nonoverlap_cnt_group.append(
            {'cnt': tmp_group, 'edge_number': avg_edge_number, 'color': avg_color, 'size': avg_size,
                'count': count})

    # end each label make group for

    # draw final result
    final_group_cnt = []
    img = resi_input_img / 3.0    # darken the image to make the contour visible
    # sort list from little to large
    final_nonoverlap_cnt_group.sort(key=lambda x: len(x['cnt']), reverse=False)

    for tmp_group in final_nonoverlap_cnt_group:

        if len(tmp_group) < 2:
            continue

        final_group_cnt.append(tmp_group['cnt'])
        img = drawer.draw_one_color(tmp_group['cnt'], img)

    if _evaluate:
        resize_ratio = resize_height / float(img_height)
        tp, fp, fn, pr, re, fm, er = evaluate_detection_performance(img, img_name, final_group_cnt,
                                                                    resize_ratio, evaluate_csv_path)
        evaluation_csv.append([img_name, tp, fp, fn, pr, re, fm, er])

    contour_image = cv2.resize(img, (0, 0), fx=1/resize_factor, fy=1/resize_factor)
    combine_image = np.concatenate((input_img, contour_image), axis=1)

    cv2.imwrite(output_dir + img_name + '_l_FinalResult.jpg', combine_image)

    print('Finished in ', time.time() - start_time, ' s')

    print('-----------------------------------------------------------')
    each_img_time = time.time() - start_time
    if each_img_time > max_time:
        max_time = each_img_time
        max_time_img = img_name
    if each_img_time < min_time:
        min_time = each_img_time
        min_time_img = img_name


if _evaluate:
    f = open(evaluate_csv_path + 'evaluate-bean.csv', "wb")
    w = csv.writer(f)
    w.writerows(evaluation_csv)
    f.close()

print('img:', max_time_img, ' max_time:', max_time, 's')
print('img:', min_time_img, 'min_time:', min_time, 's')
print('All finished in ', time.time() - t_start_time, ' s')

