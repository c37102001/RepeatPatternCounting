import numpy as np
import cv2
import os
import time
import csv
import argparse

import matplotlib.pyplot as plt

from canny import canny_edge_detect
from utils import check_overlap, avg_img_gradient, evaluate_detection_performance
from misc import check_and_cluster, ContourDrawer
from ipdb import set_trace as pdb
from tqdm import tqdm

GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
ORANGE = (0, 128, 255)
YELLOW = (0, 255, 255)
LIGHT_BLUE = (255, 255, 0)
PURPLE = (205, 0, 205)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
switchColor = [(255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 128, 0),
               (255, 0, 128), (128, 0, 255), (128, 255, 0), (0, 128, 255), (0, 255, 128), (128, 128, 0), (128, 0, 128),
               (0, 128, 128), (255, 64, 0), (255, 0, 64), (64, 255, 0), (64, 0, 255), (0, 255, 64), (0, 64, 255)]

# 736 will make the colony perfomance the best. (ref to yun-tao colony)
resize_height = 736.0
# sliding window's split number

# Tells that which method is used first
_use_structure_edge = True
# Only used in SF
_enhance_edge = True
# Decide whether local/global equalization to use (True-->Local)
_gray_value_redistribution_local = True
# Decide if excecute 1st evalution
_evaluate = False

input_path = '../input/image/'
# edge_path = '../input/edge_image/'  # structure forest output
# output_path = '../output/'
edge_path = '../input/hed_edge_image/'  # hed edge
output_path = '../output_hed/'

csv_output = '../output_csv_6_8[combine_result_before_filter_obvious]/'
evaluate_csv_path = '../evaluate_data/groundtruth_csv/generalize_csv/'

DRAW_PROCESS = False
IMG_LIST = ['IMG_ (19).jpg', 'IMG_ (28).jpg', 'IMG_ (29).jpg', ]
TEST = False
TEST_IMG = 'IMG_ (1).jpg'
EXCEPTION_LIST = ['IMG_ (3).jpg', 'IMG_ (9).jpg']

def get_edge_group(drawer, edged, edge_type, do_enhance=False, do_draw=False):
    if do_draw:
        img_path = '{}{}_b_OriginEdge{}.jpg'.format(output_path, img_name, edge_type)
        cv2.imwrite(img_path, edged)

    if do_enhance:  
        # Enhance edge
        if _gray_value_redistribution_local:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            edged = clahe.apply(edged)
        else:
            edged = cv2.equalizeHist(edged) # golbal equalization

        if do_draw:
            cv2.imwrite(output_path + img_name + '_c_enhanced_edge[' + str(edge_type) + '].jpg', edged)

    # find contours
    edged = cv2.threshold(edged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(edged, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[-2]
    contours.sort(key=lambda x: len(x), reverse=False)

    final_group = check_and_cluster(contours, drawer, edge_type, DRAW_PROCESS)

    return final_group


t_start_time = time.time()
max_time_img = ''
min_time_img = ''
min_time = 99999.0
max_time = 0.0
evaluation_csv = [['Image name', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F_measure', 'Error_rate']]


# for i, img_name in enumerate(tqdm(os.listdir(input_path))):
for i, img_name in enumerate(tqdm(IMG_LIST)):
    start_time = time.time()

    if TEST:
        img_name = TEST_IMG

    print('[Input] %s' % img_name)
    color_image_ori = cv2.imread(input_path + img_name)  # (1365, 2048, 3)
    img_name = '.'.join(img_name.split('.')[:-1])        # remove .jpg
    height, width = color_image_ori.shape[:2]  # 1365, 2048
    # resize_height=736, image_resi.shape=(736,1104,3)
    image_resi = cv2.resize(color_image_ori, (0, 0), fx=resize_height / height, fy=resize_height / height)
    drawer = ContourDrawer(image_resi, output_path, img_name)
    if DRAW_PROCESS:
        cv2.imwrite(output_path + img_name + '_a_original_image.jpg', image_resi)

    # decide whcih method (canny/structure forest)
    _use_structure_edge = True
    if _use_structure_edge:
        # edge_img = edge_path + img_name + '_edge.jpg'
        edge_img = edge_path + img_name + '_hed.png'
        # assert os.path.isfile(edge_img), 'EDGE FILE does not exist!'
        if not os.path.isfile(edge_img):
            print('EDGE FILE does not exist!')
            continue

    # combine two edge detection result
    final_differ_edge_group = []

    # check if two edge detection method is both complete
    canny_edge = canny_edge_detect(image_resi)
    canny_group = get_edge_group(drawer, canny_edge, edge_type='Canny', do_draw=DRAW_PROCESS)
    for edge_group in canny_group:
        final_differ_edge_group.append(edge_group)

    if _use_structure_edge:
        structure_edge = cv2.imread(edge_img, cv2.IMREAD_GRAYSCALE)
        structure_edge = cv2.resize(structure_edge, (0, 0), fx=resize_height / height, fy=resize_height / height)
        structure_group = get_edge_group(drawer, structure_edge, edge_type='Structure', do_enhance=True, do_draw=DRAW_PROCESS)
        for edge_group in structure_group:
            final_differ_edge_group.append(edge_group)

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

    
    drawer.reset()
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
        drawer.draw(np.array(tmp_cnt_group))

        final_group.append({'cnt': tmp_cnt_group, 'avg_area': avg_area, 'cover_area': cnt_area,
                            'color_gradient': avg_color_gradient, 'shape_factor': avg_shape_factor,
                            'obvious_weight': 0, 'group_dic': tmp_group})

    if DRAW_PROCESS:
        desc = 'g_RemoveOverlapCombineCnt'
        drawer.save(desc)

    # line 637 - line 712 obviousity filter
    contour_image = drawer.contour_image
    obvious_list = ['cover_area', 'color_gradient', 'shape_factor']
    # sort final cnt group by cover_area , shape_factor and color_gradient
    for obvious_para in obvious_list:

        if obvious_para == 'color_gradient':
            avg_gradient = avg_img_gradient(image_resi)
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
            cv2.drawContours(contour_image, np.array(final_group[i]['cnt']), -1, GREEN, 2)

        for i in range(obvious_index + 1, len(final_group)):
            COLOR = RED
            '''0.8 Changeable'''
            if obvious_para == 'shape_factor' and final_group[i]['shape_factor'] >= 0.8:
                COLOR = GREEN
                final_group[i]['obvious_weight'] += 1
            cv2.drawContours(contour_image, np.array(final_group[i]['cnt']), -1, COLOR, 2)

        if DRAW_PROCESS:
            cv2.imwrite(output_path + img_name + '_h_para[' + obvious_para + ']_obvious(Green).jpg', contour_image)

        plt.bar(x=range(len(area_list)), height=area_list)
        plt.title(obvious_para + ' cut_point : ' + str(obvious_index) + '  | value: ' + str(
            final_group[obvious_index][obvious_para]))

        if DRAW_PROCESS:
            plt.savefig(output_path + img_name + '_h_para[' + obvious_para + ']_obvious_his.png')
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
    contour_image = image_resi.copy()
    contour_image[:] = contour_image[:] / 3.0    # darken the image to make the contour visible

    # sort list from little to large
    final_nonoverlap_cnt_group.sort(key=lambda x: len(x['cnt']), reverse=False)

    for tmp_group in final_nonoverlap_cnt_group:

        if len(tmp_group) < 2:
            continue

        final_group_cnt.append(tmp_group['cnt'])

        contour_image = drawer.draw(tmp_group['cnt'], contour_image)

    if _evaluate:
        resize_ratio = resize_height / float(height)
        tp, fp, fn, pr, re, fm, er = evaluate_detection_performance(image_resi, img_name, final_group_cnt,
                                                                    resize_ratio, evaluate_csv_path)
        evaluation_csv.append([img_name, tp, fp, fn, pr, re, fm, er])

    contour_image = cv2.resize(contour_image, (0, 0), fx=height / resize_height, fy=height / resize_height)
    combine_image = np.concatenate((color_image_ori, contour_image), axis=1)

    cv2.imwrite(output_path + img_name + '_l_FinalResult.jpg', combine_image)

    print('Finished in ', time.time() - start_time, ' s')

    print('-----------------------------------------------------------')
    each_img_time = time.time() - start_time
    if each_img_time > max_time:
        max_time = each_img_time
        max_time_img = img_name
    if each_img_time < min_time:
        min_time = each_img_time
        min_time_img = img_name

    if TEST:
        break

if _evaluate:
    f = open(evaluate_csv_path + 'evaluate-bean.csv', "wb")
    w = csv.writer(f)
    w.writerows(evaluation_csv)
    f.close()

print('img:', max_time_img, ' max_time:', max_time, 's')
print('img:', min_time_img, 'min_time:', min_time, 's')
print('All finished in ', time.time() - t_start_time, ' s')

