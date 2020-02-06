import numpy as np
import cv2
import os
import time
import csv
import math
import argparse

import matplotlib.pyplot as plt

from canny import canny_edge_detect
from contour import find_contour, check_property, check_overlap
from misc import check_and_cluster, ContourDrawer
from ipdb import set_trace as pdb

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


# Filter Flag
_remove_small_and_big = True
_remove_high_density = True
_remove_too_many_edge = True

# Tells that which method is used first
_use_structure_edge = True
# Only used in SF
_enhance_edge = True
# Decide whether local/global equalization to use (True-->Local)
_gray_value_redistribution_local = True
# Decide if excecute 1st evalution
_evaluate = False

input_path = '../input/image/'
# structure forest output
edge_path = '../input/edge_image/'
output_path = '../output/'

csv_output = '../output_csv_6_8[combine_result_before_filter_obvious]/'
evaluate_csv_path = '../evaluate_data/groundtruth_csv/generalize_csv/'

_writeImg = {'original_image': False, 'original_edge': False, 'enhanced_edge': False, 'original_contour': False,
             'contour_filtered': False, 'size': True, 'shape': True, 'color': True, 'cluster_histogram': False,
             'original_result': False, 'each_obvious_result': False, 'combine_obvious_result': False,
             'obvious_histogram': False, 'each_group_result': False, 'result_obvious': False,
             'final_each_group_result': False, 'final_result': True}

_show_resize = [(720, 'height'), (1200, 'width')][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_img', type=str, default='IMG_ (33).jpg')
    args = parser.parse_args()

    max_time_img = ''
    min_time_img = ''
    min_time = 99999.0
    max_time = 0.0

    evaluation_csv = [['Image name', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F_measure', 'Error_rate']]

    for i, img_name in enumerate(os.listdir(input_path)):

        if args.test:
            img_name = args.test_img

        color_image_ori = cv2.imread(input_path + img_name)  # (1365, 2048, 3)
        print('[Input] %s' % img_name)
        img_name = '.'.join(img_name.split('.')[:-1])

        start_time = time.time()

        # decide whcih method (canny/structure forest)
        _use_structure_edge = True
        if _use_structure_edge:
            edge_img = edge_path + img_name + '_edge.jpg'
            assert os.path.isfile(edge_img), 'EDGE FILE does not exist!'

        height, width = color_image_ori.shape[:2]  # 1365, 2048

        # resize_height=736, image_resi.shape=(736,1104,3)
        image_resi = cv2.resize(color_image_ori, (0, 0), fx=resize_height / height, fy=resize_height / height)

        if _writeImg['original_image']:
            cv2.imwrite(output_path + img_name + '_a_original_image.jpg', image_resi)

        # combine two edge detection result
        final_differ_edge_group = []

        # check if two edge detection method is both complete

        for j in range(2):

            if _use_structure_edge:
                edge_type = 'structure'
                # read edge image from matlab
                edge_image_ori = cv2.imread(edge_img, cv2.IMREAD_GRAYSCALE)
                height, width = edge_image_ori.shape[:2]
                edged = cv2.resize(edge_image_ori, (0, 0), fx=resize_height / height, fy=resize_height / height)

            else:
                edge_type = 'canny'
                edged = canny_edge_detect(image_resi)

            if _writeImg['original_edge']:
                cv2.imwrite(output_path + img_name + '_b_original_edge[' + str(edge_type) + '].jpg', edged)

            if _enhance_edge and _use_structure_edge:
                # enhance and close the edge
                print('Enhance edge')
                # local equalization
                # refer to : http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
                if _gray_value_redistribution_local:

                    # create a CLAHE object (Arguments are optional).
                    # ref: https://www.jianshu.com/p/19ff65ac3844
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    edged = clahe.apply(edged)

                else:
                    # golbal equalization
                    edged = cv2.equalizeHist(edged)

                if _writeImg['enhanced_edge']:
                    cv2.imwrite(output_path + img_name + '_c_enhanced_edge[' + str(edge_type) + '].jpg', edged)
                    # end enhance edge if

            if _use_structure_edge:
                _use_structure_edge = False
            else:
                _use_structure_edge = True

            contours = find_contour(edged)

            drawer = ContourDrawer(output_path, img_name)
            final_group = check_and_cluster(image_resi, contours, drawer, edge_type, _writeImg)

            for f_edge_group in final_group:
                final_differ_edge_group.append(f_edge_group)
        
        # ================== Edited to here =======================

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

        compare_overlap_queue = CheckOverlap(compare_overlap_queue, keep='group_weight')

        color_index = 0
        contour_image = np.zeros(image_resi.shape, np.uint8)
        contour_image[:] = BLACK
        _label = [x['label'] for x in compare_overlap_queue]
        print('label_dic:', [(y, _label.count(y)) for y in set(_label)])

        final_group = []

        for label_i in range(total_group_number):
            COLOR = switchColor[color_index % len(switchColor)]
            color_index += 1
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
            cv2.drawContours(contour_image, np.array(tmp_cnt_group), -1, COLOR, 2)

            final_group.append({'cnt': tmp_cnt_group, 'avg_area': avg_area, 'cover_area': cnt_area,
                                'color_gradient': avg_color_gradient, 'shape_factor': avg_shape_factor,
                                'obvious_weight': 0, 'group_dic': tmp_group})

        # end find final group for

        if _writeImg['original_result']:
            cv2.imwrite(output_path + img_name + '_g_remove_overlap_combine_cnt.jpg', contour_image)

            # end _combine_two_edge_result_before_filter_obvious if
        # ====================================================================================
        # line 637 - line 712 obviousity filter
        obvious_list = ['cover_area', 'color_gradient', 'shape_factor']
        # sort final cnt group by cover_area , shape_factor and color_gradient
        for obvious_para in obvious_list:

            if obvious_para == 'color_gradient':
                avg_img_gradient = Avg_Img_Gredient(image_resi)
                final_group.append({'cnt': [], 'cover_area': [], 'color_gradient': avg_img_gradient, 'shape_factor': [],
                                    'obvious_weight': -1})

            final_group.sort(key=lambda x: x[obvious_para], reverse=True)
            obvious_index = len(final_group) - 1
            max_diff = 0
            area_list = [final_group[0][obvious_para]]

            if obvious_para == 'color_gradient' and final_group[0]['obvious_weight'] < 0:
                final_group.remove({'cnt': [], 'cover_area': [], 'color_gradient': avg_img_gradient, 'shape_factor': [],
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

            if _writeImg['each_obvious_result']:
                cv2.imwrite(output_path + img_name + '_h_para[' + obvious_para + ']_obvious(Green)[' + str(
                    edge_type) + '].jpg', contour_image)

            plt.bar(x=range(len(area_list)), height=area_list)
            plt.title(obvious_para + ' cut_point : ' + str(obvious_index) + '  | value: ' + str(
                final_group[obvious_index][obvious_para]) + '  |[' + str(edge_type) + ']')

            if _writeImg['obvious_histogram']:
                plt.savefig(output_path + img_name + '_h_para[' + obvious_para + ']_obvious_his[' + str(
                    edge_type) + '].png')
            plt.close()

            if obvious_para == 'color_gradient':
                final_group.remove({'cnt': [], 'cover_area': [], 'color_gradient': avg_img_gradient, 'shape_factor': [],
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
        contour_image[:] = contour_image[:] / 3.0

        # sort list from little to large
        final_nonoverlap_cnt_group.sort(key=lambda x: len(x['cnt']), reverse=False)

        for tmp_group in final_nonoverlap_cnt_group:

            if len(tmp_group) < 2:
                continue

            final_group_cnt.append(tmp_group['cnt'])
            contour_image_each = image_resi.copy()
            # darken the image to make the contour visible
            contour_image_each[:] = contour_image_each[:] / 3.0
            COLOR = switchColor[color_index % len(switchColor)]
            color_index += 1
            cv2.drawContours(contour_image, np.array(tmp_group['cnt']), -1, COLOR, 2)
            cv2.drawContours(contour_image_each, np.array(tmp_group['cnt']), -1, COLOR, 2)

            if _writeImg['final_each_group_result']:
                cv2.imwrite(output_path + img_name + '_k_label[' + str(color_index) + ']_Count[' + str(
                    tmp_group['count']) + ']_size[' + str(tmp_group['size']) + ']_color' + str(
                    tmp_group['color']) + '_edgeNumber[' + str(tmp_group['edge_number']) + '].jpg', contour_image_each)

                # end final_nonoverlap_cnt_group for

        if _evaluate:
            resize_ratio = resize_height / float(height)
            tp, fp, fn, pr, re, fm, er = Evaluate_detection_performance(image_resi, img_name, final_group_cnt,
                                                                        resize_ratio, evaluate_csv_path)
            evaluation_csv.append([img_name, tp, fp, fn, pr, re, fm, er])

        contour_image = cv2.resize(contour_image, (0, 0), fx=height / resize_height, fy=height / resize_height)
        combine_image = np.concatenate((color_image_ori, contour_image), axis=1)

        if _writeImg['final_result']:
            cv2.imwrite(output_path + img_name + '_l_final_result.jpg', combine_image)

        print('Finished in ', time.time() - start_time, ' s')

        print('-----------------------------------------------------------')
        each_img_time = time.time() - start_time
        if each_img_time > max_time:
            max_time = each_img_time
            max_time_img = img_name
        if each_img_time < min_time:
            min_time = each_img_time
            min_time_img = img_name

        if args.test:
            break

    if _evaluate:
        f = open(evaluate_csv_path + 'evaluate-bean.csv', "wb")
        w = csv.writer(f)
        w.writerows(evaluation_csv)
        f.close()

    print('img:', max_time_img, ' max_time:', max_time, 's')
    print('img:', min_time_img, 'min_time:', min_time, 's')



def Evaluate_detection_performance(img, fileName, final_group_cnt, resize_ratio, evaluate_csv_path):
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


def Avg_Img_Gredient(img, model='lab'):
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


def CheckOverlap(cnt_dic_list, keep='keep_inner'):
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
                    if IsOverlap(cnt_dic_list[cnt_i]['cnt'], cnt_dic_list[cnt_k]['cnt']):

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
            if IsOverlapAll(c_dic, checked_list):
                continue
            checked_list.append(c_dic)

            # end keep if

    return checked_list





if __name__ == '__main__':
    t_start_time = time.time()

    main()
    print('All finished in ', time.time() - t_start_time, ' s')
