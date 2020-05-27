import numpy as np
import cv2
import os
import time
import csv
import math
import get_contour_feature
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
import matplotlib.pyplot as plt

import ipdb

GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
ORANGE = (0, 128, 255)
YELLOW = (0, 255, 255)
LIGHT_BLUE = (255, 255, 0)
PURPLE = (205, 0, 205)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# switchColor = [(255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 128, 0),
#                (255, 0, 128), (128, 0, 255), (128, 255, 0), (0, 128, 255), (0, 255, 128), (128, 128, 0), (128, 0, 128),
#                (0, 128, 128), (255, 64, 0), (255, 0, 64), (64, 255, 0), (64, 0, 255), (0, 255, 64), (0, 64, 255)]

switchColor = [(255, 255, 0), (0, 255, 128), (117, 203, 0), (188, 169, 249), (254, 87, 249), (255,255,255)]
# 736 will make the colony perfomance the best. (ref to yun-tao colony)
resize_height = 736.0
# sliding window's split number 
split_n_row = 1
split_n_column = 1

gaussian_para = 3

# Several Flags
_sharpen = True

# Filter Flag
_remove_small_and_big = True
_remove_high_density = True
_remove_too_many_edge = True

_gaussian_filter = True
# Tells that which method is used first
_use_structure_edge = True
# Only used in SF
_enhance_edge = True
# Decide whether local/global equalization to use (True-->Local)
_gray_value_redistribution_local = True
# Decide if excecute 1st evalution 
_evaluate = False

input_path = '../data/general/image/'
# structure forest output
edge_input_path = '../data/general/edge/SF_matlab/'
output_path = '../output/legacy_matlabSF/'

csv_output = '../output_csv_6_8[combine_result_before_filter_obvious]/'
evaluate_csv_path = '../evaluate_data/groundtruth_csv/generalize_csv/'
# When doing Canny edge detection , this para decide which channel to be the gradient standard
_edge_by_channel = ['bgr_gray']

_showImg = {'original_image': False, 'original_edge': False, 'enhanced_edge': False, 'original_contour': False,
            'contour_filtered': False, 'size': False, 'shape': False, 'color': False, 'cluster_histogram': False,
            'original_result': False, 'each_obvious_result': False, 'combine_obvious_result': False,
            'obvious_histogram': False, 'each_group_result': False, 'result_obvious': False,
            'final_each_group_result': False, 'final_result': False}
_writeImg = {'original_image': False, 'original_edge': False, 'enhanced_edge': False, 'original_contour': False,
            'contour_filtered': False, 'size': False, 'shape': False, 'color': False, 'cluster_histogram': False,
            'original_result': False, 'each_obvious_result': False, 'combine_obvious_result': False,
            'obvious_histogram': False, 'each_group_result': False, 'result_obvious': False,
            'final_each_group_result': False, 'final_result': True}

_show_resize = [(720, 'height'), (1200, 'width')][0]

test_one_img = {'test': False, 'filename': 'IMG_ (62).jpg'}


def main():
    max_time_img = ''
    min_time_img = ''
    min_time = 99999.0
    max_time = 0.0

    evaluation_csv = [['Image name', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F_measure', 'Error_rate']]

    # line88 - line113 input exception
    for i, fileName in enumerate(os.listdir(input_path)):

        if fileName[-3:] != 'jpg' and fileName[-3:] != 'JPG' and fileName[-4:] != 'jpeg' and fileName[-3:] != 'png':
            print("Wrong format file: " + fileName)
            continue

        start_time = time.time()

        if test_one_img['test'] and i > 0:
            break

        if test_one_img['test']:
            fileName = test_one_img['filename']

        print('Input:', fileName)

        if not os.path.isfile(input_path + fileName):
            print(input_path + fileName)
            print('FILE does not exist!')
            break

        # ===========================
        # decide whcih method (canny/structure forest)
        _use_structure_edge = True
        # ============================

        if not os.path.isfile(edge_input_path + fileName[:-4] + '_edge.jpg') and _use_structure_edge:
            print(edge_input_path + fileName[:-4] + '_edge.jpg')
            print('EDGE FILE does not exist!')
            break

        # read color image
        color_image_ori = cv2.imread(input_path + fileName)  # (1365, 2048, 3)

        height, width = color_image_ori.shape[:2]  # 1365, 2048

        # resize_height=736, image_resi.shape=(736,1104,3)
        image_resi = cv2.resize(color_image_ori, (0, 0), fx=resize_height / height, fy=resize_height / height)

        if _writeImg['original_image']:
            cv2.imwrite(output_path + fileName[:-4] + '_a_original_image.jpg', image_resi)

            # combine two edge detection result
        final_differ_edge_group = []

        # check if two edge detection method is both complete

        for j in range(2):

            edge_type = 'structure'

            if _use_structure_edge:

                # read edge image from matlab 
                edge_image_ori = cv2.imread(edge_input_path + fileName[:-4] + '_edge.jpg', cv2.IMREAD_GRAYSCALE)
                height, width = edge_image_ori.shape[:2]
                edged = cv2.resize(edge_image_ori, (0, 0), fx=resize_height / height, fy=resize_height / height)

            else:
                edge_type = 'canny'

                # filter the noise
                if _gaussian_filter:
                    print('Gaussian filter')
                    image_resi = cv2.GaussianBlur(image_resi, (gaussian_para, gaussian_para), 0)

                if _sharpen:
                    print('Sharpening')
                    image_resi = Sharpen(image_resi)

                re_height, re_width = image_resi.shape[:2]

                offset_r = re_height / split_n_row  # 736/1 = 736
                offset_c = re_width / split_n_column  # 1104

                print('Canny Detect edge')
                edged = np.zeros(image_resi.shape[:2], np.uint8)

                for row_n in np.arange(0, split_n_row, 0.5):  # 0, 0.5
                    for column_n in np.arange(0, split_n_column, 0.5):  # 0, 0.5

                        r_l = int(row_n * offset_r)  # 736*0, 736*0.5
                        r_r = int((row_n + 1) * offset_r)  # 736*1, 736*1.5->736*1
                        c_l = int(column_n * offset_c)
                        c_r = int((column_n + 1) * offset_c)

                        if row_n == split_n_row - 0.5:
                            r_r = int(re_height)
                        if column_n == split_n_column - 0.5:
                            c_r = int(re_width)

                        BGR_dic, HSV_dic, LAB_dic = SplitColorChannel(image_resi[r_l: r_r, c_l: c_r])

                        channel_img_dic = {'bgr_gray': BGR_dic['img_bgr_gray'], 'b': BGR_dic['img_b'],
                                           'g': BGR_dic['img_g'], 'r': BGR_dic['img_r'], 'h': HSV_dic['img_h'],
                                           's': HSV_dic['img_s'], 'v': HSV_dic['img_v'], 'l': LAB_dic['img_l'],
                                           'a': LAB_dic['img_a'], 'b': LAB_dic['img_b']}
                        channel_thre_dic = {'bgr_gray': BGR_dic['thre_bgr_gray'], 'b': BGR_dic['thre_b'],
                                            'g': BGR_dic['thre_g'], 'r': BGR_dic['thre_r'], 'h': HSV_dic['thre_h'],
                                            's': HSV_dic['thre_s'], 'v': HSV_dic['thre_v'], 'l': LAB_dic['thre_l'],
                                            'a': LAB_dic['thre_a'], 'b': LAB_dic['thre_b']}

                        for chan in _edge_by_channel:
                            if channel_thre_dic[chan] > 20:
                                edged[r_l: r_r, c_l: c_r] |= cv2.Canny(
                                    channel_img_dic[chan], channel_thre_dic[chan] * 0.5, channel_thre_dic[chan])

            # end detect edge else
            if _showImg['original_edge']:
                cv2.imshow(fileName + ' origianl_edge[' + str(edge_type) + ']', ShowResize(edged))
                cv2.waitKey(100)
            if _writeImg['original_edge']:
                cv2.imwrite(output_path + fileName[:-4] + '_b_original_edge[' + str(edge_type) + '].jpg', edged)

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

                if _showImg['enhanced_edge']:
                    cv2.imshow(fileName + ' enhanced_edge[' + str(edge_type) + ']', ShowResize(edged))
                    cv2.waitKey(100)
                if _writeImg['enhanced_edge']:
                    cv2.imwrite(output_path + fileName[:-4] + '_c_enhanced_edge[' + str(edge_type) + '].jpg', edged)
                    # end enhance edge if

            if _use_structure_edge:
                _use_structure_edge = False
            else:
                _use_structure_edge = True

            print('Find countour')
            edged = cv2.threshold(edged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            # ==============
            # contour detection 
            contours = cv2.findContours(edged, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[-2]
            # ==============
            contour_image = np.zeros(image_resi.shape, np.uint8)

            color_index = 0
            for c in contours:
                COLOR = switchColor[color_index % len(switchColor)]
                color_index += 1
                cv2.drawContours(contour_image, [c], -1, COLOR, 2)

            if _showImg['original_contour']:
                cv2.imshow(fileName + ' original_contour[' + str(edge_type) + ']', ShowResize(contour_image))
                cv2.waitKey(100)
            if _writeImg['original_contour']:
                cv2.imwrite(output_path + fileName[:-4] + '_d_original_contour[' + str(edge_type) + '].jpg',
                            contour_image)

            tmp_cnt_list = [contours[0]]
            # the first contour
            tmp_cnt = contours[0]
            # check isOverlap
            # Since if is overlap , the order of the overlapped contours will be continuous 
            # The following four lines the goal is as same as CheckOverlap(.., 'keep inner') and much more easier.
            for c in contours[1:]:
                if not IsOverlap(tmp_cnt, c):
                    tmp_cnt_list.append(c)
                tmp_cnt = c

            contours = tmp_cnt_list

            noise = 0
            contour_list = []
            re_height, re_width = image_resi.shape[:2]

            # line264 - line 376 Find Contour and Filter
            print('Filter contour')
            print('------------------------')

            # decide if use small contour filter
            small_cnt_cover_area = 0.0
            small_cnt_count = 0
            for c in contours:
                cnt_area = max(len(c), cv2.contourArea(c))
                if cnt_area < 60:
                    small_cnt_cover_area += cnt_area
                    small_cnt_count += 1

            # normal pic for small noise more than 500
            '''60 Changeable'''
            if small_cnt_count > 500:
                cnt_min_size = 60
            # colony pic for small noise less than 500 (400 for colonies and 100 for error tolerance)
            else:
                cnt_min_size = 4

            # remove contours by some limitations    
            for c in contours:

                # CountCntArea(c,image_resi)

                if _remove_small_and_big:
                    # remove too small or too big contour
                    # contour perimeter less than 1/3 image perimeter 
                    '''Changeable'''
                    if len(c) < cnt_min_size or len(c) > (re_height + re_width) * 2 / 3.0:
                        continue

                if _remove_high_density:
                    # remove contour whose density is too large or like a line
                    area = cv2.contourArea(c)
                    shape_factor = 4 * np.pi * area / float(pow(len(c), 2))
                    if cv2.contourArea(cv2.convexHull(c)) == 0:
                        continue
                    solidity = area / cv2.contourArea(cv2.convexHull(c))
                    if area < 4 or solidity < 0.5:
                        noise += 1
                        continue

                if _remove_too_many_edge:
                    # remove contour which has too many edge
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 10, True)

                    if len(approx) > 50:
                        continue

                contour_list.append(c)
            # end filter contour for

            if len(contour_list) == 0:
                continue

            print('------------------------')

            # draw contour by different color
            contour_image = np.zeros(image_resi.shape, np.uint8)
            contour_image[:] = BLACK
            color_index = 0
            for c in contour_list:
                COLOR = switchColor[color_index % len(switchColor)]
                color_index += 1
                cv2.drawContours(contour_image, [c], -1, COLOR, 2)

            if _showImg['contour_filtered']:
                cv2.imshow(fileName + ' contour_filtered[' + str(edge_type) + ']', ShowResize(contour_image))
                cv2.waitKey(100)
            if _writeImg['contour_filtered']:
                cv2.imwrite(output_path + fileName[:-4] + '_e_contour_filtered[' + str(edge_type) + '].jpg',
                            contour_image)
            # __________end of drawing______________________


            print('Extract contour feature')
            # line 382 - line 520 feature extraction and cluster
            # Get contour feature
            '''
            @param 
            c_list = contour_list
            cnt_dic_list : every contour will save its coordinate and following 4 attributes
            
            # 3 filter factors
            cnt_shape_list : shape vector list , dimension : dynamic
            cnt_color_list : RGB , dimension = 3
            cnt_size_list : size , dimension = 1
            
            cnt_color_gradient_list : one of obviousity factors
            '''
            c_list, cnt_shape_list, cnt_color_list, cnt_size_list, cnt_color_gradient_list = get_contour_feature.extract_feature(
                image_resi, contour_list)

            cnt_dic_list = []
            for i in range(len(c_list)):
                cnt_dic_list.append(
                    {'cnt': c_list[i], 'shape': cnt_shape_list[i], 'color': cnt_color_list[i], 'size': cnt_size_list[i],
                     'color_gradient': cnt_color_gradient_list[i]})

            feature_dic = {'cnt': c_list, 'shape': cnt_shape_list, 'color': cnt_color_list, 'size': cnt_size_list}

            para = ['size', 'shape', 'color']

            # total contour number
            cnt_N = len(c_list)

            if cnt_N < 1:
                print('No any contour!')
                continue

            label_list_dic = {}

            print('Respectively use shape, color, and size as feature set to cluster')
            # Respectively use shape, color, and size as feature set to cluster
            for para_index in range(len(para)):

                print('para:', para[para_index])

                contour_feature_list = feature_dic[para[para_index]]

                # hierarchical clustering
                # output the classified consequence
                label_list = Hierarchical_clustering(contour_feature_list, fileName, para[para_index], edge_type)

                unique_label, label_counts = np.unique(label_list, return_counts=True)

                # draw contours of each group refer to the result clustered by size, shape or color
                contour_image = np.zeros(image_resi.shape, np.uint8)
                contour_image[:] = BLACK
                color_index = 0
                for label in unique_label:
                    COLOR = switchColor[color_index % len(switchColor)]
                    color_index += 1
                    tmp_splited_group = []
                    for i in range(len(label_list)):
                        if label_list[i] == label:
                            tmp_splited_group.append(c_list[i])
                    cv2.drawContours(contour_image, np.array(tmp_splited_group), -1, COLOR, 2)

                if _showImg[para[para_index]]:
                    cv2.imshow('cluster by :' + para[para_index] + '[' + str(edge_type) + ']',
                               ShowResize(contour_image))
                    cv2.waitKey(100)
                if _writeImg[para[para_index]]:
                    cv2.imwrite(
                        output_path + fileName[:-4] + '_f_para[' + para[para_index] + ']_[' + str(edge_type) + '].jpg',
                        contour_image)

                    # save the 3 types of the classified output
                label_list_dic[para[para_index]] = label_list
            # end para_index for

            # intersect the label clustered by size, shpae, and color
            # ex: [0_1_1 , 2_0_1]
            combine_label_list = []
            for i in range(cnt_N):
                combine_label_list.append(
                    str(label_list_dic['size'][i]) + '_' + str(label_list_dic['shape'][i]) + '_' + str(
                        label_list_dic['color'][i]))

            unique_label, label_counts = np.unique(combine_label_list, return_counts=True)
            label_dic = dict(zip(unique_label, label_counts))

            # find the final group by the intersected label and draw
            final_group = []
            contour_image = np.zeros(image_resi.shape, np.uint8)
            contour_image[:] = BLACK

            color_index = 0
            for label in unique_label:
                COLOR = switchColor[color_index % len(switchColor)]
                color_index += 1
                tmp_group = []
                for i in range(cnt_N):
                    if combine_label_list[i] == label:
                        tmp_group.append(cnt_dic_list[i])

                tmp_cnt_group = []

                # for each final group count obvious factor
                for cnt_dic in tmp_group:
                    cnt = cnt_dic['cnt']
                    tmp_cnt_group.append(cnt)

                if len(tmp_cnt_group) < 2:
                    continue

                cv2.drawContours(contour_image, np.array(tmp_cnt_group), -1, COLOR, 2)

                final_group.append({'cnt': tmp_cnt_group, 'obvious_weight': 0, 'group_dic': tmp_group})

            # end find final group for
            # sort the group from the max area to min group and get max count

            if _showImg['original_result']:
                cv2.imshow(fileName + ' original_result[' + str(edge_type) + ']', ShowResize(contour_image))
                cv2.waitKey(100)
            if _writeImg['original_result']:
                cv2.imwrite(output_path + fileName[:-4] + '_g_original_result[' + str(edge_type) + '].jpg',
                            contour_image)

            if len(final_group) < 1:
                print('No any pattern')
                continue

            # ====================================================================================

            # line 536 - line 632 combine two edge detection results

            for f_edge_group in final_group:
                final_differ_edge_group.append(f_edge_group)
                # end two edge method for

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

        if _showImg['original_result']:
            cv2.imshow(fileName + ' remove_overlap_combine_cnt', ShowResize(contour_image))
            cv2.waitKey(100)
        if _writeImg['original_result']:
            cv2.imwrite(output_path + fileName[:-4] + '_g_remove_overlap_combine_cnt.jpg', contour_image)

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

            if _showImg['each_obvious_result']:
                cv2.imshow(
                    fileName + ' obvious_para:[' + obvious_para + '] | Green for obvious[' + str(edge_type) + ']',
                    ShowResize(contour_image))
                cv2.waitKey(100)
            if _writeImg['each_obvious_result']:
                cv2.imwrite(output_path + fileName[:-4] + '_h_para[' + obvious_para + ']_obvious(Green)[' + str(
                    edge_type) + '].jpg', contour_image)

            plt.bar(x=range(len(area_list)), height=area_list)
            plt.title(obvious_para + ' cut_point : ' + str(obvious_index) + '  | value: ' + str(
                final_group[obvious_index][obvious_para]) + '  |[' + str(edge_type) + ']')

            if _showImg['obvious_histogram']:
                plt.show()
            if _writeImg['obvious_histogram']:
                plt.savefig(output_path + fileName[:-4] + '_h_para[' + obvious_para + ']_obvious_his[' + str(
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

            if _showImg['final_each_group_result']:
                cv2.imshow(
                    fileName + ' _label[' + str(color_index) + ']_Count[' + str(tmp_group['count']) + ']_size[' + str(
                        tmp_group['size']) + ']_color' + str(tmp_group['color']) + '_edgeNumber[' + str(
                        tmp_group['edge_number']) + ']', ShowResize(contour_image_each))
                cv2.waitKey(0)
            if _writeImg['final_each_group_result']:
                cv2.imwrite(output_path + fileName[:-4] + '_k_label[' + str(color_index) + ']_Count[' + str(
                    tmp_group['count']) + ']_size[' + str(tmp_group['size']) + ']_color' + str(
                    tmp_group['color']) + '_edgeNumber[' + str(tmp_group['edge_number']) + '].jpg', contour_image_each)

                # end final_nonoverlap_cnt_group for

        if _evaluate:
            resize_ratio = resize_height / float(height)
            tp, fp, fn, pr, re, fm, er = Evaluate_detection_performance(image_resi, fileName, final_group_cnt,
                                                                        resize_ratio, evaluate_csv_path)
            evaluation_csv.append([fileName, tp, fp, fn, pr, re, fm, er])

        contour_image = cv2.resize(contour_image, (0, 0), fx=height / resize_height, fy=height / resize_height)
        combine_image = np.concatenate((color_image_ori, contour_image), axis=1)

        if _showImg['final_result']:
            cv2.imshow(fileName + ' final_result', ShowResize(combine_image))
            cv2.waitKey(0)
        if _writeImg['final_result']:
            cv2.imwrite(output_path + fileName[:-4] + '_l_final_result.jpg', combine_image)

        print('Finished in ', time.time() - start_time, ' s')

        print('-----------------------------------------------------------')
        each_img_time = time.time() - start_time
        if each_img_time > max_time:
            max_time = each_img_time
            max_time_img = fileName
        if each_img_time < min_time:
            min_time = each_img_time
            min_time_img = fileName

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


def Sharpen(img):
    kernel_sharpen = np.array([[-1, -1, -1, -1, -1],
                               [-1, 2, 2, 2, -1],
                               [-1, 2, 8, 2, -1],
                               [-1, 2, 2, 2, -1],
                               [-1, -1, -1, -1, -1]]) / 8.0

    return cv2.filter2D(img, -1, kernel_sharpen)


def Eucl_distance(a, b):
    if type(a) != np.ndarray:
        a = np.array(a)
    if type(b) != np.ndarray:
        b = np.array(b)

    return np.linalg.norm(a - b)


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


def IsOverlap(cnt1, cnt2):
    '''
    Determine that if one contour contains another one.
    '''

    if cnt1 == [] or cnt2 == []:
        return False

    c1M = GetCentroid(cnt1)
    c2M = GetCentroid(cnt2)
    c1_min_d = MinDistance(cnt1)
    c2_min_d = MinDistance(cnt2)
    moment_d = Eucl_distance(c1M, c2M)

    if min(c1_min_d, c2_min_d) == 0:
        return False

    return (moment_d < c1_min_d or moment_d < c2_min_d) and max(c1_min_d, c2_min_d) / min(c1_min_d, c2_min_d) <= 3


def IsOverlapAll(cnt_dic, cnt_dic_list):
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


def SplitColorChannel(img):
    '''
    Find all the attribute of three color models. (RGB/HSV/LAB)
    Return in a dictionary type.
    '''

    bgr_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # bgr_gray = cv2.GaussianBlur(bgr_gray, (3, 3), 0)
    thresh_bgr_gray = cv2.threshold(bgr_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]

    bgr_b = img[:, :, 0]
    bgr_g = img[:, :, 1]
    bgr_r = img[:, :, 2]
    bgr_b = cv2.GaussianBlur(bgr_b, (5, 5), 0)
    bgr_g = cv2.GaussianBlur(bgr_g, (5, 5), 0)
    bgr_r = cv2.GaussianBlur(bgr_r, (5, 5), 0)
    thresh_bgr_b = cv2.threshold(bgr_b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    thresh_bgr_g = cv2.threshold(bgr_g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    thresh_bgr_r = cv2.threshold(bgr_r, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
    hsv_h = hsv[:, :, 0]
    hsv_s = hsv[:, :, 1]
    hsv_v = hsv[:, :, 2]
    hsv_h = cv2.GaussianBlur(hsv_h, (5, 5), 0)
    hsv_s = cv2.GaussianBlur(hsv_s, (5, 5), 0)
    hsv_v = cv2.GaussianBlur(hsv_v, (5, 5), 0)
    thresh_hsv_h = cv2.threshold(hsv_h, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    thresh_hsv_s = cv2.threshold(hsv_s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    thresh_hsv_v = cv2.threshold(hsv_v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab = cv2.GaussianBlur(lab, (5, 5), 0)
    lab_l = lab[:, :, 0]
    lab_a = lab[:, :, 1]
    lab_b = lab[:, :, 2]
    lab_l = cv2.GaussianBlur(lab_l, (5, 5), 0)
    lab_a = cv2.GaussianBlur(lab_a, (5, 5), 0)
    lab_b = cv2.GaussianBlur(lab_b, (5, 5), 0)
    thresh_lab_l = cv2.threshold(lab_l, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    thresh_lab_a = cv2.threshold(lab_a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    thresh_lab_b = cv2.threshold(lab_b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]

    return {'img_bgr_gray': bgr_gray, 'img_bgr': img, 'img_b': bgr_b, 'img_g': bgr_g, 'img_r': bgr_r,
            'thre_bgr_gray': thresh_bgr_gray, 'thre_b': thresh_bgr_b, 'thre_g': thresh_bgr_g, 'thre_r': thresh_bgr_r
            }, {'img_hsv': hsv, 'img_h': hsv_h, 'img_s': hsv_s, 'img_v': hsv_v, 'thre_h': thresh_hsv_h,
                'thre_s': thresh_hsv_s, 'thre_v': thresh_hsv_v
                }, {'img_lab': lab, 'img_l': lab_l, 'img_a': lab_a, 'img_b': lab_b, 'thre_l': thresh_lab_l,
                    'thre_a': thresh_lab_a, 'thre_b': thresh_lab_b}


def ShowResize(img):
    '''
    Resize the image depend on ratio parameter('_show_resize[]') before showing image.
    
    @Return : a resized image
    '''

    h, w = img.shape[:2]

    if _show_resize[1] == 'height':
        ratio = _show_resize[0] / float(h)
    else:
        ratio = _show_resize[0] / float(w)

    return cv2.resize(img, (0, 0), fx=ratio, fy=ratio)


def MinDistance(cnt):
    '''
    Calculate the minimum distance between centroid to the contour.
    '''

    cM = GetCentroid(cnt)
    if len(cnt[0][0]) == 1:
        cnt = cnt[0]
    min_d = Eucl_distance((cnt[0][0][0], cnt[0][0][1]), cM)
    for c in cnt:
        d = Eucl_distance((c[0][0], c[0][1]), cM)
        if d < min_d:
            min_d = d

    return min_d


def LAB2Gray(img):
    _w, _h, _c = img.shape

    gray = np.zeros(img.shape[:2], np.uint8)

    for i in range(_w):
        for k in range(_h):
            a = int(img[i][k][1])
            b = int(img[i][k][2])
            gray[i, k] = (a + b) / 2

    return gray


def GetCentroid(cnt):
    '''
    Calculate the average coordinate as centroid.
    '''

    num = len(cnt)
    if num < 2:
        return cnt
    cx = 0
    cy = 0
    for c in cnt:
        if isinstance(c[0][0], np.ndarray):
            c = c[0]
        cx += float(c[0][0])
        cy += float(c[0][1])

    return float(cx) / num, float(cy) / num


def Hierarchical_clustering(feature_list, fileName, para, edge_type, cut_method='elbow'):
    '''
    @Goal 
    Cluster the contours.
    
    ref1 : https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
    ref2 : http://mirlab.org/jang/books/dcpr/dcHierClustering.asp?title=3-2%20Hierarchical%20Clustering%20(%B6%A5%BCh%A6%A1%A4%C0%B8s%AAk)&language=chinese
    '''

    if len(feature_list) < 2:
        return [0] * len(feature_list)

    all_same = True
    for feature in feature_list:
        if feature != feature_list[0]:
            all_same = False
            break

    if all_same:
        print('all in one group!')
        return [0] * len(feature_list)

        # hierarchically link cnt by order of distance from distance method 'ward'
    # Output a hierarchical tree as ppt. page 22.
    cnt_hierarchy = linkage(feature_list, 'ward')

    '''Combine two groups only when the group distance is smaller than the max__cut_distance.'''
    max_cut_distance = 0
    if cut_method == 'elbow' or True:
        last = cnt_hierarchy[:, 2]
        last = [x for x in last if x > 0]

        '''distance of grooup distance'''
        acceleration = np.diff(last)

        # acceleration = map(abs, np.diff(acceleration) )

        # acceleration_rev = acceleration[::-1]
        # print 'acceleration:',acceleration

        if len(acceleration) < 2:
            return [0] * len(feature_list)
        avg_diff = sum(acceleration) / float(len(acceleration))
        tmp = acceleration[0]

        avg_list = [x for x in acceleration if x > avg_diff]
        avg_diff = sum(avg_list) / float(len(avg_list))

        '''
        5 Changeable, compute a ratio as a reference which decide the max_cut_distance (dynamic).
        Which the ratio is (its' own distance of group distance ) / (5 previous (if it exists ) distance of group distance average. )
        '''
        off_set = 5

        rario = []
        cut_point_list = []
        for i in range(1, len(acceleration)):

            if acceleration[i] > avg_diff:
                # cut_point_list.append( [ i, acceleration[i]/(tmp/float(i) ) ] )

                tmp_offset_prev = off_set
                prev = i - off_set
                if prev < 0:
                    prev = 0
                    tmp_offset_prev = i - prev
                rario.append(acceleration[i] / (sum(acceleration[prev:i]) / float(tmp_offset_prev)))
                cut_point_list.append([i, acceleration[i] / (sum(acceleration[prev:i]) / float(tmp_offset_prev))])
                # cut_point_list.append( [ i, acceleration[i] ] )
                # print 'i:',i+1,' ratio:',acceleration[i]/( sum(acceleration[n:i]) / float(off_set) )

            tmp += acceleration[i]

        if len(cut_point_list) < 1:
            print('all in one group!')
            return [0] * len(feature_list)

        cut_point_list.sort(key=lambda x: x[1], reverse=True)

        # print 'cut index:',cut_point_list[0][0]+1,' diff len:',len(acceleration)
        max_cut_distance = last[cut_point_list[0][0]]
        max_ratio = cut_point_list[0][1]

        if max_ratio < 2.0:
            print('all in one group! max_ratio:', max_ratio)
            return [0] * len(feature_list)

            # max_cut_distance = last[acceleration.argmax()]
    # elif cut_method == 'inconsistency':

    # plt.bar(left=range(len(rario)),height=rario)
    plt.bar(x=range(len(acceleration)), height=acceleration)
    plt.title(para + ' cut_point : ' + str(cut_point_list[0][0] + 1) + '  | value: ' + str(
        round(acceleration[cut_point_list[0][0]], 2)) + ' | ratio: ' + str(round(max_ratio, 2)))

    if _showImg['cluster_histogram']:
        plt.show()
    if _writeImg['cluster_histogram']:
        plt.savefig(output_path + fileName[:-4] + '_f_para[' + para + ']_his[' + str(edge_type) + '].png')
    plt.close()

    # print 'acceleration.argmax():',acceleration.argmax()
    clusters = fcluster(cnt_hierarchy, max_cut_distance, criterion='distance')
    print('----------------------------------')
    return clusters


if __name__ == '__main__':
    t_start_time = time.time()

    main()
    # _local = False
    ##output_path = './output_global/'    
    # main()
    print('All finished in ', time.time() - t_start_time, ' s')
