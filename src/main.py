import numpy as np
import cv2
import os
import math
import time
import csv
from argparse import ArgumentParser
from configparser import ConfigParser
import matplotlib.pyplot as plt
from ipdb import set_trace as pdb

from canny import canny_edge_detect
from drawer import ContourDrawer
from get_group_cnts import get_group_cnts
from utils import check_overlap, count_avg_gradient, evaluate_detection_performance

parser = ArgumentParser()
parser.add_argument('--test', action='store_true')
parser.add_argument('--img', type=str, help='img name with extention, like "IMG_ (16).jpg"')
parser.add_argument('--do_draw', action='store_true', help='draw all figures')
args = parser.parse_args()
test = args.test
do_draw = args.do_draw

cfg = ConfigParser()
cfg.read('config.ini')

# path config
path_cfg = cfg['path']
input_dir = path_cfg['input_dir']
output_dir = path_cfg['output_dir']
strct_edge_dir = path_cfg['strct_edge_dir']
hed_edge_dir = path_cfg['hed_edge_dir']
csv_output = path_cfg['csv_output']
evaluate_csv_path = path_cfg['evaluate_csv_path']

# image config
img_cfg = cfg['img_cfg']
if not test:
    img_list = img_cfg['img_list'].split(',')
else:
    img_list = [args.img]
resize_height = img_cfg.getfloat('resize_height')
use_canny = img_cfg.getboolean('use_canny')
use_structure = img_cfg.getboolean('use_structure')
use_hed = img_cfg.getboolean('use_hed')
use_combine = img_cfg.getboolean('use_combine')
keep_overlap = img_cfg['keep_overlap'].split(',')

# evaluate config
eval_cfg = cfg['evaluate']
evaluate = eval_cfg.getboolean('evaluate')
evaluation_csv = eval_cfg['evaluation_csv'].split(',')


# for i, img_name in enumerate(os.listdir(input_dir)):
for i, img_path in enumerate(img_list):
    img_path = img_path.strip()
    print(f'\n[Progress]: {i+1} / {len(img_list)}')
    start = time.time()

    #======================================== 0. Preprocess ==========================================
    
    # check format
    img_name, img_ext = img_path.rsplit('.', 1)     # ['IMG_ (33)',  'jpg']
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
    if do_draw:
        cv2.imwrite(output_dir + img_name + '_a_original_image.jpg', resi_input_img)


    #===================================== 1. Get grouped contours  ===================================
    
    groups_cnt_dicts = []
    if use_canny:
        edge_img = canny_edge_detect(img)
        for keep in keep_overlap:
            _groups_cnt_dicts = get_group_cnts(drawer, edge_img, 'Canny', keep=keep, do_enhance=False, do_draw=do_draw)
            groups_cnt_dicts.extend(_groups_cnt_dicts)
    
    edge_imgs = []
    if use_structure:
        edge_path = strct_edge_dir + img_name + '_edge.jpg'
        edge_type = 'Structure'
        edge_imgs.append((edge_path, edge_type))
    if use_hed:
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

        for keep in keep_overlap:
            _groups_cnt_dicts = get_group_cnts(drawer, edge_img, edge_type, keep=keep, do_draw=do_draw)
            groups_cnt_dicts.extend(_groups_cnt_dicts)

    if use_combine:
        strct_edge_path = strct_edge_dir + img_name + '_edge.jpg'
        hed_edge_path = hed_edge_dir + img_name + '_hed.png'

        if os.path.isfile(strct_edge_path) and os.path.isfile(hed_edge_path):
            strct_edge = cv2.imread(strct_edge_path, cv2.IMREAD_GRAYSCALE)
            hed_edge = cv2.imread(hed_edge_path, cv2.IMREAD_GRAYSCALE)
            edge = (strct_edge + hed_edge)
            edge = cv2.resize(edge, (0, 0), fx=resize_height / img_height, fy=resize_height / img_height)

            for keep in keep_overlap:
                _groups_cnt_dicts = get_group_cnts(drawer, edge, 'Combine', keep=keep, do_draw=do_draw)
                groups_cnt_dicts.extend(_groups_cnt_dicts)
        else:
            print('[Error] Lack of edge images for combine')

    # ============================== 2. Remove group overlap and combine ==============================
    
    # add label and group weight(num of cnts in the group) into contour dictionary
    for i, group_cnt_dicts in enumerate(groups_cnt_dicts):
        for group_cnt_dict in group_cnt_dicts:
            group_cnt_dict['label'] = i
            group_cnt_dict['group_weight'] = len(group_cnt_dicts)

    # flatten to a list of contour dict
    cnt_dicts = [group_cnt_dict for group_cnt_dicts in groups_cnt_dicts for group_cnt_dict in group_cnt_dicts]
    labels = [cnt_dict['label'] for cnt_dict in cnt_dicts]  # show original labels and counts
    print('before remove overlapped (label, counts): ', [(label, labels.count(label)) for label in set(labels)])

    # check overlapped cnts and change their labels or remove them
    cnt_dicts = check_overlap(cnt_dicts)
    labels = [cnt_dict['label'] for cnt_dict in cnt_dicts]  # show labels and counts after removed overlapped cnts
    print('after remove overlapped (label, counts): ', [(label, labels.count(label)) for label in set(labels)])

    if do_draw:
        img = drawer.blank_img()
        for label in set(labels):
            cnts = [cnt_dict['cnt'] for cnt_dict in cnt_dicts if cnt_dict['label'] == label]
            img = drawer.draw_same_color(cnts, img)
        drawer.save(img, 'g_RemoveOverlapCombineCnt')

    # =============================== 3. Count group obviousity factors ===================================

    group_dicts = []
    for label in set(labels):
        group_cnt_dicts = [cnt_dict for cnt_dict in cnt_dicts if cnt_dict['label'] == label]

        group_cnts = []
        avg_color_gradient = 0.0
        avg_shape_factor = 0.0
        total_area = 0.0

        # count obvious factor for each group 
        for group_cnt_dict in group_cnt_dicts:
            cnt = group_cnt_dict['cnt']
            cnt_area = cv2.contourArea(cnt)
            convex_area = cv2.contourArea(cv2.convexHull(cnt))

            total_area += cnt_area
            avg_shape_factor += cnt_area / convex_area
            avg_color_gradient += group_cnt_dict['color_gradient']
            group_cnts.append(cnt)

        avg_shape_factor /= len(group_cnt_dicts)
        avg_color_gradient /= len(group_cnt_dicts)

        group_dicts.append({
            'group_cnts': group_cnts,
            'area': total_area,
            'color_gradient': avg_color_gradient, 
            'shape': avg_shape_factor,
            'votes': 0,
        })

    # ================================= 4. Vote groups by obviousity =====================================

    obvious_factors = ['area', 'shape', 'color_gradient']
    for factor in obvious_factors:
        group_dicts.sort(key=lambda group_dict: group_dict[factor], reverse=False)
        factor_list = [group[factor] for group in group_dicts]
        
        # find obvious index
        diff = np.diff(factor_list)
        obvious_index = np.where(diff == max(diff))[0][0] + 1

        # check cover_area
        if factor == 'area':
            obvious_area = factor_list[obvious_index]
            for i in range(obvious_index-1, -1, -1):
                area = factor_list[i]
                if area * 2 > obvious_area:     # include closed area factor
                    obvious_area = area
                    obvious_index = i
        
        # check shape factor
        elif factor == 'shape':
            obvious_shape = factor_list[obvious_index]
            for i, shape_factor in enumerate(factor_list[:obvious_index]):
                if shape_factor > 0.8 * obvious_shape:   # include closed shape factor
                    obvious_index = i
                    break
        
        # check color_gradient
        elif factor == 'color_gradient':
            avg_gradient = count_avg_gradient(resi_input_img)   # count whole image avg color gradient
            for i in range(obvious_index, len(factor_list)):    # exclude those less than avg_gradient
                if factor_list[i] < avg_gradient:
                    obvious_index += 1

            # skip if all color gradients are less tha avg_gradient
            if obvious_index == len(factor_list):
                print('No color_gradient result')
                continue
        
        for group in group_dicts[obvious_index:]:
            group['votes'] += 1

        if do_draw:
            img = drawer.blank_img()
            for group in group_dicts[obvious_index:]:
                img = drawer.draw_same_color(group['group_cnts'], img, color=(0, 255, 0))  # green for obvious
            for group in group_dicts[:obvious_index]:
                img = drawer.draw_same_color(group['group_cnts'], img, color=(0, 0, 255))  # red for others
            drawer.save(img, desc=f'h_Obvious{factor.capitalize()}')

            plt.bar(x=range(len(factor_list)), height=factor_list)
            plt.title(f'{factor} cut idx: {obvious_index} | value: {factor_list[obvious_index]}')
            plt.savefig(f'{output_dir}{img_name}_h_Obvious{factor.capitalize()}_hist.png')
            plt.close()
    
    # ============================ 5. Choose groups with most votes ==================================

    obvious_groups = []
    most_votes_group = max(group_dicts, key=lambda x: x['votes'])
    most_votes = most_votes_group['votes']
    for group in group_dicts:
        # TODO can further specify accept 2 when the loss weight is from color
        if group['votes'] >= min(2, most_votes):
            obvious_groups.append(group)
    print(f'Total Groups: {len(obvious_groups)}')

    # contours with same label
    group_cnts = [group['group_cnts'] for group in obvious_groups]
    
    # draw final result
    img = resi_input_img / 3.0    # darken the image to make the contour visible
    for cnts in group_cnts:
        img = drawer.draw_same_color(cnts, img)
    img = cv2.resize(img, (0, 0), fx=1/resize_factor, fy=1/resize_factor)
    img = np.concatenate((input_img, img), axis=1)
    drawer.save(img, 'l_FinalResult')

    spent_time = time.time() - start
    print(f'Finished in {spent_time} s')

    print('-----------------------------------------------------------')

    if evaluate:
        resize_ratio = resize_height / float(img_height)
        tp, fp, fn, pr, re, fm, er = evaluate_detection_performance(img, img_name, group_cnts,
                                                                    resize_ratio, evaluate_csv_path)
        evaluation_csv.append([img_name, tp, fp, fn, pr, re, fm, er])

    if test:
        break

if evaluate:
    f = open(evaluate_csv_path + 'evaluate-bean.csv', "wb")
    w = csv.writer(f)
    w.writerows(evaluation_csv)
    f.close()
