import numpy as np
import cv2
import os
import sys
import traceback
import math
import time
import csv
from argparse import ArgumentParser
from configparser import ConfigParser
import matplotlib.pyplot as plt
from ipdb import set_trace as pdb

from canny import canny_edge_detect
from drawer import ContourDrawer
from get_contours import get_contours
from get_features import get_features
from get_clusters import get_clusters
from utils import remove_overlap, remove_outliers, count_avg_gradient, evaluate_detection_performance

parser = ArgumentParser()
parser.add_argument('--test_all', action='store_true', help='test all images in image dir')
parser.add_argument('--draw', action='store_true', help='draw all figures')
parser.add_argument('--mark', action='store_true', help='mark contour num in figure')
args = parser.parse_args()
do_draw = args.draw
do_mark = args.mark

cfg = ConfigParser()
cfg.read('config.ini')

# path config
path_cfg = cfg['path']
input_dir = path_cfg['input_dir']
output_dir = path_cfg['output_dir']
csv_output = path_cfg['csv_output']
evaluate_csv_path = path_cfg['evaluate_csv_path']

# image config
img_cfg = cfg['img_cfg']
if args.test_all:
    img_list = os.listdir(input_dir)
else:
    img_list = img_cfg['img_list'].split(',')
resize_height = eval(img_cfg['resize_height'])
use_edge = img_cfg['use_edge'].split(',')

# filter contour config
filter_cfg = cfg['filter_cfg']

# cluster config
cluster_cfg = cfg['cluster_cfg']

# obviousity thresold config
obvs_cfg = cfg['obviousity_cfg']
area_thres = eval(obvs_cfg['area_thres'])
solidity_thres = eval(obvs_cfg['solidity_thres'])
gradient_thres = eval(obvs_cfg['gradient_thres'])

# evaluate config
eval_cfg = cfg['evaluate']
evaluate = eval(eval_cfg['evaluate'])
evaluation_csv = eval_cfg['evaluation_csv'].split(',')


def main(i, img_path):
    img_path = img_path.strip()
    print(f'\n[Progress]: {i+1} / {len(img_list)}')
    start = time.time()

    #======================================== 0. Preprocess ==========================================
    
    # check format
    img_name, img_ext = img_path.rsplit('.', 1)     # ['IMG_ (33)',  'jpg']
    if img_ext not in ['jpg', 'png', 'jpeg']:
        raise FormatException(f'Format not supported: {img_path}')
        # print(f'[Error] Format not supported: {img_path}')

    print('[Input] %s' % img_path)
    input_img = cv2.imread(input_dir + img_path)
    img_height = input_img.shape[0]               # shape: (1365, 2048, 3)
    
    # resize_height=736, shape: (1365, 2048, 3) -> (736,1104,3)
    resize_factor = resize_height / img_height
    resi_input_img = cv2.resize(input_img, (0, 0), fx=resize_factor, fy=resize_factor)
    drawer = ContourDrawer(resi_input_img.copy(), output_dir, img_name, do_mark=do_mark)
    if do_draw:
        drawer.save(resi_input_img.copy(), '0_original_image')


    #================================ 1. Get and filter contours  ===================================
    
    contours = []
    for edge_type in use_edge:
        edge_type = edge_type.strip()
        edge_dir, img_extension, do_enhance = img_cfg[edge_type].split(',')
        do_enhance = eval(do_enhance)
        
        # get edge image
        if edge_type == 'Canny':
            edge_img = canny_edge_detect(resi_input_img.copy())
        else:
            edge_path = edge_dir.strip() + img_name + img_extension.strip()
            if not os.path.isfile(edge_path):
                print(f'[Error] EDGE FILE {edge_path} does not exist!')
                continue
            edge_img = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
            edge_img = cv2.resize(edge_img, (0, 0), fx=resize_factor, fy=resize_factor)     # shape: (736, *)

        # find and filter contours
        contours.extend(get_contours(filter_cfg, drawer, edge_img, edge_type, do_enhance, do_draw))
    
    # Remove overlap
    contours = remove_overlap(contours)
    print(f'[Remove overlap] # after removing overlaps: {len(contours)}')
    if do_draw or True:
        img = drawer.draw(contours)
        drawer.save(img, '1-5_RemoveOverlap')

    # # Remove outliers
    # contours = remove_outliers(contours)
    # print(f'[Remove size outliers] # after removing outliers: {len(contours)}')
    
    # =================== 2. Get contour features and cluster =========================

    # Get contour features
    cnt_dicts = get_features(resi_input_img.copy(), contours)
    
    # cluster contours into groups
    groups_cnt_dicts = get_clusters(cluster_cfg, contours, cnt_dicts, drawer, do_draw)
    
    # add label and group weight(num of cnts in the group) into contour dictionary
    for i, group_cnt_dicts in enumerate(groups_cnt_dicts):
        for group_cnt_dict in group_cnt_dicts:
            group_cnt_dict['label'] = i
            group_cnt_dict['group_weight'] = len(group_cnt_dicts)

    # flatten to a list of contour dict
    cnt_dicts = [group_cnt_dict for group_cnt_dicts in groups_cnt_dicts for group_cnt_dict in group_cnt_dicts]
    labels = [cnt_dict['label'] for cnt_dict in cnt_dicts]  # show original labels and counts
    print('[Cluster results] (label, counts): ', [(label, labels.count(label)) for label in set(labels)])
    
    # =============================== 3. Count group obviousity factors ===================================

    group_dicts = []
    for label in set(labels):
        group_cnt_dicts = [cnt_dict for cnt_dict in cnt_dicts if cnt_dict['label'] == label]
        if len(group_cnt_dicts) < 2:
            continue

        group_cnts = []
        avg_color_gradient = 0.0
        avg_solidity_factor = 0.0
        total_area = 0.0

        # count obvious factor for each group 
        for group_cnt_dict in group_cnt_dicts:
            cnt = group_cnt_dict['cnt']
            cnt_area = cv2.contourArea(cnt)
            convex_area = cv2.contourArea(cv2.convexHull(cnt))

            total_area += cnt_area
            avg_solidity_factor += cnt_area / convex_area
            avg_color_gradient += group_cnt_dict['color_gradient']
            group_cnts.append(cnt)

        avg_solidity_factor /= len(group_cnt_dicts)
        avg_color_gradient /= len(group_cnt_dicts)

        group_dicts.append({
            'group_cnts': group_cnts,
            'area': total_area,
            'color_gradient': avg_color_gradient, 
            'solidity': avg_solidity_factor,
            'votes': 0,
        })

    # ================================= 4. Obviousity voting =====================================

    factors = ['area', 'solidity', 'color_gradient']
    thres_params = [area_thres, solidity_thres, gradient_thres]
    for factor, thres_param in zip(factors, thres_params):
            
        # sorting by factor from small to large
        group_dicts.sort(key=lambda group_dict: group_dict[factor], reverse=False)
        factor_list = [group[factor] for group in group_dicts]
        
        if len(factor_list) == 1:
            obvious_index = 0
        else:
            diff = np.diff(factor_list)
            obvious_index = np.where(diff == max(diff))[0][0] + 1
        obvious_value = factor_list[obvious_index]

        thres = obvious_value * thres_param
        for i, factor_value in enumerate(factor_list[:obvious_index]):
            if factor_value > thres:
                obvious_index = i
                break

        for group in group_dicts[obvious_index:]:
            group['votes'] += 1

        if do_draw:
            img = drawer.blank_img()
            for group in group_dicts[obvious_index:]:
                img = drawer.draw_same_color(group['group_cnts'], img, color=(0, 255, 0))  # green for obvious
            for group in group_dicts[:obvious_index]:
                img = drawer.draw_same_color(group['group_cnts'], img, color=(0, 0, 255))  # red for others
            drawer.save(img, desc=f'4_Obvious_{factor}')
        
            plt.bar(x=range(len(factor_list)), height=factor_list)
            plt.title(f'{factor} cut idx: {obvious_index} | threshold: {thres: .3f}')
            plt.savefig(f'{output_dir}{img_name}_4_Obvious_{factor}.png')
            plt.close()
    
    # ============================ 5. Choose groups with most votes ==================================

    obvious_groups = []
    most_votes_group = max(group_dicts, key=lambda x: x['votes'])
    most_votes = most_votes_group['votes']
    for group in group_dicts:
        # TODO can further specify accept 2 when the loss weight is from color
        if group['votes'] == most_votes:
            obvious_groups.append(group)

    # contours with same label
    group_cnts = [group['group_cnts'] for group in obvious_groups]
    
    # draw final result
    img = resi_input_img / 3.0    # darken the image to make the contour visible
    for cnts in group_cnts:
        img = drawer.draw_same_color(cnts, img)
    img = cv2.resize(img, (0, 0), fx=1/resize_factor, fy=1/resize_factor)
    img = np.concatenate((input_img, img), axis=1)
    drawer.save(img, '5_FinalResult')

    print(f'Total Groups: {len(obvious_groups)} (cnt num: {[len(g["group_cnts"]) for g in obvious_groups]})')
    print(f'Finished in {time.time() - start} s')

    print('-----------------------------------------------------------')

    if evaluate:
        resize_ratio = resize_height / float(img_height)
        tp, fp, fn, pr, re, fm, er = evaluate_detection_performance(img, img_name, group_cnts,
                                                                    resize_ratio, evaluate_csv_path)
        evaluation_csv.append([img_name, tp, fp, fn, pr, re, fm, er])


for i, img_path in enumerate(img_list):
    try:
        main(i, img_path)
    except KeyboardInterrupt:
        break
    except Exception as e:
        error_class = e.__class__.__name__
        detail = e.args[0]
        cl, exc, tb = sys.exc_info()
        lastCallStack = traceback.extract_tb(tb)[-1]
        fileName = lastCallStack[0]
        lineNum = lastCallStack[1]
        funcName = lastCallStack[2]
        errMsg = f'File "{fileName}", line {lineNum}, in {funcName}: [{error_class}] {detail}'
        print(f'[{img_path}] {errMsg}')
        continue
