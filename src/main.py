import numpy as np
import cv2
import os
import sys
import traceback
import math
import time
import shutil
import csv
from argparse import ArgumentParser
from configparser import ConfigParser
import matplotlib.pyplot as plt
from ipdb import set_trace as pdb

from edge_detection import canny_edge_detect, sobel_edge_detect
from drawer import ContourDrawer
from get_contours import get_contours
from get_features import get_features
from get_clusters import get_clusters
from utils import remove_group_overlap, filter_small_group, evaluate_detection_performance
from RCF.run_rcf import make_single_rcf
from HED.run_hed import make_single_hed
from SF.run_sf import make_single_sf

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
if eval(path_cfg['clear_output_dir']):
    shutil.rmtree(output_dir)
    os.makedirs(output_dir)

# architecture config
arch_cfg = cfg['arch']
do_second_clus = eval(arch_cfg['do_second_clus'])

# image config
img_cfg = cfg['img_cfg']
if args.test_all:
    img_list = os.listdir(input_dir)
else:
    img_list = img_cfg['img_list'].split(',')
    if '-' in img_list[0]:
        img_from, img_end = img_list[0].split('-')
        img_list = [f'IMG_ ({i}).jpg' for i in range(eval(img_from), eval(img_end) + 1)]
resize_height = eval(img_cfg['resize_height'])
use_edge = img_cfg['use_edge'].split(',')

# filter contour config
filter_cfg = cfg['filter_cfg']

# cluster config
cluster_cfg = cfg['cluster_cfg']
cluster2_cfg = cfg['cluster2_cfg']

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
    if img_ext not in ['jpg', 'png', 'jpeg', 'JPG']:
        raise FormatException(f'Format not supported: {img_path}')

    print('[Input] %s' % img_path)
    if not os.path.isfile(input_dir + img_path):
        raise IOError(f'No such image: {img_path}')
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

        elif edge_type == 'Sobel':
            edge_img = sobel_edge_detect(resi_input_img.copy())

        else:
            edge_path = edge_dir.strip() + img_name + img_extension.strip()
            if not os.path.isfile(edge_path):
                if edge_type == 'RCF':
                    print('[Input] Making RCF edge image...')
                    make_single_rcf(img_path, input_dir, edge_dir)
                if edge_type == 'HED':
                    print('[Input] Making HED edge image...')
                    make_single_hed(img_path, input_dir, edge_dir)
                if edge_type == 'Structure':
                    print('[Input] Making SF edge image...')
                    make_single_sf(img_path, input_dir, edge_dir)
            edge_img = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
            edge_img = cv2.resize(edge_img, (0, 0), fx=resize_factor, fy=resize_factor)     # shape: (736, *)

        # find and filter contours
        contours.extend(get_contours(filter_cfg, drawer, edge_img, edge_type, do_enhance, do_draw))
    
    if do_draw or True:
        drawer.save(drawer.draw(contours), '2_CombineCnts')

    # =================== 2. Get contour features, cluster and remove overlap =========================
    

    # Get contour features
    contours, cnt_dicts = get_features(resi_input_img.copy(), contours, drawer, do_draw)
    
    # cluster contours into groups
    cnt_dicts, labels = get_clusters(cluster_cfg, contours, cnt_dicts, drawer, do_draw)
    
    # Remove overlap
    cnt_dicts, labels = remove_group_overlap(cnt_dicts, labels, drawer, do_draw)
    
    # do second clustering
    if do_second_clus:
        contours = [cnt_dict['cnt'] for cnt_dict in cnt_dicts]
        cnt_dicts, labels = get_clusters(cluster2_cfg, contours, cnt_dicts, drawer, do_draw, second=True)

    # filter group with too less contours
    cnt_dicts, labels = filter_small_group(cnt_dicts, labels, drawer, do_draw)


    # =============================== 3. Count group obviousity factors ===================================


    group_dicts = []
    cnt_grads = []  # for drawing color gradients
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

            # for drawing color gradients
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            (x, y, w, h) = cv2.boundingRect(approx)
            center = (int(x+(w/2)), int(y+(h/2)))
            cnt_grads.append((group_cnt_dict['color_gradient'], center))

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


    # factors = ['area', 'solidity', 'color_gradient']
    factors = ['solidity', 'color_gradient']
    # thres_params = [area_thres, solidity_thres, gradient_thres]
    thres_params = [solidity_thres, gradient_thres]
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

        if factor == 'color_gradient':
            thres = max(thres, 90) if factor_list[-1] > 100 else min(thres, 90)
        # thres = min(thres, 90) if factor == 'color_gradient' else thres # TODO
        for i, factor_value in enumerate(factor_list[:obvious_index]):
            if factor_value > thres:
                obvious_index = i
                break

        for group in group_dicts[obvious_index:]:
            group['votes'] += 1

        if do_draw or True:
            img = drawer.blank_img()
            for group in group_dicts[obvious_index:]:
                img = drawer.draw_same_color(group['group_cnts'], img, color=(0, 255, 0))  # green for obvious
            for group in group_dicts[:obvious_index]:
                img = drawer.draw_same_color(group['group_cnts'], img, color=(0, 0, 255))  # red for others

            if factor == 'color_gradient':
                for grad, center in cnt_grads:
                    img = cv2.putText(img, f'{int(grad)}', center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
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
        if group['votes'] >= most_votes:
            obvious_groups.append(group)
    
    # contours with same label
    group_cnts = [group['group_cnts'] for group in obvious_groups]
    print(f'Total Groups: {len(obvious_groups)} (cnt num: {[len(g["group_cnts"]) for g in obvious_groups]})')
    
    # draw final result
    img = resi_input_img / 3.0    # darken the image to make the contour visible
    for cnts in group_cnts:
        img = drawer.draw_same_color(cnts, img)
    img = cv2.resize(img, (0, 0), fx=1/resize_factor, fy=1/resize_factor)
    img = np.concatenate((input_img, img), axis=1)
    drawer.save(img, '5_FinalResult')

    print(f'Finished in {time.time() - start} s')

    print('-----------------------------------------------------------')

    if evaluate:
        resize_ratio = resize_height / float(img_height)
        tp, fp, fn, pr, re, fm, er = evaluate_detection_performance(img, img_name, group_cnts,
                                                                    resize_ratio, evaluate_csv_path)
        evaluation_csv.append([img_name, tp, fp, fn, pr, re, fm, er])


total_start = time.time()
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
print(f'Total finished in {time.time() - total_start} s')