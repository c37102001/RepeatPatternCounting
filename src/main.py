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
from utils import remove_group_overlap, filter_small_group, evaluate_detection_performance, do_CLAHE
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
source_path = os.path.join(path_cfg['source_path'], path_cfg['data_name'])
img_path = os.path.join(source_path, path_cfg['img_dir'])
edge_path = os.path.join(source_path, path_cfg['edge_dir'])
if not os.path.exists(img_path):
    raise IOError(f'Image Path "{img_path}" Not Found')

# output image path
output_path = os.path.join(path_cfg['output_path'], path_cfg['data_name'])
if not os.path.exists(output_path):
    os.makedirs(output_path)
elif eval(path_cfg['clear_results']):
    shutil.rmtree(output_path)
    os.makedirs(output_path)

# evaluation path
cluster_csv_path = os.path.join(source_path, "cluster_csv")
if not os.path.exists(cluster_csv_path):
    os.makedirs(cluster_csv_path)
eval_csv_path = os.path.join(source_path, "eval_csv")
output1_csv_path = os.path.join(eval_csv_path, 'eval_1.csv')
with open(output1_csv_path, 'w+') as f:
    f.write('Image name,TP,FP,FN,Precision,Recall,F_measure,Error_rate,Time\n')
output2_csv_path = os.path.join(eval_csv_path, 'eval_2.csv')
with open(output2_csv_path, 'w+') as f:
    f.write('Image name,TP,FP,FN,Precision,Recall,F_measure,Error_rate,Time\n')

# architecture config
arch_cfg = cfg['arch']
do_second_clus = eval(arch_cfg['do_second_clus'])
filter_by_gradient = eval(arch_cfg['filter_by_gradient'])
do_eval = eval(arch_cfg['do_eval'])
amb_idx = eval(arch_cfg['amb_idx'])

# image config
img_cfg = cfg['img_cfg']
if args.test_all:
    img_list = os.listdir(img_path)
else:
    img_list = img_cfg['img_list'].split(',')
    if '-' in img_list[0]:
        img_from, img_end = img_list[0].split('-')
        # img_list = [f'IMG_ ({i}).jpg' for i in range(eval(img_from), eval(img_end) + 1)]
        img_list = [f'img ({i}).jpg' for i in range(eval(img_from), eval(img_end) + 1)]
resize_height = eval(img_cfg['resize_height'])
file_ext = img_cfg['edge_file_extension']
use_edge = img_cfg['use_edge'].split(',')

# filter contour config
filter_cfg = cfg['filter_cfg']

# cluster config
cluster_cfg = cfg['cluster_cfg']
cluster2_cfg = cfg['cluster2_cfg']

# obviousity thresold config
obvs_cfg = cfg['obviousity_cfg']
area_thres = eval(obvs_cfg['area_thres'])
gradient_thres = eval(obvs_cfg['gradient_thres'])
solidity_thres = eval(obvs_cfg['solidity_thres'])



def main(img_file):         # img_file = 'IMG_ (33).jpg'
    
    #======================================== 0. Preprocess ==========================================
    
    start = time.time()
    img_name, img_ext = img_file.rsplit('.', 1)     # ['IMG_ (33)',  'jpg']
    input_img = cv2.imread(img_path + img_file)
    # input_img = do_CLAHE(input_img)
    
    img_height = input_img.shape[0]               # shape: (1365, 2048, 3)
    resize_factor = resize_height / img_height  # resize_height=736, shape: (1365, 2048, 3) -> (736,1104,3)
    resi_input_img = cv2.resize(input_img, (0, 0), fx=resize_factor, fy=resize_factor)
    
    drawer = ContourDrawer(resi_input_img.copy(), output_path, img_name, do_mark=do_mark)
    if do_draw:
        drawer.save(resi_input_img.copy(), '0_original_image')


    #================================ 1. Get and filter contours  ===================================

    # i want to do_enhance
    contours = []
    for edge_type in use_edge:
        edge_type = edge_type.strip()
        
        # get edge image
        if edge_type == 'Canny':
            edge_img = canny_edge_detect(resi_input_img.copy())

        elif edge_type == 'Sobel':
            edge_img = sobel_edge_detect(resi_input_img.copy())

        else:
            edge_folder_path = os.path.join(edge_path, edge_type)
            edge_img_file = img_name + f'_{edge_type.lower()}' + file_ext
            edge_img_path = os.path.join(edge_folder_path, edge_img_file)
            
            if not os.path.isfile(edge_img_path):
                if not os.path.exists(edge_folder_path):
                    os.makedirs(edge_folder_path)
                if edge_type == 'RCF':
                    print('[Input] Making RCF edge image...')
                    make_single_rcf(input_img, edge_img_path)
                if edge_type == 'HED':
                    print('[Input] Making HED edge image...')
                    make_single_hed(input_img, edge_img_path)
                if edge_type == 'SF':
                    print('[Input] Making SF edge image...')
                    make_single_sf(input_img, edge_img_path)

            edge_img = cv2.imread(edge_img_path, cv2.IMREAD_GRAYSCALE)
            edge_img = cv2.resize(edge_img, (0, 0), fx=resize_factor, fy=resize_factor)     # shape: (736, *)

            # sobel = sobel_edge_detect(resi_input_img.copy())
            # edge_img = (0.5 * sobel + 0.5 * edge_img).astype(np.uint8)

            # canny = edge_img = canny_edge_detect(resi_input_img.copy())
            # sobel = sobel_edge_detect(resi_input_img.copy())
            # edge_img = (0.2 * canny + 0.4 * sobel + 0.4 * edge_img).astype(np.uint8)
            

        # Find and filter contours
        contours.extend(get_contours(filter_cfg, drawer, edge_img, edge_type, do_draw))

    if do_draw:
        drawer.save(drawer.draw(contours), '2_CombineCnts')
    # return
    # =================== 2. Get contour features, cluster and remove overlap =========================
    

    # Get contour features
    contours, cnt_dicts = get_features(resi_input_img.copy(), contours, drawer, do_draw, filter_by_gradient)
    
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


    factors = ['area', 'solidity', 'color_gradient']
    thres_params = [area_thres, solidity_thres, gradient_thres]
    for factor, thres_param in zip(factors, thres_params):
            
        # sorting by factor from small to large
        group_dicts.sort(key=lambda group_dict: group_dict[factor], reverse=False)
        factor_list = [group[factor] for group in group_dicts]
        
        obvious_index = len(factor_list) - 1
        obvious_value = factor_list[obvious_index]
        thres = obvious_value * thres_param

        if factor == 'color_gradient':
            if obvious_value < 40:
                thres = 0
            elif thres > 100:
                thres = 100
        
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
            plt.savefig(os.path.join(output_path, f'{img_name}_4_Obvious_{factor}.png'))
            plt.close()

    
    # ============================ 5. Choose groups with most votes ==================================


    most_votes_group = max(group_dicts, key=lambda x: x['votes'])
    most_votes = most_votes_group['votes']
    obvious_groups = [g for g in group_dicts if g['votes'] == most_votes]
    obvious_group_cnts = [group['group_cnts'] for group in obvious_groups]     # contours with same label
    print(f'Priority1 Groups: {len(obvious_groups)} (cnt num: {[len(g["group_cnts"]) for g in obvious_groups]})')
    
    # draw final result
    img = resi_input_img / 3.0    # darken the image to make the contour visible
    for cnts in obvious_group_cnts:
        img = drawer.draw_same_color(cnts, img)
    img = cv2.resize(img, (0, 0), fx=1/resize_factor, fy=1/resize_factor)
    img = np.concatenate((input_img, img), axis=1)
    drawer.save(img, '5_FinalResult1')
    spent = time.time() - start

    if do_eval:
        tp, fp, fn, pr, re, fm, er, _ = evaluate_detection_performance(
        resi_input_img.copy(), img_file, obvious_group_cnts, resize_factor, eval_csv_path, amb_idx)
    
        with open(output1_csv_path, 'a+') as f:
            writer = csv.writer(f)
            writer.writerow([img_file, tp, fp, fn, pr, re, fm, er, spent])


    # draw all contours result
    second_obvious_groups = [g for g in group_dicts if g['votes'] >= most_votes-1]
    sencond_group_cnts = [group['group_cnts'] for group in second_obvious_groups]
    print(f'Priority2 Groups: {len(second_obvious_groups)} (cnt num: {[len(g["group_cnts"]) for g in second_obvious_groups]})')
    
    img = resi_input_img / 3.0    # darken the image to make the contour visible
    for cnts in sencond_group_cnts:
        img = drawer.draw_same_color(cnts, img)
    img = cv2.resize(img, (0, 0), fx=1/resize_factor, fy=1/resize_factor)
    img = np.concatenate((input_img, img), axis=1)
    drawer.save(img, '5_FinalResult2')

    if do_eval:
        tp, fp, fn, pr, re, fm, er, group_results = evaluate_detection_performance(
            resi_input_img.copy(), img_file, sencond_group_cnts, resize_factor, eval_csv_path, amb_idx)
        
        with open(output2_csv_path, 'a+') as f:
            writer = csv.writer(f)
            writer.writerow([img_file, tp, fp, fn, pr, re, fm, er, spent])

        with open(os.path.join(cluster_csv_path, img_file + '.csv'), 'w+') as f:
            writer = csv.writer(f)
            writer.writerows(group_results)
            

    print(f'Finished in {spent} s')
    print('-----------------------------------------------------------')


total_start = time.time()
for i, img_file in enumerate(img_list):
    print(f'\n[Progress]: {i+1} / {len(img_list)}')
    img_file = img_file.strip()

    if os.path.isfile(os.path.join(img_path, img_file)):
        print(f'[Input] {os.path.join(img_path, img_file)}')
    else:
        print(f'No such image: {os.path.join(img_path, img_file)}')
        continue
    
    try:
        main(img_file)
        
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
        print(f'[{img_file}] {errMsg}')
        continue
print(f'Total finished in {time.time() - total_start} s')