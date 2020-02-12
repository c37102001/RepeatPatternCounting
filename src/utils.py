import cv2
import numpy as np
from misc import remove_overlap
from clustering import hierarchical_clustering
from extract_feature import get_contour_feature
from ipdb import set_trace as pdb

# Decide whether local/global equalization to use (True-->Local)
_gray_value_redistribution_local = True


def get_edge_group(drawer, edge_img, edge_type, keep='inner', do_enhance=True, do_draw=False):
    ''' Do contour detection, filter contours, feature extract and cluster.

    Args:
        edge_img: (ndarray) edge img element 0~255, sized [736, N]
        edge_type: (str) edge type name
        do_enhance: (bool) whether enhance edge img
        do_draw: (bool) whether draw process figures
    
    Returns:
        grouped_cnt: (list of list of dict), sized [# of groups, # of cnts in this group (every cnt is a dict)]
        grouped_cnt[0][0] = {
            'cnt': contours[i],
            'shape': cnt_pixel_distances[i],
            'color': cnt_avg_lab[i],
            'size': cnt_norm_size[i],
            'color_gradient': cnt_color_gradient[i]
        }
    '''
    output_path = drawer.output_path
    img_name = drawer.img_name

    if do_draw:
        img_path = '{}{}_b_OriginEdge{}.jpg'.format(output_path, img_name, edge_type)
        cv2.imwrite(img_path, edge_img)

    if do_enhance:  
        # Enhance edge
        if _gray_value_redistribution_local:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            edge_img = clahe.apply(edge_img)
        else:
            edge_img = cv2.equalizeHist(edge_img) # golbal equalization

        if do_draw:
            cv2.imwrite(output_path + img_name + '_c_enhanced_edge[' + str(edge_type) + '].jpg', edge_img)

    '''find contours'''
    # threshold to 0 or 255
    edge_img = cv2.threshold(edge_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # find closed contours, return (list of ndarray), len = Num_of_cnts, ele = (Num_of_pixels, 1, 2(x,y))
    contours = cv2.findContours(edge_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[-2]

    if do_draw:
        img = drawer.draw(contours)
        desc = 'd_OriginContour_{}'.format(edge_type)
        drawer.save(img, desc)
    
    # filter contours by area, perimeter, solidity, edge_num
    height, width = drawer.color_img.shape[:2]
    contours = filter_contours(contours, height, width)
    if do_draw:
        img = drawer.draw(contours)
        desc = 'e1_Filterd_{}'.format(edge_type)
        drawer.save(img, desc)
    
    # remove outer overlap contour
    if not keep == 'all':
        contours, discard_contours = remove_overlap(contours, 'inner')
        if do_draw:
            img = drawer.draw_same_color(contours)
            img = drawer.draw_same_color(discard_contours, img)
            desc = 'e2_Overlapped_{}'.format(edge_type)
            drawer.save(img, desc)

            img = drawer.draw_same_color(contours)
            desc = 'e2_RemovedOverlapped_{}'.format(edge_type)
            drawer.save(img, desc)
    
    # return if too less contours
    if len(contours) <= 3:
        print(f'[Warning] {edge_type} contours less than 3.')
        return []

    # Extract contour color, size, shape, color_gradient features
    cnt_feature_dic_list, feature_dic = get_contour_feature(drawer.color_img, contours, edge_type)
    
    # cluster feature into groups
    grouped_cnt = cluster_features(contours, cnt_feature_dic_list, feature_dic, drawer, edge_type, do_draw)
    
    return grouped_cnt


def filter_contours(contours, re_height, re_width):
    MIN_PERIMETER = 60
    MAX_CNT_SIZE = 1 / 5
    MIN_CNT_SIZE = 1 / 30000
    SOLIDITY_THRE = 0.5
    MAX_EDGE_NUM = 50
    
    accept_contours = []
    for c in contours:
        perimeter = len(c)
        if perimeter < MIN_PERIMETER or perimeter > (re_height + re_width) * 2 / 3.0:
            continue
        
        contour_area = cv2.contourArea(c)
        if contour_area < (re_height * re_width) * MIN_CNT_SIZE or contour_area > (re_height * re_width) * MAX_CNT_SIZE:
            continue

        convex_area = cv2.contourArea(cv2.convexHull(c))
        solidity = contour_area / convex_area
        if solidity < SOLIDITY_THRE:
            continue

        epsilon = 0.01 * cv2.arcLength(c, closed=True)
        edge_num = len(cv2.approxPolyDP(c, epsilon, closed=True))
        if edge_num > MAX_EDGE_NUM:
            continue

        accept_contours.append(c)

    return accept_contours


def cluster_features(contours, cnt_feature_dic_list, feature_dic, drawer, edge_type, do_draw=False):

    feature_label = {}
    # Respectively use shape, color, and size as feature set to cluster
    for feature_type in ['size', 'shape', 'color']:
        feature_list = feature_dic[feature_type]

        # label_list = ndarray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2]), len=len(ft_list)
        label_list = hierarchical_clustering(feature_list, feature_type, edge_type, drawer, do_draw)

        if do_draw:
            img = drawer.blank_img()
            for label in set(label_list):
                grouped_cnts = [c for i, c in enumerate(contours) if label_list[i] == label]
                img = drawer.draw_same_color(grouped_cnts, img)
            desc = 'f_Feature{}_{}'.format(feature_type.capitalize(), edge_type)
            drawer.save(img, desc)

        feature_label[feature_type] = label_list


    # combine the label clustered by size, shape, and color. ex: (0,1,1), (2,0,1)
    combine_labels = []
    for size, shape, color in zip(feature_label['size'], feature_label['shape'], feature_label['color']):
        combine_labels.append((size, shape, color))

    # find the final group by the intersected label and draw
    img = drawer.blank_img()
    final_group = []
    for combine_label in set(combine_labels):
        if combine_labels.count(combine_label) < 2:
            continue

        label_idx = [idx for idx, label in enumerate(combine_labels) if label == combine_label]
        grouped_ft_dict_list = [cnt_feature_dic_list[i] for i in label_idx]
        final_group.append(grouped_ft_dict_list)

        # for do_draw
        cnts = [contours[i] for i in label_idx]
        img = drawer.draw_same_color(cnts, img)
        
    if do_draw:
        desc = 'g_OriginalResult_{}'.format(edge_type)
        drawer.save(img, desc)
    
    return final_group
