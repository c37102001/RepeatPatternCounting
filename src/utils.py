import cv2
import numpy as np
from misc import remove_overlap
from clustering import hierarchical_clustering
from extract_feature import get_contour_feature
from ipdb import set_trace as pdb

# Decide whether local/global equalization to use (True-->Local)
_gray_value_redistribution_local = True

def get_edge_group(drawer, edge_img, edge_type, keep, do_enhance=True, do_draw=False):
    ''' Do contour detection, filter contours, feature extract and cluster.

    Args:
        edge_img: (ndarray) edge img element 0~255, sized [736, N]
        edge_type: (str) edge type name
        do_enhance: (bool) whether enhance edge img
        do_draw: (bool) whether draw process figures
    
    Returns:
        grouped_cnt: (list of dict)
        grouped_cnt[0] = {
            'cnt': (list of ndarray) sized [num_of_cnts, num_of_pixels, 1, 2]
            'obvious_weight': (int) (e.g. 0)
            'group_dic': (list of dict)
        }
        grouped_cnt[0]['group_dic'] = {
            'cnt': (ndarray) sized [num_of_pixels, 1, 2]
            'shape': (list of float normalized to 0~1) len = shape_sample_num(90/180/360)
            'color': (list of float) (e.g. [45.83, 129.31, 133.44]) len = 3?
            'size': (list of float) (e.g. [0.07953])
            'color_gradient': (float) (e.g. 64.5149)
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
        desc = 'd_OriginContour_{}'.format(edge_type)
        lasy_do_draw(drawer, contours, desc)

    # filter contours by area, perimeter, solidity, edge_num
    height, width = drawer.color_img.shape[:2]
    contours = filter_contours(contours, height, width)
    if do_draw or True:
        desc = '_e1_Filterd_{}'.format(edge_type)
        lasy_do_draw(drawer, contours, desc)

    # remove outer overlap contour
    if not keep == 'all':
        contours = remove_overlap(contours, keep)
        if do_draw or True:
            desc = '_e2_RemoveOverlap_{}'.format(edge_type)
            lasy_do_draw(drawer, contours, desc)

    # Extract contour color, size, shape, color_gradient features
    cnt_feature_dic_list, feature_dic = get_contour_feature(drawer.color_img, contours, edge_type)
    
    # cluster feature into groups
    grouped_cnt = cluster_features(contours, cnt_feature_dic_list, feature_dic, drawer, edge_type, do_draw)
    
    return grouped_cnt


def cluster_features(contours, cnt_feature_dic_list, feature_dic, drawer, edge_type, do_draw=False):

    label_list_dic = {}
    # Respectively use shape, color, and size as feature set to cluster
    for feature_type in ['size', 'shape', 'color']:
        print(f'[{edge_type}] {feature_type}')

        feature_list = feature_dic[feature_type]

        # hierarchical clustering, output the classified consequence
        label_list = hierarchical_clustering(feature_list, drawer.img_name, feature_type, edge_type, do_draw=do_draw)
        # pdb()

        unique_label, label_counts = np.unique(label_list, return_counts=True) 
        # array([1, 2]), array([ 66, 101])

        if do_draw:
            drawer.reset()
            for label in unique_label:
                tmp_splited_group = []
                for i in range(len(label_list)):
                    if label_list[i] == label:
                        tmp_splited_group.append(contours[i])
                
                drawer.draw(np.array(tmp_splited_group))

            desc = 'f_Feature{}_{}'.format(feature_type.capitalize(), edge_type)
            drawer.save(desc)
        
        # save the 3 types of the classified output
        label_list_dic[feature_type] = label_list


    # combine the label clustered by size, shape, and color. ex: [0_1_1 , 2_0_1]
    combine_label_list = []
    for size, shape, color in zip(label_list_dic['size'], label_list_dic['shape'], label_list_dic['color']):
        combine_label_list.append('%d_%d_%d' % (size, shape, color))

    unique_label, label_counts = np.unique(combine_label_list, return_counts=True)

    # find the final group by the intersected label and draw
    drawer.reset()
    final_group = []
    color_index = 0
    for label in unique_label:
        tmp_group = []
        for i in range(len(contours)):
            if combine_label_list[i] == label:
                tmp_group.append(cnt_feature_dic_list[i])

        tmp_cnt_group = [cnt_dic['cnt'] for cnt_dic in tmp_group]

        if len(tmp_cnt_group) < 2:
            continue

        drawer.draw(np.array(tmp_cnt_group))
        final_group.append({'cnt': contours, 'obvious_weight': 0, 'group_dic': tmp_group})

    if do_draw:
        desc = 'g_OriginalResult_{}'.format(edge_type)
        drawer.save(desc)

    return final_group


def filter_contours(contours, re_height, re_width):
    MIN_PERIMETER = 60
    MAX_IMG_SIZE = 5
    MIN_IMG_SIZE = 30000
    SOLIDITY_THRE = 0.5
    MAX_EDGE_NUM = 50
    
    accept_contours = []
    for c in contours:
        perimeter = len(c)
        if perimeter < MIN_PERIMETER or perimeter > (re_height + re_width) * 2 / 3.0:
            continue
        
        contour_area = cv2.contourArea(c)
        if contour_area < (re_height * re_width) / MIN_IMG_SIZE or contour_area > (re_height * re_width) / MAX_IMG_SIZE:
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


def lasy_do_draw(drawer, contours, desc):
    drawer.reset()
    for contour in contours:
        drawer.draw([contour])
    drawer.save(desc)
