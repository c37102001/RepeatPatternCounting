import cv2
import numpy as np
from utils import add_border_edge
from clustering import hierarchical_clustering
from extract_feature import get_contour_feature
from ipdb import set_trace as pdb

# Decide whether local/global equalization to use (True-->Local)
_gray_value_redistribution_local = True


def get_contours(drawer, edge_img, edge_type, do_enhance=True, do_draw=False):
    ''' Do contour detection, filter contours, feature extract and cluster.

    Args:
        edge_img: (ndarray) edge img element 0~255, sized [736, N]
        edge_type: (str) edge type name
        do_enhance: (bool) whether enhance edge img
        do_draw: (bool) whether draw process figures
    
    Returns:
        groups_cnt_dicts: (list of list of dict), sized [# of groups, # of cnts in this group (every cnt is a dict)]
        groups_cnt_dicts[0][0] = {
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
            desc = f'1_{edge_type}-0_OriginEdge'
            drawer.save(edge_img, desc)

    if do_enhance:
        # Enhance edge
        if _gray_value_redistribution_local:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(9, 9))
            edge_img = clahe.apply(edge_img)
        else:
            edge_img = cv2.equalizeHist(edge_img) # golbal equalization

        if do_draw:
            desc = f'1_{edge_type}-1_EnhancedEdge'
            drawer.save(edge_img, desc)

    # threshold to 0 or 255
    edge_img = cv2.threshold(edge_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if do_draw:
        desc = f'1_{edge_type}-1-1_Threshold'
        drawer.save(edge_img, desc)

    # morphology close
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    edge_img = cv2.morphologyEx(edge_img, cv2.MORPH_CLOSE, kernel)
    if do_draw:
        desc = f'1_{edge_type}-1-2_closeEdge'
        drawer.save(edge_img, desc)    
    
    # add edge on border
    edge_img = add_border_edge(edge_img)

    # find closed contours, return (list of ndarray), len = Num_of_cnts, ele = (Num_of_pixels, 1, 2(x,y))
    contours, hierarchy = cv2.findContours(edge_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    print(f'[{edge_type}] # of original contours: {len(contours)}')
    if do_draw:
        img = drawer.draw(contours)
        desc = f'1_{edge_type}-2_OriginContour'
        drawer.save(img, desc)
    
    # filter contours that has children (outer overlap contour)
    inner_contours = []
    for contour, has_child in zip(contours, hierarchy[0,:,2]):
        if has_child == -1:
            inner_contours.append(contour)
    contours = inner_contours
    print(f'[{edge_type}] # after removing overlapped: {len(contours)}')
    if do_draw:
        img = drawer.draw(contours)
        desc = f'1_{edge_type}-3_RemovedOuter'
        drawer.save(img, desc)
    
    # filter contours by area, perimeter, convex property
    height, width = drawer.color_img.shape[:2]
    contours = filter_contours(contours, height, width, drawer)
    print(f'[{edge_type}] # after filtering: {len(contours)}')
    if do_draw:
        img = drawer.draw(contours)
        desc = f'1_{edge_type}-4_Filterd'
        drawer.save(img, desc)
    
    # return if too less contours
    if len(contours) <= 1:
        print(f'[Warning] {edge_type} contours less than 1.')
        return []
    
    return contours


def filter_contours(contours, re_height, re_width, drawer):
    MIN_PERIMETER = 100
    MAX_CNT_SIZE = 1 / 5
    MIN_CNT_SIZE = 1 / 15000
    MIN_AREA_OVER_LEN = 3
    MIN_CONVEX_AREA_OVER_LEN = 10
    
    accept_contours = []
    for i, c in enumerate(contours):
        perimeter = cv2.arcLength(c, closed=True)
        contour_area = cv2.contourArea(c)
        approx = cv2.approxPolyDP(c, 0.01 * perimeter, True)
        convex = cv2.convexHull(c)
        convex_area = cv2.contourArea(convex)
        convex_peri = cv2.arcLength(convex, closed=True)
        if perimeter < MIN_PERIMETER or perimeter > (re_height + re_width) * 2 / 3.0:
            continue
        if not (re_height * re_width) * MIN_CNT_SIZE <= contour_area <= (re_height * re_width) * MAX_CNT_SIZE:
            continue
        if len(approx) <= 2:
            continue
        
        if contour_area / perimeter < MIN_AREA_OVER_LEN:
            if not 1.5 <= perimeter / convex_peri <= 2.5:
                continue
            if convex_area / convex_peri < MIN_CONVEX_AREA_OVER_LEN:
                continue

            dists = [abs(cv2.pointPolygonTest(convex, tuple(point[0]), True)) for point in c]
            cnt_points_in_convex = sum([1 for dist in dists if dist <= convex_peri * 0.01])
            in_convex_precent = cnt_points_in_convex / len(c)
            if in_convex_precent < 0.8:
                continue
            
            # use convex as new contour
            img = drawer.draw_same_color([convex], color=(255,255,255), thickness=1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            c = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]
        
        else:
            img = drawer.draw_same_color([c], color=(255,255,255), thickness=1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            c = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]

        accept_contours.append(c)
    
    median_area = np.median([cv2.contourArea(c) for c in accept_contours])
    accept_contours = filter(lambda c: cv2.contourArea(c) > median_area * 0.6, accept_contours)
    accept_contours = [*accept_contours]
    return accept_contours


def cluster_features(contours, cnt_dicts, drawer, do_draw=False):

    # Do hierarchicalclustering by shape, color, and size
    label_dict = {}
    for feature_type in ['size', 'shape', 'color']:
        feature_list = [cnt_dic[feature_type] for cnt_dic in cnt_dicts]

        # ndarray e.g. ([1, 1, 1, 1, 1, 3, 3, 2, 2, 2]), len=#feature_list
        labels = hierarchical_clustering(feature_list, feature_type, drawer, do_draw)
        print(f'{feature_type}:\t{labels}')
        label_dict[feature_type] = labels

        if do_draw:
            img = drawer.blank_img()
            for label in set(labels):
                cnt_dic_list_by_groups = [c for i, c in enumerate(contours) if labels[i] == label]
                img = drawer.draw_same_color(cnt_dic_list_by_groups, img)
            desc = f'2-1_{feature_type.capitalize()}Group'
            drawer.save(img, desc)

    # combine the label clustered by size, shape, and color. ex: (0,1,1), (2,0,1)
    combine_labels = []
    for size, shape, color in zip(label_dict['size'], label_dict['shape'], label_dict['color']):
        combine_labels.append((size, shape, color))

    # find the final group by the intersected label and draw
    img = drawer.blank_img()
    groups_cnt_dicts = []
    for combine_label in set(combine_labels):

        group_idx = [idx for idx, label in enumerate(combine_labels) if label == combine_label]
        group_cnt_dicts = [cnt_dicts[i] for i in group_idx]
        groups_cnt_dicts.append(group_cnt_dicts)

        # for do_draw
        cnts = [contours[i] for i in group_idx]
        img = drawer.draw_same_color(cnts, img)
        
    if do_draw:
        desc = f'2-2_GroupedResult'
        drawer.save(img, desc)
    
    return groups_cnt_dicts
