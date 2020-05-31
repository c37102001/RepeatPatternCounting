import cv2
import numpy as np
from utils import add_border_edge
from ipdb import set_trace as pdb


def get_contours(filter_cfg, drawer, edge_img, edge_type, do_draw=False):
    ''' Do contour detection, filter contours, feature extract and cluster.

    Args:
        edge_img: (ndarray) edge img element 0~255, sized [736, N]
        edge_type: (str) edge type name
        close_ks: (int) close kernel size, skip closing if close_ks == 0
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
    img_name = drawer.img_name
    if do_draw:
        desc = f'1_{edge_type}-0_OriginEdge'
        drawer.save(edge_img, desc)

    # do CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(9, 9))
    edge_img = clahe.apply(edge_img)
    if do_draw:
        desc = f'1_{edge_type}-1_EnhancedEdge'
        drawer.save(edge_img, desc)

    # threshold to 0 or 255
    edge_img = cv2.threshold(edge_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if do_draw:
        desc = f'1_{edge_type}-1-1_Threshold'
        drawer.save(edge_img, desc)

    # morphology close
    kernel_size = min(edge_img.shape) // 100
    # kernel_size = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
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
    
    # filter contours by area, perimeter, convex property
    height, width = drawer.color_img.shape[:2]
    contours = filter_contours(filter_cfg, contours, height, width, drawer)
    print(f'[{edge_type}] # after filtering: {len(contours)}')
    if do_draw:
        img = drawer.draw(contours)
        desc = f'1_{edge_type}-4_Filtered'
        drawer.save(img, desc)
    
    # return if too less contours
    if len(contours) <= 1:
        print(f'[Warning] {edge_type} contours less than 1.')
        return []
    
    return contours


def filter_contours(filter_cfg, contours, re_height, re_width, drawer):
    img_size = re_height * re_width
    MIN_CNT_AREA = eval(filter_cfg['min_area']) * img_size
    MAX_CNT_AREA = eval(filter_cfg['max_area']) * img_size
    MIN_AREA_OVER_PERI = eval(filter_cfg['min_area_over_peri'])
    
    accept_contours = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, closed=True)
        approx = cv2.approxPolyDP(contour, 0.05 * perimeter, True)
        if not MIN_CNT_AREA <= area <= MAX_CNT_AREA:
            continue
        if len(approx) <= 2:
            continue
        if area / perimeter < MIN_AREA_OVER_PERI:
            convex, convex_peri = get_convex_contour(contour, drawer)

            # count how many countour points are close to its convex
            dists = [abs(cv2.pointPolygonTest(convex, tuple(point[0]), True)) for point in contour]
            cnt_points_in_convex = sum([1 for dist in dists if dist <= convex_peri * 0.02])
            
            # if original contour covers more than 80% to its convex contour, replace it with convex.
            # cnt_points_in_convex is divided b 2 because it's line-shape overlap contour.
            if cnt_points_in_convex / 2 > len(convex) * 0.8:
                accept_contours.append(convex)
            
            continue
        
        # change some contours into convex
        if len(approx) <= 8:
            convex, convex_peri = get_convex_contour(contour, drawer)
            dists = [abs(cv2.pointPolygonTest(contour, tuple(point[0]), True)) for point in convex]
            convex_points_in_cnt = sum([1 for dist in dists if dist <= convex_peri * 0.01])
            if convex_points_in_cnt > len(convex) * 0.9:
                contour = convex

        accept_contours.append(contour)

    median_area = np.median([cv2.contourArea(c) for c in accept_contours])
    accept_contours = filter(lambda c: cv2.contourArea(c) > median_area * 0.6, accept_contours)
    accept_contours = [*accept_contours]
    
    return accept_contours


def get_convex_contour(contour, drawer):
    convex = cv2.convexHull(contour)
    convex_peri = cv2.arcLength(convex, closed=True)
    
    # get full convex contour points instead of only courner points.
    convex_img = drawer.draw_same_color([convex], color=(255,255,255), thickness=1)
    convex_img = cv2.cvtColor(convex_img, cv2.COLOR_BGR2GRAY)
    convex = cv2.findContours(convex_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]
    
    return convex, convex_peri
