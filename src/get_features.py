import cv2
import numpy as np
import math
import mahotas
from utils import get_centroid, eucl_distance, scale_contour
from ipdb import set_trace as pdb
from tqdm import tqdm


def get_features(color_img, contours, drawer, do_draw, filter_by_gradient):
    ''' Extract contour color, size, shape, color_gradient features

    Args:
        color_img: (ndarray) resized colored input img, sized [736, N, 3]
        contours: (list of ndarray), len = Num_of_cnts
        contours[0].shape = (Num_of_pixels, 1, 2)
    Returns:
        cnt_dic_list = [{
            'cnt': contours[i],
            'shape': cnt_pixel_distances[i],
            'color': cnt_avg_lab[i],
            'size': cnt_norm_size[i],
            'color_gradient': cnt_color_gradient[i]
        } for i in range(len(contours))]
    '''
    
    accept_cnts = []
    cnt_pixel_distances = []
    cnt_color_gradient = []
    sample_number = 180
    all_grads = [get_cnt_color_gradient(c, color_img) for c in contours]
    accept_grads = [g for g in all_grads if g > 40]
    mean_grad = sum(accept_grads) / len(accept_grads) if len(accept_grads) else 0

    for contour, grad in tqdm(zip(contours, all_grads), desc='[Get features]', total=len(contours)):
        
        if filter_by_gradient:
            if grad < 20 or (mean_grad > 60 and grad < 40):
                continue
        cnt_color_gradient.append(grad)
        accept_cnts.append(contour)

        pixel_features = []
        cM = get_centroid(contour)

        for pixel in contour:
            pixel = pixel[0]

            vector = pixel - cM
            horizon = (0, 1)
            distance = eucl_distance(pixel, cM)
            angle = angle_between(vector, horizon)
            
            pixel_features.append({
                'coordinate': pixel,
                'distance': distance, 
                'angle': angle
            })

        max_distance = max([f['distance'] for f in pixel_features])
        for f in pixel_features:
            f['distance'] = f['distance'] / max_distance

        # find main rotate angle by fit ellipse
        ellipse = cv2.fitEllipse(contour)   # ((694.17, 662.93), (10.77, 22.17), 171.98)
        main_angle = ellipse[2]

        # rotate contour pixels to fit main angle and re-calculate pixels' angle.
        pixel_features = rotate_contour(pixel_features, main_angle)
        
        # shape feature
        pixel_distances = [f['distance'] for f in pixel_features]
        dist_sample_step = len(pixel_distances) / sample_number
        pixel_distances = [pixel_distances[math.floor(i*dist_sample_step)] for i in range(sample_number)]
        cnt_pixel_distances.append(pixel_distances)

    contours = accept_cnts
    if do_draw or filter_by_gradient:
        drawer.save(drawer.draw(contours), '2-0_FilterByGrad')
    
    cnt_size = list(map(cv2.contourArea, contours))
    max_size = max(cnt_size)
    cnt_norm_size = [[size / max_size] for size in cnt_size]

    # color feature
    cnt_avg_lab = [get_color_feature(contour, color_img) for contour in contours]

    # texture feature
    # cnt_textures = [get_texture_feature(contour, color_img) for contour in contours]

    cnt_dic_list = [{
        'cnt': contours[i],
        'shape': cnt_pixel_distances[i],
        'color': cnt_avg_lab[i],
        'size': cnt_norm_size[i],
        # 'texture': cnt_textures[i],
        'color_gradient': cnt_color_gradient[i]
    } for i in range(len(contours))]

    return contours, cnt_dic_list


def get_cnt_color_gradient(contour, im):
    im_area = im.shape[0] * im.shape[1]
    cnt_inner = scale_contour(contour, 'inner', im_area)
    cnt_outer = scale_contour(contour, 'outer', im_area)

    color_gradients = []
    im_lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
    for pt_i, pt_o in zip(cnt_inner, cnt_outer):
        i_x, i_y = pt_i[0][0], pt_i[0][1]
        o_x, o_y = pt_o[0][0], pt_o[0][1]

        if (0 <= o_x < im.shape[1]) and (0 <= o_y < im.shape[0]):
            lab_in = im_lab[i_y, i_x]   # (3)
            lab_out = im_lab[o_y, o_x]  # (3)
            gradient = np.sqrt(np.sum((lab_in - lab_out) ** 2))
            color_gradients.append(gradient)
    
    color_gradients = sorted([g for g in color_gradients if g>0])
    # avg_gradient = np.median(color_gradients) if len(color_gradients) else 0
    avg_gradient = sum(color_gradients) / len(color_gradients) if len(color_gradients) else 0
    
    return avg_gradient


def angle_between(vec1, vec2):
    '''Return angle(ranges from 0~360 degree) measured from vec2 to vec1'''
    ang1 = np.arctan2(*vec1[::-1])
    ang2 = np.arctan2(*vec2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))


def rotate_contour(pixel_features, main_angle):
    '''
    Find the nearest pixel to the long axis of the ellipse(includes angle 0 and 180),
    and shift pixels order to make the starting pixel to the begining.

    Args:
        pixel_features: (list of dict) each dict refers to features of a pixel on the contour.
        pixel_features[0] = {
            'distance': distance between pixel and contour centroid.
            'angle': angle between vector(centroid->pixel) and horizon.
            'coordinate': (ndarray), x(row) and y(column) index of pixel, sized [2, ].
        }
        main_angle: main rotate angle of the contour from the fitting ellipse.
    '''
    
    # find pixels nearest to long axis on angle 0 and 180 respectively
    pixel_on_0 = min(pixel_features, key=lambda pixel: abs(pixel['angle'] - main_angle))
    pixel_on_180 = min(pixel_features, key=lambda pixel: abs(pixel['angle'] - main_angle - 180))

    # choose the pixel with less distance to the centroid as starting pixel, record its angle and index.
    start_pixel = pixel_on_0 if pixel_on_0['distance'] < pixel_on_180['distance'] else pixel_on_180
    start_angle = start_pixel['angle']
    start_index = 0
    for i, f in enumerate(pixel_features):
        if f['distance'] == start_pixel['distance'] and \
           f['angle'] == start_pixel['angle'] and \
           np.array_equal(f['coordinate'], start_pixel['coordinate']):
           start_index = i
           break
    
    # shift pixels order to make the starting pixel to the begining.
    pixel_features = pixel_features[start_index:] + pixel_features[:start_index]

    # re-calculate the angle starting from starting point's angle.
    for pixel in pixel_features:
        pixel['angle'] -= start_angle
        if pixel['angle'] < 0:
            pixel['angle'] += 360

    return pixel_features


def get_color_feature(cnt, img):
    mask = np.zeros(img.shape[:2], np.uint8)
    # Fill the contour in order to get the inner points
    cv2.drawContours(mask, [cnt], -1, 255, -1)      # thickness=-1: the contour interiors are drawn
    cv2.drawContours(mask, [cnt], -1, 0, 1)

    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    cnt_lab = img_lab[mask == 255]      # Get contour interiors lab value, sized [#interior pixels, 3]
    avg_lab = np.mean(cnt_lab, axis=0)  # sized [3], e.g. [230.19, 125.96, 134.15]

    avg_lab[0] = avg_lab[0] * 0.5       # lower l chennel

    return avg_lab

def get_texture_feature(cnt, img):
    mask = np.zeros(img.shape[:3], np.uint8)
    # Fill the contour in order to get the inner points
    cv2.drawContours(mask, [cnt], -1, (255,255,255), -1)      # thickness=-1: the contour interiors are drawn
    cv2.drawContours(mask, [cnt], -1, (0,0,0), 1)

    x,y,w,h = cv2.boundingRect(cnt)
    cnt_img = img[y:y+h, x:x+w]
    cnt_mask = mask[y:y+h, x:x+w]

    # lower gray scale
    cnt_img = cv2.cvtColor(cnt_img, cv2.COLOR_BGR2LAB)
    cnt_img[0] = 0
    cnt_img = cv2.cvtColor(cnt_img, cv2.COLOR_LAB2BGR)
    
    cnt_img = (cnt_img * (cnt_mask / 255)).astype(np.uint8)
    cnt_img = cv2.cvtColor(cnt_img, cv2.COLOR_BGR2GRAY)

    hara = mahotas.features.haralick(cnt_img).mean(axis=0)  # (13, 4) > (13)
    
    return hara