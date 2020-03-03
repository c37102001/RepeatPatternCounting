import cv2
import numpy as np
from ipdb import set_trace as pdb


# =================================================================================


def remove_outliers(contours, m=3):
    outlier_idx = []

    sizes = np.array([cv2.contourArea(c) for c in contours])
    mean = np.mean(sizes)
    std = np.std(sizes)
    
    outlier_idx = np.where((abs(sizes - mean)/ std) > m)[0].tolist()
    keep_idx = [i for i in range(len(contours)) if i not in outlier_idx]
    keep_contours = [contours[idx] for idx in keep_idx]

    return keep_contours


# =================================================================================


def remove_overlap(contours):
    # sort from min to max
    contours.sort(key=lambda x: cv2.contourArea(x), reverse=False)
    overlap_idx = []

    for i, cnt1 in tqdm(enumerate(contours[:-1]), total=len(contours[:-1]), desc='[Remove overlap]'):
        for j, cnt2 in enumerate(contours[i+1: ], start=i+1):
            if is_overlap(cnt1, cnt2):
                overlap_idx.append(j)
    
    overlap_idx = list(set(overlap_idx))
    keep_idx = [i for i in range(len(contours)) if i not in overlap_idx]
    keep_contours = [contours[idx] for idx in keep_idx]
    
    return keep_contours


# =================================================================================


def count_avg_gradient(img, model='lab'):
    # Count the average gardient of the whole image

    height, width = img.shape[:2]
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])

    if model == 'lab':
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)    # different if not convert from uint8
        lab_l = lab[:, :, 0]        # sized [736, *]
        lab_a = lab[:, :, 1]
        lab_b = lab[:, :, 2]

        gradient_list = []
        for lab_channel in [lab_l, lab_a, lab_b]:
            gradient = cv2.filter2D(lab_channel, -1, kernel)    # sized [736, *]
            gradient_list.append(gradient)                      # sized [3, 736, *]
        
        gradient_list = [g**2 for g in gradient_list]            # sized [3, 736, *]
        gradient_list = sum(gradient_list)                      # sized [736, *]
        gradient_list = np.sqrt(gradient_list)                  # sized [736, *]
        avg_gradient = np.mean(gradient_list)                   # float, e.g. 30.926353
        
    return avg_gradient


# =================================================================================


def remove_outliers(contours, m=3):
    outlier_idx = []

    sizes = np.array([cv2.contourArea(c) for c in contours])
    mean = np.mean(sizes)
    std = np.std(sizes)
    
    outlier_idx = np.where((abs(sizes - mean)/ std) > m)[0].tolist()
    keep_idx = [i for i in range(len(contours)) if i not in outlier_idx]
    keep_contours = [contours[idx] for idx in keep_idx]

    return keep_contours


# ================== extract feature part ============================


def sample_by_angle(img, pixel_features, n_sample):
    '''sample pixel by angle'''
    
    sample_angles = []
    sample_pixels = []
    angle_tolerance = 0.3
    
    # sample pixels by angle
    per_angle = 360.0 / n_sample
    for angle in np.arange(0, 360, per_angle):
        # find pixel nearest to current sample angle
        # key = lambda pixel: abs(pixel['angle'] - angle)
        key = lambda pixel: min(abs(pixel['angle'] - angle), pixel['distance'])
        pixel = min(pixel_features, key=key)

        # only sample the pixel if is less than angle_tolerance, or we'll interpolate it later
        if abs(pixel['angle'] - angle) < angle_tolerance:
            sample_angles.append(angle)
            sample_pixels.append(pixel)
    
    sample_angles.append(360.0)
    sample_pixels.append(
        {'distance': sample_pixels[0]['distance'], 'angle': 360.0, 'coordinate': pixel_features[0]['coordinate']})

    # Output the coordinate of the sample points.
    pixel_coordinates = []

    # Output the color gradient of the sample points as an obviousity .
    cnt_color_gradient = []

    # use interpolat to complete the sample angle distance
    for i in range(len(sample_angles) - 1):
        pixel_coordinates.append(sample_pixels[i]['coordinate'])
        cnt_color_gradient.append(color_gradient_by_angle(img, sample_pixels[i]['coordinate'], sample_angles[i]))
        
        for inter_angle in np.arange(sample_angles[i] + per_angle, sample_angles[i + 1], per_angle):
            inter_coordinate = corr_interpolation(
                sample_angles[i], 
                sample_pixels[i]['coordinate'], 
                sample_angles[i + 1],
                sample_pixels[i + 1]['coordinate'], 
                inter_angle)
            pixel_coordinates.append(inter_coordinate)
            
            cnt_color_gradient.append(color_gradient_by_angle(img, inter_coordinate, inter_angle))

    cnt_color_gradient = sum(cnt_color_gradient) / len(cnt_color_gradient)

    return pixel_coordinates, cnt_color_gradient


def dist_interpolation(a, a_d, b, b_d, i):
    return (abs(i - a) * b_d + abs(b - i) * a_d) / float(abs(b - a))


def corr_interpolation(a, a_d, b, b_d, i):
    return np.array([int(round(a_d[0] + (b_d[0] - a_d[0]) * (float(i - a) / (b - a)))),
                     int(round(a_d[1] + (b_d[1] - a_d[1]) * (float(i - a) / (b - a))))])


def color_gradient_by_angle(img, coordinate, angle):
    '''Calculate the color gradient of the sample points.'''

    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    img_l = img_lab[:, :, 0]
    img_a = img_lab[:, :, 1]
    img_b = img_lab[:, :, 2]

    coordinate_a = [0, 0]
    coordinate_b = [0, 0]

    '''Find two related points in the 5*5 matrix'''
    if (0 <= angle < 15) or (angle >= 345) or (165 <= angle < 195):
        coordinate_a = [coordinate[0], coordinate[1] - 2]
        coordinate_b = [coordinate[0], coordinate[1] + 2]
    elif (15 <= angle < 38) or (195 <= angle < 218):
        coordinate_a = [coordinate[0] + 1, coordinate[1] - 2]
        coordinate_b = [coordinate[0] - 1, coordinate[1] + 2]
    elif (38 <= angle < 53) or (218 <= angle < 233):
        coordinate_a = [coordinate[0] + 2, coordinate[1] - 2]
        coordinate_b = [coordinate[0] - 2, coordinate[1] + 2]
    elif (53 <= angle < 75) or (233 <= angle < 255):
        coordinate_a = [coordinate[0] + 2, coordinate[1] - 1]
        coordinate_b = [coordinate[0] - 2, coordinate[1] + 1]
    elif (75 <= angle < 105) or (255 <= angle < 285):
        coordinate_a = [coordinate[0] + 2, coordinate[1]]
        coordinate_b = [coordinate[0] - 2, coordinate[1]]
    elif (105 <= angle < 128) or (285 <= angle < 308):
        coordinate_a = [coordinate[0] + 2, coordinate[1] + 1]
        coordinate_b = [coordinate[0] - 2, coordinate[1] - 1]
    elif (128 <= angle < 143) or (308 <= angle < 323):
        coordinate_a = [coordinate[0] + 2, coordinate[1] + 2]
        coordinate_b = [coordinate[0] - 2, coordinate[1] - 2]
    elif (143 <= angle < 165) or (323 <= angle < 345):
        coordinate_a = [coordinate[0] + 1, coordinate[1] + 2]
        coordinate_b = [coordinate[0] - 1, coordinate[1] - 2]

    '''Prevent the contour is near the boundary'''
    height, width = img.shape[:2]
    coordinate_a[0] = max(coordinate_a[0], 0)
    coordinate_a[1] = max(coordinate_a[1], 0)
    coordinate_b[0] = max(coordinate_b[0], 0)
    coordinate_b[1] = max(coordinate_b[1], 0)
    coordinate_a[0] = min(coordinate_a[0], width - 1)
    coordinate_a[1] = min(coordinate_a[1], height - 1)
    coordinate_b[0] = min(coordinate_b[0], width - 1)
    coordinate_b[1] = min(coordinate_b[1], height - 1)

    # (x,y) x for height, y for width
    '''Take the color gradient value of the related points'''
    point_a_l_value = float(img_l[coordinate_a[1], coordinate_a[0]])
    point_a_a_value = float(img_a[coordinate_a[1], coordinate_a[0]])
    point_a_b_value = float(img_b[coordinate_a[1], coordinate_a[0]])
    point_b_l_value = float(img_l[coordinate_b[1], coordinate_b[0]])
    point_b_a_value = float(img_a[coordinate_b[1], coordinate_b[0]])
    point_b_b_value = float(img_b[coordinate_b[1], coordinate_b[0]])

    return math.sqrt(
        pow(abs(point_a_l_value - point_b_l_value), 2) + pow(abs(point_a_a_value - point_b_a_value), 2) + pow(
            abs(point_a_b_value - point_b_b_value), 2))


# ================== color gradient testing ============================


im = cv2.imread('sample_img.png')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(imgray, 254, 255, cv2.THRESH_BINARY)[1]
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

def scale_contour(cnts, type, im_area):
    cnts_scaled = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area / im_area >= 0.01:
            scale = 0.95 if type == 'inner' else 1.05
        elif area / im_area >= 0.001:
            scale = 0.9 if type == 'inner' else 1.1
        else:
            scale = 0.85 if type == 'inner' else 1.17
        
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        cnt_norm = cnt - [cx, cy]
        cnt_scaled = cnt_norm * scale
        cnt_scaled = cnt_scaled + [cx, cy]
        cnt_scaled = cnt_scaled.astype(np.int32)
        
        cnts_scaled.append(cnt_scaled)
    
    return cnts_scaled


cnt_inners = scale_contour(contours, 'inner', im.shape[0] * im.shape[1])
cnt_outers = scale_contour(contours, 'outer', im.shape[0] * im.shape[1])

im_copy = im.copy()
for contour, cnt_inner, cnt_outer in zip(contours, cnt_inners, cnt_outers):
    cv2.drawContours(im_copy, [contour], 0, (0, 0, 0), 1)
    cv2.drawContours(im_copy, [cnt_inner], 0, (255, 255, 255), 1)
    cv2.drawContours(im_copy, [cnt_outer], 0, (255, 255, 255), 1)
cv2.imwrite('result.png', im_copy)

avg_color_gradients = []
for cnt_inner, cnt_outer in zip(cnt_inners, cnt_outers):
    color_gradients = []
    im_lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    for pt_i, pt_o in zip(cnt_inner, cnt_outer):
        i_x, i_y = pt_i[0][0], pt_i[0][1]
        o_x, o_y = pt_o[0][0], pt_o[0][1]

        if (0 <= o_x <= im.shape[1]) and (0 <= o_y <= im.shape[0]):
            lab_in = im_lab[i_y, i_x]   # (3)
            lab_out = im_lab[o_y, o_x]  # (3)
            gradient = np.sqrt(np.sum((lab_in - lab_out) ** 2))
            color_gradients.append(gradient)
    color_gradients = sorted([g for g in color_gradients if g>0])
    avg_gradient = np.median(color_gradients) if len(color_gradients) else 0
    # avg_gradient = sum(color_gradients) / len(color_gradients) if len(color_gradients) else 0
    avg_color_gradients.append(avg_gradient)

im_copy = im.copy()
for index, (c, g) in enumerate(zip(contours, avg_color_gradients)):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.01 * peri, True)
    (x, y, w, h) = cv2.boundingRect(approx)
    img = cv2.putText(im_copy, str(g), (int(x+(w/2)), int(y+(h/2))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 4, 4), 2)
cv2.imwrite('result2.png', im_copy)


# =================================================================================


def remove_outliers(contours, m=3):
    outlier_idx = []

    sizes = np.array([cv2.contourArea(c) for c in contours])
    mean = np.mean(sizes)
    std = np.std(sizes)
    
    outlier_idx = np.where((abs(sizes - mean)/ std) > m)[0].tolist()
    keep_idx = [i for i in range(len(contours)) if i not in outlier_idx]
    keep_contours = [contours[idx] for idx in keep_idx]

    return keep_contours



