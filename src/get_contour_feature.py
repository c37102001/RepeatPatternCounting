'''
Goal :
Extract features for clustering from filtered contours.
The features contains three clustering features (color, size, shape) and one obviousity feature.(color gradient)

'''

import cv2
import numpy as np
import math
import ipdb
from utils import get_centroid, eucl_distance

GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
ORANGE = (0, 128, 255)
YELLOW = (0, 255, 255)
LIGHT_BLUE = (255, 255, 0)
PURPLE = (205, 0, 205)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def extract_feature(image, contours):
    height, width, channel = image.shape

    assert len(contours) > 0, print('No any contour')

    '''
    record the distance between pixels and the centroid
    the number of sample distance depend on the dimension of the contour
    '''
    cnt_sample_distance_list = []

    ''' record the color gradient of the contour '''
    cnt_color_gradient_list = []

    '''
    several probable dimension of contour shape
    If pixel s of the contour is between 4-8 , then we take 4 as its dimension.
    '''
    factor_360 = [4, 8, 20, 40, 90, 180, 360]

    most_cnt_len = len(contours[int(len(contours) * 0.8)])
    sample_number = 360
    min_value = 1000
    for factor in factor_360:
        differ = abs(most_cnt_len - factor)
        if differ < min_value:
            min_value = differ
            sample_number = factor

    for contour in contours:
        feature_list = []
        max_distance = 0
        cx, cy = get_centroid(contour)
        for pixel in contour:
            pixel = pixel[0]
            px, py = pixel[0], pixel[1]
            # the (0,0) in image is the left top point
            v1 = (py - cy, cx - px)
            v2 = (0, height)

            if v1[0] == 0 and v1[1] >= 0:
                angle = 0.0
            elif v1[0] == 0 and v1[1] < 0:
                angle = 180.0
            else:
                angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                angle = angle * 180 / np.pi

            if v1[0] < 0:
                angle = 360 - angle

            distance = eucl_distance(pixel, (cx, cy))
            feature_list.append({'distance': distance, 'angle': angle, 'coordinate': pixel})
            max_distance = max(distance, max_distance)
        for f in feature_list:
            f['distance'] = f['distance'] / max_distance

        '''Ellipse fitting'''
        ellipse = cv2.fitEllipse(contour)
        # ((460.30, 714.03), (8.67, 21.88), 50.22039794921875) long, short axis and rotate angle
        feature_list = rotate_contour(feature_list, ellipse[2])
        sample_distance_list, sample_coordinate_list, cnt_color_gradient = sample_by_angle(image, feature_list, sample_number)

        cnt_sample_distance_list.append(sample_distance_list)
        cnt_color_gradient_list.append(cnt_color_gradient)

    max_size = len(max(contours, key=lambda x: len(x)))
    cnt_intensity_list = [FindCntAvgLAB(contour, image) for contour in contours]
    cnt_normsize_list = [[len(contour) / max_size] for contour in contours]

    cnt_feature_dic_list = [{'cnt': contours[i],
                             'shape': cnt_sample_distance_list[i],
                             'color': cnt_intensity_list[i],
                             'size': cnt_normsize_list[i],
                             'color_gradient': cnt_color_gradient_list[i]
                             } for i in range(len(contours))]
    feature_dic = {'shape': cnt_sample_distance_list,
                   'color': cnt_intensity_list,
                   'size': cnt_normsize_list}
    return cnt_feature_dic_list, feature_dic


'''
Goal :
shift the cnt_list  to make the starting point to the main angle 

@param 
main angle : Let the ellipse fit the contour and take the long axis points' angle as main angle
(We take y+ axis as 0')
'''


def rotate_contour(feature_list, main_angle):
    min_index = 0
    angle_offset = 0
    dis_0 = 1000
    dis_180 = 1000
    angle_0_dis = 0
    angle_180_dis = 0
    angle_0 = 0
    angle_180 = 0
    index_0 = 0
    index_180 = 0

    for i, feature in enumerate(feature_list):

        if abs(feature['angle'] - main_angle) < dis_0:
            dis_0 = abs(feature['angle'] - main_angle)
            angle_0_dis = feature['distance']
            index_0 = i
            angle_0 = feature['angle']

        if abs(feature['angle'] - main_angle - 180) < dis_180:
            dis_180 = abs(feature['angle'] - main_angle - 180)
            angle_180_dis = feature['distance']
            index_180 = i
            angle_180 = feature['angle']

        if angle_0_dis < angle_180_dis:
            min_index = index_0
            angle_offset = angle_0
        else:
            min_index = index_180
            angle_offset = angle_180

    feature_list = feature_list[min_index:] + feature_list[:min_index]

    for feature in feature_list:
        feature['angle'] -= angle_offset
        if feature['angle'] < 0:
            feature['angle'] += 360

    return feature_list


'''
@param 
contour_list : rotated (shifted) contour list that the starting point to the main angle 
n_sample : the sample refers to the PR80 point's dimension
'''


def sample_by_angle(img, feature_list, n_sample):
    '''
    Record the angle of the sample points to the centorid
    EX : If n_sample = 4 , the list will contain [90,180,270,360].
    '''
    angle_hash = []
    '''
    Record distance, angle and corrdinate of the sample points.
    The order (length) will be as same as angle_hash.
    '''
    sample_list = []
    '''
    EX : If the PR80 point is 40' , the acceptable angle could be 39.7' - 40.3' ; otherwise,
    doing the interpolation.
    '''
    angle_err = 0.3

    per_angle = 360.0 / n_sample

    for angle in np.arange(0, 360, per_angle):
        deviation = 10
        sample_angle = 0
        sample_distance = 0
        sample_coordinate = 0
        angle_match = False

        for feature in feature_list:
            if abs(feature['angle'] - angle) < min(angle_err, deviation):
                angle_match = True
                deviation = abs(feature['angle'] - angle)
                sample_angle = feature['angle']
                sample_distance = feature['distance']
                sample_coordinate = feature['coordinate']
        if angle_match:
            angle_hash.append(angle)
            sample_list.append({'distance': sample_distance, 'angle': sample_angle, 'coordinate': sample_coordinate})

    '''
    Output the list that record the distance of the sample points.
    The distance list actually represents the shape vector.
    '''
    sample_distance_list = []

    '''
    Output the coordinate of the sample points.
    '''
    sample_coordinate_list = []

    angle_hash.append(360.0)
    sample_list.append(
        {'distance': sample_list[0]['distance'], 'angle': 360.0, 'coordinate': feature_list[0]['coordinate']})

    '''
    Output the color gradient of the sample points as an obviousity .
    '''
    cnt_color_gradient = 0.0

    # use interpolat to complete the sample angle distance
    for i in range(len(angle_hash) - 1):
        sample_distance_list.append(sample_list[i]['distance'])
        sample_coordinate_list.append(sample_list[i]['coordinate'])
        cnt_color_gradient += Color_distance_by_angle(img, sample_list[i]['coordinate'], angle_hash[i])

        for inter_angle in np.arange(angle_hash[i] + per_angle, angle_hash[i + 1], per_angle):
            sample_distance_list.append(Interpolation(angle_hash[i], sample_list[i]['distance'], angle_hash[i + 1],
                                               sample_list[i + 1]['distance'], inter_angle))

            Inter_coordinate = Interpolation_coordinate(angle_hash[i], sample_list[i]['coordinate'], angle_hash[i + 1],
                                                        sample_list[i + 1]['coordinate'], inter_angle)
            sample_coordinate_list.append(Inter_coordinate)
            cnt_color_gradient += Color_distance_by_angle(img, Inter_coordinate, inter_angle)

    cnt_color_gradient /= n_sample
    return sample_distance_list, sample_coordinate_list, cnt_color_gradient



'''
Goal : 
Calculate the color gradient of the sample points.
'''


def Color_distance_by_angle(img, coordinate, angle, modle='lab'):
    if modle == 'lab':
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
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

        '''Prevent that the contour is near the boundary'''
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


def Interpolation(a, a_d, b, b_d, i):
    return (abs(i - a) * b_d + abs(b - i) * a_d) / float(abs(b - a))


def Interpolation_coordinate(a, a_d, b, b_d, i):
    return np.array([int(round(a_d[0] + (b_d[0] - a_d[0]) * (float(i - a) / (b - a)))),
                     int(round(a_d[1] + (b_d[1] - a_d[1]) * (float(i - a) / (b - a))))])


'''
Calculate the color feature used for clustring.
'''


def FindCntAvgLAB(cnt, img):
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[:] = 0
    cnt = cv2.convexHull(np.array(cnt))
    '''Fill the contour in order to get the inner points'''
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    cv2.drawContours(mask, [cnt], -1, 0, 1)

    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    '''Get the lab value according to the coordinate of all points inside the contour'''
    cnt_lab = img_lab[mask == 255]
    num = len(cnt_lab)

    if num < 1:
        return [0.0, 0.0, 0.0]

    avg_lab = [0.0, 0.0, 0.0]
    for lab in cnt_lab:
        # print rgb
        avg_lab[0] += lab[0]
        avg_lab[1] += lab[1]
        avg_lab[2] += lab[2]

    for i in range(len(avg_lab)):
        avg_lab[i] /= float(num)

    # count color intensity by A, B (LAB)
    # intensity = math.sqrt(pow(avg_lab[1], 2) + pow(avg_lab[2], 2))

    # return intensity
    return avg_lab


def FindCntHsvHis(cnt, img):
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[:] = 0

    cv2.drawContours(mask, [cnt], -1, 255, -1)
    cv2.drawContours(mask, [cnt], -1, 0, 1)

    avg = [0.0, 0.0, 0.0]
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cnt_hsv = img_hsv[mask == 255]
    num = len(cnt_hsv)

    if num == 0:
        mask[:] = 0
        cnt = cv2.convexHull(np.array(cnt))
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        cv2.drawContours(mask, [cnt], -1, 0, 1)
        cnt_hsv = img_hsv[mask == 255]
        num = len(cnt_hsv)

    H_scale = [0] * 256
    S_scale = [0] * 256
    V_scale = [0] * 256

    for hsv in cnt_hsv:
        # print rgb
        H_scale[hsv[0]] += 1
        S_scale[hsv[1]] += 1
        V_scale[hsv[2]] += 1

    maxH = max(H_scale)
    maxS = max(S_scale)
    maxV = max(V_scale)

    for i in range(256):
        H_scale[i] /= float(maxH)
        S_scale[i] /= float(maxS)
        V_scale[i] /= float(maxV)

    return H_scale + S_scale + V_scale


def FindCntLabHis(cnt, img):
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[:] = 0

    cv2.drawContours(mask, [cnt], -1, 255, -1)
    cv2.drawContours(mask, [cnt], -1, 0, 1)

    avg = [0.0, 0.0, 0.0]
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    cnt_lab = img_lab[mask == 255]
    num = len(cnt_lab)

    if num == 0:
        mask[:] = 0
        cnt = cv2.convexHull(np.array(cnt))
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        cv2.drawContours(mask, [cnt], -1, 0, 1)
        cnt_lab = img_lab[mask == 255]
        num = len(cnt_lab)

    L_scale = [0] * 256
    A_scale = [0] * 256
    B_scale = [0] * 256

    for lab in cnt_lab:
        # print rgb
        L_scale[lab[0]] += 1
        A_scale[lab[1]] += 1
        B_scale[lab[2]] += 1

    maxL = max(L_scale)
    maxA = max(A_scale)
    maxB = max(B_scale)

    for i in range(256):
        L_scale[i] /= float(maxL)
        A_scale[i] /= float(maxA)
        B_scale[i] /= float(maxB)

    return A_scale + B_scale


def FindCntRgbHis(cnt, img):
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[:] = 0

    cv2.drawContours(mask, [cnt], -1, 255, -1)
    cv2.drawContours(mask, [cnt], -1, 0, 1)

    avg = [0.0, 0.0, 0.0]
    cnt_rgb = img[mask == 255]
    num = len(cnt_rgb)

    if num == 0:
        mask[:] = 0
        cnt = cv2.convexHull(np.array(cnt))
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        cv2.drawContours(mask, [cnt], -1, 0, 1)
        cnt_rgb = img[mask == 255]
        num = len(cnt_rgb)

    R_scale = [0] * 256
    G_scale = [0] * 256
    B_scale = [0] * 256

    for rgb in cnt_rgb:
        # print rgb
        R_scale[rgb[0]] += 1
        G_scale[rgb[1]] += 1
        B_scale[rgb[2]] += 1

    maxR = max(R_scale)
    maxG = max(G_scale)
    maxB = max(B_scale)

    for i in range(256):
        R_scale[i] /= float(maxR)
        G_scale[i] /= float(maxG)
        B_scale[i] /= float(maxB)

    return R_scale + G_scale + B_scale


def FindCntAvgRGB(cnt, img):
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[:] = 0

    cv2.drawContours(mask, [cnt], -1, 255, -1)
    cv2.drawContours(mask, [cnt], -1, 0, 1)

    avg = [0.0, 0.0, 0.0]
    cnt_rgb = img[mask == 255]
    num = len(cnt_rgb)

    if num == 0:
        mask[:] = 0
        cnt = cv2.convexHull(np.array(cnt))
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        cv2.drawContours(mask, [cnt], -1, 0, 1)
        cnt_rgb = img[mask == 255]
        num = len(cnt_rgb)

    for rgb in cnt_rgb:
        # print rgb
        avg[0] += rgb[0]
        avg[1] += rgb[1]
        avg[2] += rgb[2]

    return [float(avg[0]) / (num * 255), float(avg[1]) / (num * 255), float(avg[2]) / (num * 255)]
    # return [ float(avg[0])/(num), float(avg[1])/(num), float(avg[2])/(num) ]
