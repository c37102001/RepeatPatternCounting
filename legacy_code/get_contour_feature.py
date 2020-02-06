'''
Goal : 
Extract features for clustering from filtered contours.
The features contains three clustering features (color, size, shape) and one obviousity feature.(color gradient)

'''

import cv2, time
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import ipdb

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

    if len(contours) < 1:
        print('No any contour')
        return [], [], [], [], []

    '''
    record the distance between pixels and the centroid
    the number of sample distance depend on the dimension of the contour
    '''
    c_list_d = []

    ''' record the color gradient of the contour '''
    cnt_color_gradient_list = []

    '''every pixel coordinate in the contour'''
    c_list = []

    '''
    several probable dimension of contour shape
    If pixel s of the contour is between 4-8 , then we take 4 as its dimension.
    '''
    factor_360 = [4, 8, 20, 40, 90, 180, 360]
    min_contour_len = len(contours[0])

    ordered_cnt_list = contours
    # sort list from little to large
    ordered_cnt_list.sort(key=lambda x: len(x), reverse=False)

    most_cnt_len = len(ordered_cnt_list[int(len(ordered_cnt_list) * 0.8)])
    sample_number = 360
    min_value = 1000

    for factor in factor_360[1:]:
        differ = abs(most_cnt_len - factor)
        if differ < min_value:
            min_value = differ
            sample_number = factor

    sample_number = min(sample_number, 360)
    sample_number = max(sample_number, 4)

    for i in range(len(contours)):
        if len(contours[i]) < min_contour_len:
            min_contour_len = len(contours[i])

    for i in range(len(contours)):
        tmp_list = []
        if (len(contours[i]) < 10):
            continue

        M = cv2.moments(contours[i])  # find centroid
        if M['m00'] == 0:
            continue

        c_list.append(contours[i])

        # centroid(cx,cy)
        cx = (M['m10'] / M['m00'])
        cy = (M['m01'] / M['m00'])

        max_dis = 0
        img = image.copy()

        for c in contours[i]:

            # the (0,0) in image is the left top point
            v1 = (c[0][1] - cy, cx - c[0][0])
            v2 = (0, height)

            if v1[0] == 0 and v1[1] >= 0:
                angle = 0.0
            elif v1[0] == 0 and v1[1] < 0:
                angle = 180.0
            else:
                angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                angle = angle * 180 / np.pi

            if (v1[0] < 0):
                angle = 360 - angle

            if Eucl_distance(c[0], (cx, cy)) > max_dis:
                max_dis = Eucl_distance(c[0], (cx, cy))
            tmp_list.append({'distance': Eucl_distance(c[0], (cx, cy)), 'angle': angle, 'coordinate': c[0]})

        for t in tmp_list:
            t['distance'] = t['distance'] / max_dis

        '''Ellipse fitting'''
        ellipse = cv2.fitEllipse(contours[i])

        tmp_list = rotate_contour(tmp_list, ellipse[2])

        distance_list, coordinate_list, cnt_color_gradient = sample_by_angle(image, tmp_list, sample_number)

        # if len(contours[i]) < 50:
        # distance_list = [1.0]*len(distance_list)

        c_list_d.append(distance_list)
        cnt_color_gradient_list.append(cnt_color_gradient)

    # end contour for

    if len(c_list) < 1:
        print('No any contour')
        return [], [], [], [], []

    cnt_rgb_list = []
    cnt_lab_list = []
    cnt_hsv_list = []
    cnt_intensity_list = []
    size_list = []
    max_size = len(max(c_list, key=lambda x: len(x)))
    for cnt in c_list:
        # cnt_rgb_list.append( FindCntAvgRGB(cnt, image) )   # 3 dimension
        # cnt_rgb_list.append( FindCntRgbHis(cnt, image) )   # 256*3 dimension
        # cnt_lab_list.append( FindCntLabHis(cnt, image) )   # 256*3 dimension
        # cnt_hsv_list.append( FindCntHsvHis(cnt, image) )   # 256*3 dimension
        cnt_intensity_list.append(FindCntAvgLAB(cnt, image))  # 3 dimension

        size_list.append([max(len(cnt), 30) / float(max_size)])
        # size_list.append( [len(cnt)] )

    return c_list, c_list_d, cnt_intensity_list, size_list, cnt_color_gradient_list


'''
Goal :
shift the cnt_list  to make the starting point to the main angle 

@param 
main angle : Let the ellipse fit the contour and take the long axis points' angle as main angle
(We take y+ axis as 0')
'''


def rotate_contour(contour_list, main_angle):
    # print main_angle

    min_distance = 1000
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

    for i in range(len(contour_list)):

        if abs(contour_list[i]['angle'] - main_angle) < dis_0:
            dis_0 = abs(contour_list[i]['angle'] - main_angle)
            angle_0_dis = contour_list[i]['distance']
            index_0 = i
            angle_0 = contour_list[i]['angle']

        if abs(contour_list[i]['angle'] - main_angle - 180) < dis_180:
            dis_180 = abs(contour_list[i]['angle'] - main_angle - 180)
            angle_180_dis = contour_list[i]['distance']
            index_180 = i
            angle_180 = contour_list[i]['angle']

        if angle_0_dis < angle_180_dis:
            min_index = index_0
            angle_offset = angle_0
        else:
            min_index = index_180
            angle_offset = angle_180

    rotate_list = contour_list[min_index:] + contour_list[:min_index]

    for i in range(len(rotate_list)):
        rotate_list[i]['angle'] = rotate_list[i]['angle'] - angle_offset
        if rotate_list[i]['angle'] < 0:
            rotate_list[i]['angle'] += 360

    return rotate_list


'''
@param 
contour_list : rotated (shifted) contour list that the starting point to the main angle 
n_sample : the sample refers to the PR80 point's dimension
'''


def sample_by_angle(img, contour_list, n_sample):
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
    tmp_i = 0

    per_angle = 360.0 / n_sample

    for angle in np.arange(0, 360, per_angle):
        # print angle
        index = -1
        deviation = 10
        sample_angle = 0
        sample_distance = 0
        sample_coordinate = 0

        angle_match = False
        for i in range(tmp_i, len(contour_list)):

            if abs(contour_list[i]['angle'] - angle) < angle_err and abs(contour_list[i]['angle'] - angle) < deviation:
                angle_match = True
                deviation = abs(contour_list[i]['angle'] - angle)
                index = i
                sample_angle = contour_list[i]['angle']
                sample_distance = contour_list[i]['distance']
                sample_coordinate = contour_list[i]['coordinate']

            # elif index >=0 :
            # sample_list.append( { 'distance':contour_list[i-1]['distance'], 'angle':contour_list[i-1]['angle'] } )
            # angle_hash.append(angle)
            # break
        if angle_match:
            angle_hash.append(angle)
            sample_list.append({'distance': sample_distance, 'angle': sample_angle, 'coordinate': sample_coordinate})
        # end contour_list for 
    # end angle for

    '''
    Output the list that record the distance of the sample points.
    The distance list actually represents the shape vector.
    '''
    distance_list = []

    '''
    Output the coordinate of the smaple points.
    '''
    coordinate_list = []

    angle_hash.append(360.0)
    sample_list.append(
        {'distance': sample_list[0]['distance'], 'angle': 360.0, 'coordinate': contour_list[0]['coordinate']})

    '''
    Output the color gradient of the sample points as an obviousity .
    '''
    cnt_color_gradient = 0.0

    # use interpolat to complete the sample angle distance
    for i in range(len(angle_hash) - 1):
        distance_list.append(sample_list[i]['distance'])
        coordinate_list.append(sample_list[i]['coordinate'])

        cnt_color_gradient += Color_distance_by_angle(img, sample_list[i]['coordinate'], angle_hash[i])

        # print 'out:',angle_hash[i]
        for inter_angle in np.arange(angle_hash[i] + per_angle, angle_hash[i + 1], per_angle):
            distance_list.append(Interpolation(angle_hash[i], sample_list[i]['distance'], angle_hash[i + 1],
                                               sample_list[i + 1]['distance'], inter_angle))

            Inter_coordinate = Interpolation_coordinate(angle_hash[i], sample_list[i]['coordinate'], angle_hash[i + 1],
                                                        sample_list[i + 1]['coordinate'], inter_angle)

            coordinate_list.append(Inter_coordinate)
            cnt_color_gradient += Color_distance_by_angle(img, Inter_coordinate, inter_angle)

    # if len(distance_list) != 360 :
    # print distance_list
    # print 'len(coordinate_list):',coordinate_list
    return distance_list, coordinate_list, cnt_color_gradient / n_sample


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

        coordinate_A = [0, 0]
        coordinate_B = [0, 0]

        '''Find two related points in the 5*5 matrix'''
        if (angle >= 0 and angle < 15) or (angle >= 345) or (angle >= 165 and angle < 195):
            coordinate_A = [coordinate[0], coordinate[1] - 2]
            coordinate_B = [coordinate[0], coordinate[1] + 2]
        elif (angle >= 15 and angle < 38) or (angle >= 195 and angle < 218):
            coordinate_A = [coordinate[0] + 1, coordinate[1] - 2]
            coordinate_B = [coordinate[0] - 1, coordinate[1] + 2]
        elif (angle >= 38 and angle < 53) or (angle >= 218 and angle < 233):
            coordinate_A = [coordinate[0] + 2, coordinate[1] - 2]
            coordinate_B = [coordinate[0] - 2, coordinate[1] + 2]
        elif (angle >= 53 and angle < 75) or (angle >= 233 and angle < 255):
            coordinate_A = [coordinate[0] + 2, coordinate[1] - 1]
            coordinate_B = [coordinate[0] - 2, coordinate[1] + 1]
        elif (angle >= 75 and angle < 105) or (angle >= 255 and angle < 285):
            coordinate_A = [coordinate[0] + 2, coordinate[1]]
            coordinate_B = [coordinate[0] - 2, coordinate[1]]
        elif (angle >= 105 and angle < 128) or (angle >= 285 and angle < 308):
            coordinate_A = [coordinate[0] + 2, coordinate[1] + 1]
            coordinate_B = [coordinate[0] - 2, coordinate[1] - 1]
        elif (angle >= 128 and angle < 143) or (angle >= 308 and angle < 323):
            coordinate_A = [coordinate[0] + 2, coordinate[1] + 2]
            coordinate_B = [coordinate[0] - 2, coordinate[1] - 2]
        elif (angle >= 143 and angle < 165) or (angle >= 323 and angle < 345):
            coordinate_A = [coordinate[0] + 1, coordinate[1] + 2]
            coordinate_B = [coordinate[0] - 1, coordinate[1] - 2]

        '''Prevent that the contour is near the boundary'''
        height, width = img.shape[:2]
        coordinate_A[0] = max(coordinate_A[0], 0)
        coordinate_A[1] = max(coordinate_A[1], 0)
        coordinate_B[0] = max(coordinate_B[0], 0)
        coordinate_B[1] = max(coordinate_B[1], 0)
        coordinate_A[0] = min(coordinate_A[0], width - 1)
        coordinate_A[1] = min(coordinate_A[1], height - 1)
        coordinate_B[0] = min(coordinate_B[0], width - 1)
        coordinate_B[1] = min(coordinate_B[1], height - 1)

        # (x,y) x for height, y for width
        '''Take the color gradient value of the related points'''
        point_A_l_value = float(img_l[coordinate_A[1], coordinate_A[0]])
        point_A_a_value = float(img_a[coordinate_A[1], coordinate_A[0]])
        point_A_b_value = float(img_b[coordinate_A[1], coordinate_A[0]])
        point_B_l_value = float(img_l[coordinate_B[1], coordinate_B[0]])
        point_B_a_value = float(img_a[coordinate_B[1], coordinate_B[0]])
        point_B_b_value = float(img_b[coordinate_B[1], coordinate_B[0]])

        return math.sqrt(
            pow(abs(point_A_l_value - point_B_l_value), 2) + pow(abs(point_A_a_value - point_B_a_value), 2) + pow(
                abs(point_A_b_value - point_B_b_value), 2))


def Eucl_distance(a, b):
    if type(a) != np.ndarray:
        a = np.array(a)
    if type(b) != np.ndarray:
        b = np.array(b)

    return np.linalg.norm(a - b)


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

    avg = [0.0, 0.0, 0.0]
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
    intensity = math.sqrt(pow(avg_lab[1], 2) + pow(avg_lab[2], 2))

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

# if __name__ == "__main__":
# start = time.time()
# main()
# print 'Total time : ',time.time() - start ,'s'
# print 'All finished!'
