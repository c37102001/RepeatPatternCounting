import numpy as np
import cv2

switchColor = [(255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 128, 0),
               (255, 0, 128), (128, 0, 255), (128, 255, 0), (0, 128, 255), (0, 255, 128), (128, 128, 0), (128, 0, 128),
               (0, 128, 128), (255, 64, 0), (255, 0, 64), (64, 255, 0), (64, 0, 255), (0, 255, 64), (0, 64, 255)]


def check_overlap(cnt_dic_list, keep='keep_inner'):
    '''
    @param
    keep = 'keep_inner'(default) :
    Since OpenCV FindContours() will find two contours(inner and outer) of a single closed
    edge component, we choose the inner one to preserve. (The reason is that the outter one
    is easily connected with the surroundings.)

    keep = 'group_weight' :
    Since there's two edge results(Canny and SF), we should decide which contours
    to preserved if they are overlapped.
    check if overlap contours are same contour , if true makes them same label
    '''

    if cnt_dic_list == []:
        return []

    checked_list = []

    if keep == 'group_weight':

        label_list = [x['label'] for x in cnt_dic_list]
        label_change_list = []
        ''''''
        label_group_change = []
        label_change_dic = {}

        '''
        Compare each 2 contours and check if they are overlapped.
        If they are overlapped, change the label of the less group count one to the other's label whose group count is more.
        '''
        for cnt_i in range(len(cnt_dic_list) - 1):
            for cnt_k in range(cnt_i + 1, len(cnt_dic_list)):

                if cnt_dic_list[cnt_i]['group_weight'] > 0 and cnt_dic_list[cnt_k]['group_weight'] > 0:
                    if is_overlap(cnt_dic_list[cnt_i]['cnt'], cnt_dic_list[cnt_k]['cnt']):

                        if cnt_dic_list[cnt_i]['group_weight'] > cnt_dic_list[cnt_k]['group_weight']:
                            cnt_dic_list[cnt_k]['group_weight'] = 0
                            label_change_list.append((cnt_dic_list[cnt_k]['label'], cnt_dic_list[cnt_i]['label']))
                        else:
                            cnt_dic_list[cnt_i]['group_weight'] = 0
                            label_change_list.append((cnt_dic_list[cnt_i]['label'], cnt_dic_list[cnt_k]['label']))

        # check if overlap contours are same contour , if true makes them same label
        for label_change in set(label_change_list):
            '''0.5 Changeable'''
            if label_change_list.count(label_change) >= 0.5 * label_list.count(label_change[0]):
                found = False
                for label_group_i in range(len(label_group_change)):
                    if label_change[0] in label_group_change[label_group_i]:
                        found = True
                        label_group_change[label_group_i].append(label_change[1])
                    elif label_change[1] in label_group_change[label_group_i]:
                        found = True
                        label_group_change[label_group_i].append(label_change[0])

                if not found:
                    label_group_change.append([label_change[0], label_change[1]])

                # label_change_dic[label_change[0]] = label_change[1]

        for label_group in label_group_change:
            for label in label_group:
                label_change_dic[label] = label_group[0]

        for cnt_dic in cnt_dic_list:
            if cnt_dic['group_weight'] > 0:
                if cnt_dic['label'] in label_change_dic:
                    cnt_dic['label'] = label_change_dic[cnt_dic['label']]
                checked_list.append(cnt_dic)
    else:

        if keep == 'keep_inner':
            # sort list from little to large
            cnt_dic_list.sort(key=lambda x: len(x['cnt']), reverse=False)

        elif keep == 'keep_outer':
            cnt_dic_list.sort(key=lambda x: len(x['cnt']), reverse=True)

        for c_dic in cnt_dic_list:
            if is_overlap_all(c_dic, checked_list):
                continue
            checked_list.append(c_dic)

            # end keep if

    return checked_list

def is_overlap_all(cnt_dic, cnt_dic_list):
    '''
    Determine if one contour contains/contained  other contours in a input list.
    '''

    if cnt_dic == [] or len(cnt_dic_list) < 1:
        return False

    for c_dic in cnt_dic_list:
        # if len(c) == len(cnt) and GetCentroid(c) == GetCentroid(cnt):
        ##print 'same one'
        # continue
        if IsOverlap(cnt_dic['cnt'], c_dic['cnt']):
            return True

    return False


def is_overlap(cnt1, cnt2):
    """
    Determine that if one contour contains another one.
    """

    if cnt1 == [] or cnt2 == []:
        return False

    c1M = get_centroid(cnt1)
    c2M = get_centroid(cnt2)
    c1_min_d = min_distance(cnt1)
    c2_min_d = min_distance(cnt2)
    moment_d = eucl_distance(c1M, c2M)

    if min(c1_min_d, c2_min_d) == 0:
        return False

    # TODO why or? why ratio = 3?
    return (moment_d < c1_min_d or moment_d < c2_min_d) and max(c1_min_d, c2_min_d) / min(c1_min_d, c2_min_d) <= 3
    # return (moment_d < c1_min_d and moment_d < c2_min_d) and min(c1_min_d, c2_min_d) / max(c1_min_d, c2_min_d) > 0.6


def get_centroid(cnt):
    """
    Calculate the average coordinate as centroid.
    """
    if len(cnt) == 1:
        return cnt
    elif len(cnt) == 2:
        return (cnt[0] + cnt[1]) / 2

    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return cx, cy


def min_distance(cnt):
    '''
    Calculate the minimum distance between centroid to the contour.
    '''

    cM = get_centroid(cnt)
    if len(cnt[0][0]) == 1:
        cnt = cnt[0]
    min_d = eucl_distance((cnt[0][0][0], cnt[0][0][1]), cM)
    for c in cnt:
        d = eucl_distance((c[0][0], c[0][1]), cM)
        if d < min_d:
            min_d = d

    return min_d


def eucl_distance(a, b):
    if type(a) != np.ndarray:
        a = np.array(a)
    if type(b) != np.ndarray:
        b = np.array(b)

    return np.linalg.norm(a - b)
