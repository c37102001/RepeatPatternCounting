import cv2
import numpy as np
from skimage.filters import roberts, sobel
from ipdb import set_trace as pdb


def sobel_edge_detect(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.bilateralFilter(img, 9, 75, 75)
    edge_img = sobel(img)
    edge_img = np.uint8(edge_img * 255)
    return edge_img

def canny_edge_detect(img,
                      gaussian_para=7,
                      gaussian_filter=True,
                      do_sharpen=True,
                      split_n_row=1,
                      split_n_column=1,
                      edge_by_channel=['bgr_gray']):

    # filter the noise
    if gaussian_filter:
        img = cv2.GaussianBlur(img, (gaussian_para, gaussian_para), 0)

    re_height, re_width = img.shape[:2]

    offset_r = re_height / split_n_row  # 736/1 = 736
    offset_c = re_width / split_n_column  # 1104

    edged = np.zeros(img.shape[:2], np.uint8)

    for row_n in np.arange(0, split_n_row, 0.5):  # 0, 0.5
        for column_n in np.arange(0, split_n_column, 0.5):  # 0, 0.5

            r_l = int(row_n * offset_r)  # 736*0, 736*0.5
            r_r = int((row_n + 1) * offset_r)  # 736*1, 736*1.5->736*1
            c_l = int(column_n * offset_c)
            c_r = int((column_n + 1) * offset_c)

            if row_n == split_n_row - 0.5:
                r_r = int(re_height)
            if column_n == split_n_column - 0.5:
                c_r = int(re_width)

            BGR_dic, HSV_dic, LAB_dic = split_color_channel(img[r_l: r_r, c_l: c_r])

            channel_img_dic = {'bgr_gray': BGR_dic['img_bgr_gray'],
                               'bgr_b': BGR_dic['img_b'],
                               'bgr_g': BGR_dic['img_g'],
                               'bgr_r': BGR_dic['img_r'],
                               'hsv_h': HSV_dic['img_h'],
                               'hsv_s': HSV_dic['img_s'],
                               'hsv_v': HSV_dic['img_v'],
                               'lab_l': LAB_dic['img_l'],
                               'lab_a': LAB_dic['img_a'],
                               'lab_b': LAB_dic['img_b']}
            channel_thre_dic = {'bgr_gray': BGR_dic['thre_bgr_gray'],
                                'bgr_b': BGR_dic['thre_b'],
                                'bgr_g': BGR_dic['thre_g'],
                                'bgr_r': BGR_dic['thre_r'],
                                'hsv_h': HSV_dic['thre_h'],
                                'hsv_s': HSV_dic['thre_s'],
                                'hsv_v': HSV_dic['thre_v'],
                                'lab_l': LAB_dic['thre_l'],
                                'lab_a': LAB_dic['thre_a'],
                                'lab_b': LAB_dic['thre_b']}

            for chan in edge_by_channel:
                if channel_thre_dic[chan] > 20:
                    edged[r_l: r_r, c_l: c_r] |= cv2.Canny(
                        channel_img_dic[chan], channel_thre_dic[chan] * 0.5, channel_thre_dic[chan])

    return edged


def sharpen(img):
    kernel_sharpen = np.array([[-1, -1, -1, -1, -1],
                               [-1, 2, 2, 2, -1],
                               [-1, 2, 8, 2, -1],
                               [-1, 2, 2, 2, -1],
                               [-1, -1, -1, -1, -1]]) / 8.0

    return cv2.filter2D(img, -1, kernel_sharpen)


def split_color_channel(img):
    """
    Find all the attribute of three color models. (RGB/HSV/LAB)
    Return in a dictionary type.
    """

    bgr_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # bgr_gray = cv2.GaussianBlur(bgr_gray, (3, 3), 0)
    thresh_bgr_gray = cv2.threshold(bgr_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]

    bgr_b = img[:, :, 0]
    bgr_g = img[:, :, 1]
    bgr_r = img[:, :, 2]
    bgr_b = cv2.GaussianBlur(bgr_b, (5, 5), 0)
    bgr_g = cv2.GaussianBlur(bgr_g, (5, 5), 0)
    bgr_r = cv2.GaussianBlur(bgr_r, (5, 5), 0)
    thresh_bgr_b = cv2.threshold(bgr_b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    thresh_bgr_g = cv2.threshold(bgr_g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    thresh_bgr_r = cv2.threshold(bgr_r, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
    hsv_h = hsv[:, :, 0]
    hsv_s = hsv[:, :, 1]
    hsv_v = hsv[:, :, 2]
    hsv_h = cv2.GaussianBlur(hsv_h, (5, 5), 0)
    hsv_s = cv2.GaussianBlur(hsv_s, (5, 5), 0)
    hsv_v = cv2.GaussianBlur(hsv_v, (5, 5), 0)
    thresh_hsv_h = cv2.threshold(hsv_h, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    thresh_hsv_s = cv2.threshold(hsv_s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    thresh_hsv_v = cv2.threshold(hsv_v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab = cv2.GaussianBlur(lab, (5, 5), 0)
    lab_l = lab[:, :, 0]
    lab_a = lab[:, :, 1]
    lab_b = lab[:, :, 2]
    lab_l = cv2.GaussianBlur(lab_l, (5, 5), 0)
    lab_a = cv2.GaussianBlur(lab_a, (5, 5), 0)
    lab_b = cv2.GaussianBlur(lab_b, (5, 5), 0)
    thresh_lab_l = cv2.threshold(lab_l, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    thresh_lab_a = cv2.threshold(lab_a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    thresh_lab_b = cv2.threshold(lab_b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]

    BGR_dic = {
        'img_bgr': img,
        'img_bgr_gray': bgr_gray,
        'img_b': bgr_b,
        'img_g': bgr_g,
        'img_r': bgr_r,
        'thre_bgr_gray': thresh_bgr_gray,
        'thre_b': thresh_bgr_b,
        'thre_g': thresh_bgr_g,
        'thre_r': thresh_bgr_r
    }
    HSV_dic = {
        'img_hsv': hsv,
        'img_h': hsv_h,
        'img_s': hsv_s,
        'img_v': hsv_v,
        'thre_h': thresh_hsv_h,
        'thre_s': thresh_hsv_s,
        'thre_v': thresh_hsv_v
    }
    LAB_dic = {
        'img_lab': lab,
        'img_l': lab_l,
        'img_a': lab_a,
        'img_b': lab_b,
        'thre_l': thresh_lab_l,
        'thre_a': thresh_lab_a,
        'thre_b': thresh_lab_b
    }

    return BGR_dic, HSV_dic, LAB_dic


