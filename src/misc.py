import cv2
import numpy as np
from ipdb import set_trace as pdb


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






