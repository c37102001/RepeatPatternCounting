import numpy as np
import cv2
from contour import check_property, check_overlap
from cluster import Hierarchical_clustering
import get_contour_feature
from ipdb import set_trace as pdb


class ContourDrawer:
    def __init__(self, output_path, img_name):
        self.output_path = output_path
        self.img_name = img_name
        self.color_index = 0
        self.switchColor = \
            [(255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 128, 0),
             (255, 0, 128), (128, 0, 255), (128, 255, 0), (0, 128, 255), (0, 255, 128), (128, 128, 0), (128, 0, 128),
             (0, 128, 128), (255, 64, 0), (255, 0, 64), (64, 255, 0), (64, 0, 255), (0, 255, 64), (0, 64, 255)]
    
    def reset(self):
        self.contour_image = np.zeros(self.image_resi_shape, np.uint8)
    
    def draw(self, contour):
        color = self.switchColor[self.color_index % len(self.switchColor)]
        cv2.drawContours(self.contour_image, contour, -1, color, 2)
        self.color_index += 1

    def save(self, info):
        img_path = '{}{}_{}.jpg'.format(self.output_path, self.img_name, info)
        cv2.imwrite(img_path, self.contour_image)


def check_and_cluster(image_resi, contours, drawer, edge_type, _writeImg):
    drawer.image_resi_shape = image_resi.shape

    if _writeImg['original_contour'] or True:
        drawer.reset()
        for contour in contours:
            drawer.draw([contour])
        info = 'd_OriginContour_{}'.format(edge_type)
        drawer.save(info)

    re_height, re_width = image_resi.shape[:2]
    contours = check_property(contours, re_height, re_width)
    contours = check_overlap(contours)

    if _writeImg['contour_filtered'] or True:
        drawer.reset()
        for contour in contours:
            drawer.draw([contour])
        info = 'e_FilteredContour_{}'.format(edge_type)
        drawer.save(info)

    
    # Feature extraction and cluster
    print('Extract contour feature')
    cnt_feature_dic_list, feature_dic = get_contour_feature.extract_feature(image_resi, contours)
    cnt_features = [cnt_dic['cnt'] for cnt_dic in cnt_feature_dic_list]

    label_list_dic = {}
    # Respectively use shape, color, and size as feature set to cluster
    for feature_type in ['size', 'shape', 'color']:
        print('feature_type:', feature_type)

        contour_feature_list = feature_dic[feature_type]

        # hierarchical clustering, output the classified consequence
        label_list = Hierarchical_clustering(contour_feature_list, drawer.img_name, feature_type, edge_type, draw=_writeImg['cluster_histogram'])

        unique_label, label_counts = np.unique(label_list, return_counts=True) 
        # array([1, 2]), array([ 66, 101])

        if _writeImg[feature_type] or True:
            drawer.reset()
            for label in unique_label:
                tmp_splited_group = []
                for i in range(len(label_list)):
                    if label_list[i] == label:
                        tmp_splited_group.append(contours[i])
                
                drawer.draw(np.array(tmp_splited_group))

            info = 'f_Feature{}_{}'.format(feature_type.capitalize(), edge_type)
            drawer.save(info)
        
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
        final_group.append({'cnt': cnt_features, 'obvious_weight': 0, 'group_dic': tmp_group})

    if _writeImg['original_result'] or True:
        info = 'g_OriginalResult_{}'.format(edge_type)
        drawer.save(info)

    return final_group
