import numpy as np
from utils import filter_contours, remove_overlap
from cluster import hierarchical_clustering
import get_contour_feature
from ipdb import set_trace as pdb


def check_and_cluster(contours, drawer, edge_type, do_draw=False):
    '''
    Args:
        contours: (list of ndarray), len = Num_of_cnts
        contours[0].shape = (Num_of_pixels, 1, 2)
    '''

    height, width = drawer.image.shape[:2]

    # filter by area, perimeter, solidity, edge_num
    contours = filter_contours(contours, height, width)

    # remove outer overlap contour
    contours = remove_overlap(contours)

    # ============= edit to here ===================
    
    print('Feature extraction and cluster')
    # cnt_feature_dic_list = [
    #     {'cnt': contours[i], 
    #      'shape': cnt_sample_distance_list[i], 
    #      'color': cnt_intensity_list[i],
    #      'size': cnt_normsize_list[i],
    #      'color_gradient': cnt_color_gradient_list[i]} for i in range(len(contours))]
    # feature_dic = {'shape': cnt_sample_distance_list,
    #                'color': cnt_intensity_list,
    #                'size': cnt_normsize_list}
    cnt_feature_dic_list, feature_dic = get_contour_feature.extract_feature(drawer.image, contours)
    cnt_features = [cnt_dic['cnt'] for cnt_dic in cnt_feature_dic_list]

    label_list_dic = {}
    # Respectively use shape, color, and size as feature set to cluster
    for feature_type in ['size', 'shape', 'color']:
        print('[{}] feature_type:{}'.format(edge_type, feature_type))

        contour_feature_list = feature_dic[feature_type]

        # hierarchical clustering, output the classified consequence
        label_list = hierarchical_clustering(contour_feature_list, drawer.img_name, feature_type, edge_type, do_draw=do_draw)

        unique_label, label_counts = np.unique(label_list, return_counts=True) 
        # array([1, 2]), array([ 66, 101])

        if do_draw:
            drawer.reset()
            for label in unique_label:
                tmp_splited_group = []
                for i in range(len(label_list)):
                    if label_list[i] == label:
                        tmp_splited_group.append(contours[i])
                
                drawer.draw(np.array(tmp_splited_group))

            desc = 'f_Feature{}_{}'.format(feature_type.capitalize(), edge_type)
            drawer.save(desc)
        
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

    if do_draw:
        desc = 'g_OriginalResult_{}'.format(edge_type)
        drawer.save(desc)

    return final_group
