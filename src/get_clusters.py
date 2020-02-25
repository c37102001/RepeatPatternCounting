import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt
from ipdb import set_trace as pdb


def get_clusters(cluster_cfg, contours, cnt_dicts, drawer, do_draw=False):

    # Do hierarchicalclustering by shape, color, and size
    label_dict = {}
    for feature_type in ['size', 'shape', 'color']:
        feature_list = [cnt_dic[feature_type] for cnt_dic in cnt_dicts]

        # ndarray e.g. ([1, 1, 1, 1, 1, 3, 3, 2, 2, 2]), len=#feature_list
        labels = hierarchical_clustering(cluster_cfg, feature_list, feature_type, drawer, do_draw)
        label_dict[feature_type] = labels

        if do_draw:
            img = drawer.blank_img()
            for label in set(labels):
                cnt_dic_list_by_groups = [c for i, c in enumerate(contours) if labels[i] == label]
                img = drawer.draw_same_color(cnt_dic_list_by_groups, img)
            desc = f'2-1_{feature_type.capitalize()}Group'
            drawer.save(img, desc)

    # combine the label clustered by size, shape, and color. ex: (0,1,1), (2,0,1)
    combine_labels = []
    for size, shape, color in zip(label_dict['size'], label_dict['shape'], label_dict['color']):
        combine_labels.append((size, shape, color))

    # find the final group by the intersected label and draw
    img = drawer.blank_img()
    groups_cnt_dicts = []
    for combine_label in set(combine_labels):

        group_idx = [idx for idx, label in enumerate(combine_labels) if label == combine_label]
        group_cnt_dicts = [cnt_dicts[i] for i in group_idx]
        groups_cnt_dicts.append(group_cnt_dicts)

        # for do_draw
        cnts = [contours[i] for i in group_idx]
        img = drawer.draw_same_color(cnts, img)
        
    if do_draw or True:
        desc = f'2-2_GroupedResult'
        drawer.save(img, desc)
    
    return groups_cnt_dicts


def hierarchical_clustering(cluster_cfg, feature_list, feature_type, drawer, do_draw=False):
    if len(feature_list) <= 3:
        return [0] * len(feature_list)

    n_before = eval(cluster_cfg['n_before'])
    thres = eval(cluster_cfg[feature_type + '_thres'])
    
    # hierarchically link features by order of distance(measured by 'ward'), output a hierarchical tree
    # return ndarray sized [#feature_list-1, 4], 4 means (group idx1, gp idx2, gp_distance, included ele num)
    feature_dist_hierarchy = linkage(feature_list, 'ward')

    # distance between every two groups, sized [#feature - 1]
    dist_list = feature_dist_hierarchy[:, 2]

    # difference of distance between previous one, sized [#feature - 2]
    diff_list = np.diff(dist_list)

    # count avg for those who is larger than all_avg_diff as threshold
    all_avg_diff = sum(diff_list) / len(diff_list)
    larger_than_avgs_diffs = [diff for diff in diff_list if diff > all_avg_diff]
    diff_threshold = sum(larger_than_avgs_diffs) / float(len(larger_than_avgs_diffs))
    
    # count differ ratio between previous n_before differs, sized [#feature - 3]
    ratio_list = []
    for i, diff in enumerate(diff_list[1:], start=1):
        if diff < diff_threshold:         # skip if less than average ratio
            ratio_list.append(0)
            continue
        avg_index = [i for i in range(max(0, i-n_before), i)]
        diff_avg = sum(diff_list[avg_index]) / len(avg_index)
        ratio = diff / diff_avg if not diff_avg == 0 else 0
        ratio_list.append(ratio)
    
    max_ratio = max(ratio_list)
    max_ratio_idx = ratio_list.index(max_ratio)
    if max_ratio < 2.0:
        print(f'[{feature_type}] clustering all in one group! max_ratio:', max_ratio)
        return [0] * len(feature_list)

    # to find the 'target'(not always max) difference idx, plus one to max ratio idx
    target_diff_idx = max_ratio_idx + 1

    while target_diff_idx < len(diff_list) and diff_list[target_diff_idx] < thres:
        target_diff_idx += 1
    if target_diff_idx == len(diff_list):
        print(f'[{feature_type}] clustering all in one group!')
        return [0] * len(feature_list)
    
    target_diff = diff_list[target_diff_idx]

    # by distance[dist_idx] and distance[dist_idx+1], we get difference[dist_idx(=diff_idx)]
    # and we should take previous one(distance[dist_idx]) as threshold, so we choose thres_dist_idx = target_diff_idx.
    thres_dist_idx = target_diff_idx
    thres_dist = dist_list[thres_dist_idx]

    # plot difference bar chart
    if do_draw:
        plt.bar(x=range(len(diff_list)), height=diff_list)
        plt.title(f'{feature_type} cut idx: {target_diff_idx} | value: {target_diff:.3f} | ratio: {max_ratio:.3f}')
        save_path = f'{drawer.output_path}{drawer.img_name}_2-1_{feature_type.capitalize()}Group.png'
        plt.savefig(save_path)
        plt.close()

    clusters = fcluster(feature_dist_hierarchy, thres_dist, criterion='distance')
    return clusters