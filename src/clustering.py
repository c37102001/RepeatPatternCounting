import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt
from ipdb import set_trace as pdb


AVG_NUM = 5

def hierarchical_clustering(feature_list, feature_type, edge_type, drawer, do_draw=False):
    
    # hierarchically link features by order of distance(measured by 'ward'), output a hierarchical tree
    # return ndarray sized [#feature_list-1, 4], 4 means (group idx1, gp idx2, gp_distance, included ele num)
    feature_dist_hierarchy = linkage(feature_list, 'ward')   

    # distance between every two groups, sized [#feature - 1]
    dist_list = feature_dist_hierarchy[:, 2]

    # difference of distance between previous one, sized [#feature - 2]
    diff_list = np.diff(dist_list)
    
    # count differ ratio between previous AVG_NUM differs, sized [#feature - 3]
    ratio_list = []
    for i, diff in enumerate(diff_list[1:], start=1):
        avg_index = [i for i in range(max(0, i-AVG_NUM), i)]
        diff_avg = sum(diff_list[avg_index]) / len(avg_index)
        ratio = diff / diff_avg
        ratio_list.append(ratio)
    
    max_ratio = max(ratio_list)
    max_ratio_idx = ratio_list.index(max_ratio)

    # to find the 'target'(not always max) difference idx, plus one to max ratio idx
    target_diff_idx = max_ratio_idx + 1
    target_diff = diff_list[target_diff_idx]

    # by distance[dist_idx] and distance[dist_idx+1], we get difference[dist_idx(=diff_idx)]
    # and we should take previous one(distance[dist_idx]) as threshold, so we choose thres_dist_idx = target_diff_idx.
    thres_dist_idx = target_diff_idx
    thres_dist = dist_list[thres_dist_idx]

    # plot difference bar chart
    if do_draw:
        plt.bar(x=range(len(diff_list)), height=diff_list)
        plt.title(f'{feature_type} cut idx: {target_diff_idx} | value: {target_diff:.3f} | ratio: {max_ratio:.3f}')
        save_path = f'{drawer.output_path}{drawer.img_name}_f_{edge_type}({feature_type})_hist.png'
        plt.savefig(save_path)
        plt.close()

    clusters = fcluster(feature_dist_hierarchy, thres_dist, criterion='distance')
    return clusters