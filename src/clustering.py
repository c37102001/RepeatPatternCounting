import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt


def hierarchical_clustering(feature_list, img_name, para, edge_type, cut_method='elbow', do_draw=False):
    # hierarchically link cnt by order of distance from distance method 'ward'
    # Output a hierarchical tree as ppt. page 22.
    cnt_hierarchy = linkage(feature_list, 'ward')

    '''Combine two groups only when the group distance is smaller than the max__cut_distance.'''
    max_cut_distance = 0
    if cut_method == 'elbow':
        last = cnt_hierarchy[:, 2]
        last = [x for x in last if x > 0]

        '''distance of grooup distance'''
        acceleration = np.diff(last)

        # acceleration = map(abs, np.diff(acceleration) )

        # acceleration_rev = acceleration[::-1]
        # print 'acceleration:',acceleration

        if len(acceleration) < 2:
            return [0] * len(feature_list)
        avg_diff = sum(acceleration) / float(len(acceleration))
        tmp = acceleration[0]

        avg_list = [x for x in acceleration if x > avg_diff]
        avg_diff = sum(avg_list) / float(len(avg_list))

        '''
        5 Changeable, compute a ratio as a reference which decide the max_cut_distance (dynamic).
        Which the ratio is (its' own distance of group distance ) / (5 previous (if it exists ) distance of group distance average. )
        '''
        off_set = 5

        rario = []
        cut_point_list = []
        for i in range(1, len(acceleration)):

            if acceleration[i] > avg_diff:
                # cut_point_list.append( [ i, acceleration[i]/(tmp/float(i) ) ] )

                tmp_offset_prev = off_set
                prev = i - off_set
                if prev < 0:
                    prev = 0
                    tmp_offset_prev = i - prev
                rario.append(acceleration[i] / (sum(acceleration[prev:i]) / float(tmp_offset_prev)))
                cut_point_list.append([i, acceleration[i] / (sum(acceleration[prev:i]) / float(tmp_offset_prev))])
                # cut_point_list.append( [ i, acceleration[i] ] )
                # print 'i:',i+1,' ratio:',acceleration[i]/( sum(acceleration[n:i]) / float(off_set) )

            tmp += acceleration[i]

        if len(cut_point_list) < 1:
            print('all in one group!')
            return [0] * len(feature_list)

        cut_point_list.sort(key=lambda x: x[1], reverse=True)

        # print 'cut index:',cut_point_list[0][0]+1,' diff len:',len(acceleration)
        max_cut_distance = last[cut_point_list[0][0]]
        max_ratio = cut_point_list[0][1]

        if max_ratio < 2.0:
            print('all in one group! max_ratio:', max_ratio)
            return [0] * len(feature_list)

            # max_cut_distance = last[acceleration.argmax()]
    # elif cut_method == 'inconsistency':

    # plt.bar(left=range(len(rario)),height=rario)
    plt.bar(x=range(len(acceleration)), height=acceleration)
    plt.title(para + ' cut_point : ' + str(cut_point_list[0][0] + 1) + '  | value: ' + str(
        acceleration[cut_point_list[0][0]]) + ' | ratio: ' + str(max_ratio))

    # TODO fix do_draw
    if do_draw and False:
        plt.savefig(output_path + img_name + '_f_para[' + para + ']_his[' + str(edge_type) + '].png')
    plt.close()

    # print 'acceleration.argmax():',acceleration.argmax()
    clusters = fcluster(cnt_hierarchy, max_cut_distance, criterion='distance')
    return clusters