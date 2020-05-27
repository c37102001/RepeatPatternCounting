import os
import math

ANSWER_PATH = 'answer_csv/'

SOURCE_PATH = 'colony_repeat_pattern_csv/'


def countDistance(p1, p2):
    return math.sqrt(
        math.pow(int(p1[0]) - int(p2[0]), 2) + math.pow(int(p1[1]) - int(p2[1]), 2) + math.pow(int(p1[2]) - int(p2[2]),
                                                                                               2))


def countPrecision(truePositive, falsePositive):
    if truePositive == 0:
        return 0.0
    return float(truePositive) / float(truePositive + falsePositive)


def countRecall(truePositive, falseNegative):
    if truePositive == 0:
        return 0.0
    return float(truePositive) / float(truePositive + falseNegative)


def countFmeasure(precision, recall):
    if precision * recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def countRandIndex(truePositive, trueNegative, falsePositive, falseNegative):
    if float(truePositive + trueNegative) == 0:
        return 0.0
    return float(truePositive + trueNegative) / float(truePositive + trueNegative + falsePositive + falseNegative)


def saveResult(resultList):
    f = open("repeat_pattern_clustering_result.csv", 'w+')
    f.write("FileName,TruePositive,TrueNegaitive,FalsePositive,FalseNegative,RandIndex,Precision,Recall,F-measure\n")
    for result in resultList:
        f.write("%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (
        result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7], result[8]))
    f.close


def get_lst(fn):
    ans = []
    f = open(fn, 'r')
    gno = 0
    for l in f:
        # l = Group,Y,X
        # l = 0,230,30
        tmp = l.split(',')
        if tmp[0] == 'Group' or len(tmp) == 0:
            continue
        else:
            ans.append([int(tmp[0]), int(tmp[1]), int(tmp[2])])
        # check group 數
        if int(tmp[0]) > gno:
            gno = int(tmp[0])
    f.close()

    return ans, gno


def get_cp_lst(fn):
    rst_lst, gno = get_lst(fn)
    rst_lst_sub = rst_lst[1:]
    # rst_lst = [('0', '65', '291\n'), ('1', '77', '103\n'), ('0', '130', '163\n'), ...]
    ans = []
    for k in rst_lst:
        if len(rst_lst_sub) == 0:
            break

        for ks in rst_lst_sub:
            ans.append([k, ks])
        del rst_lst_sub[0]
    return ans


def get_corr_ans(gyx_input):
    global ans_lst
    tmp = []
    # print 46, len(ans_lst)
    for gyx in ans_lst:
        distance = countDistance([gyx[1], gyx[2], 0], [gyx_input[1], gyx_input[2], 0])
        if distance < 10:
            tmp.append([distance, gyx])
        # distance, gyx = (5.0, (0, 232, 136))
    # 沒對到的給例外 group
    if len(tmp) == 0:
        tmp = [100, ans_lst[-1]]
    # 一個結果
    elif len(tmp) == 1:
        tmp = tmp[0]
    # 多個結果排序取 distance 最小的
    elif len(tmp) > 1:
        # print 59,tmp
        # raw_input()
        tmp.sort(lambda x, y: cmp(x[0], y[0]))
        tmp = tmp[0]
    return tmp


def get_contang_para(ans1, ans2, prd1, prd2):
    # print ans1, ans2
    # [2, -1, -1] [1, 77, 106]
    # print prd1, prd2
    # [0, 65, 291] [1, 77, 103]

    tp = tn = fp = fn = 0
    if ans1[0] == ans2[0] and prd1[0] == prd2[0]:
        tp += 1
    elif ans1[0] != ans2[0] and prd1[0] != prd2[0]:
        tn += 1
    elif ans1[0] != ans2[0] and prd1[0] == prd2[0]:
        fp += 1
    elif ans1[0] == ans2[0] and prd1[0] != prd2[0]:
        fn += 1
    return tp, tn, fp, fn


def grouping_ans_prd_and_contang(f_ans, f_prd):
    global ans_lst
    ans_lst, gno = get_lst(f_ans)
    ans_lst.append([gno + 1, -1, -1])
    # (gno+1, -1, -1) 對不到的例外 group
    # ans_lst = [(0, 65, 291), (1, 77, 103), (), ..., (2, -1, -1)]
    print('answer set:', len(ans_lst), '(with one exception group)')
    #	print ans_lst

    prd_lst = get_cp_lst(f_prd)
    # prd_lst = [((0, 65, 291), (1, 77, 103)), ((), ()),...]
    print('check set:', len(prd_lst))
    #	print prd_lst

    contang = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    for k in prd_lst:
        # k = ((0, 65, 291), (1, 77, 103))
        p1s = get_corr_ans(k[0])
        p1 = p1s[1]  # [2, -1, -1]

        p2s = get_corr_ans(k[1])
        p2 = p2s[1]  # [0, 232, 136]

        # print 101, k[0], 'to', p1s # [100, [2, -1, -1]]
        # print 105, k[1], 'to', p2s # [5.0, [0, 232, 136]]

        ## grouping down ##
        # (p1, k[0])
        # (p2, k[1])

        ## get contang
        tp, tn, fp, fn = get_contang_para(p1, p2, k[0], k[1])
        contang['tp'] += tp
        contang['tn'] += tn
        contang['fp'] += fp
        contang['fn'] += fn

    #		print 'tp', contang['tp'], 'tn', contang['tn'], 'fp', contang['fp'], 'fn', contang['fn']
    #		raw_input()
    return contang


def evaluate(tp, tn, fp, fn):
    randidx = countRandIndex(tp, tn, fp, fn)
    precision = countPrecision(tp, fp)
    recall = countRecall(tp, fn)
    fmeasure = countFmeasure(precision, recall)
    return randidx, precision, recall, fmeasure


def main():
    pth1 = ANSWER_PATH
    pth2 = SOURCE_PATH
    l1 = os.listdir(pth1)
    l2 = os.listdir(pth2)
    # 取得 answer file (fn1) and compare file (fn2)
    resultList = []
    for k in l1:
        # k = 'DL147-3B.png.csv'
        # 		if k != 'IMG_1027.png.csv':
        # 			continue

        fn1 = pth1 + k

        tmp = k.split('.')[0]
        # tmp = 'DL147-3B'
        filename = tmp
        fn2 = pth2 + '%s.csv' % tmp
        # colonycounter_DL147-3B.csv
        print(fn1)  # filename 1
        print(fn2)  # filename 2

        contang = grouping_ans_prd_and_contang(fn1, fn2)
        print('tp', contang['tp'], 'tn', contang['tn'], 'fp', contang['fp'], 'fn', contang['fn'])
        ri, prec, reca, fmea = evaluate(contang['tp'], contang['tn'], contang['fp'], contang['fn'])
        print('precision: %f\nrecall: %f\nfmeasure: %f\n' % (prec, reca, fmea))

        resultList.append([filename, contang['tp'], contang['tn'], contang['fp'], contang['fn'], ri, prec, reca, fmea])

    		# raw_input()

    saveResult(resultList)


if __name__ == '__main__':
    main()