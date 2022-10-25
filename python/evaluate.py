import numpy as np


def compute_ap(ranks, nres):
    """
    Computes average precision for given ranked indexes.
    
    Arguments
    ---------
    ranks : zero-based ranks of positive images
    nres  : number of positive images
    
    Returns
    -------
    ap    : average precision
    """

    # number of images ranked by the system
    nimgranks = len(ranks)

    # accumulate trapezoids in PR-plot
    ap = 0

    recall_step = 1. / nres

    # 遍历排序列表
    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.
        else:
            # Pre@K
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.

    return ap


def compute_map(ranks, gnd, kappas=[]):
    """

    Parameters
    ----------
    ranks: GT_Num x Query_Num
    gnd: Query_Num
    kappas

    Returns
    -------

    """
    """
    Computes the mAP for a given set of returned results.

         Usage: 
           map = compute_map (ranks, gnd) 
                 computes mean average precsion (map) only
        
           map, aps, pr, prs = compute_map (ranks, gnd, kappas) 
                 computes mean average precision (map), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query
        
         Notes:
         1) ranks starts from 0, ranks.shape = db_size X #queries
         2) The junk results (e.g., the query itself) should be declared in the gnd stuct array
         3) If there are no positive images for some query, that query is excluded from the evaluation
    """

    map = 0.
    nq = len(gnd)  # number of queries
    # 每个查询对应一个AP
    aps = np.zeros(nq)
    # 计算AP@K
    pr = np.zeros(len(kappas))
    # 每个查询对应的AP@K
    prs = np.zeros((nq, len(kappas)))
    nempty = 0

    for i in np.arange(nq):
        # 正样本
        qgnd = np.array(gnd[i]['ok'])

        # 如果该查询图片没有对应的正样本，那么跳过本次查询
        # no positive images, skip from the average
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            continue

        # 干扰项
        try:
            qgndj = np.array(gnd[i]['junk'])
        except:
            qgndj = np.empty(0)

        # 判断正样本出现的下标位置，下标排名从小到大
        # sorted positions of positive and junk images (0 based)
        pos = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgnd)]
        # 判断干扰项出现的下标位置，下标排名从小到大
        junk = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgndj)]

        # -------------------------------------------------------
        # pos = [3, 4, 5]
        # junk = [7, 8, 9, 10]

        # ip = 0
        # ij = 0
        # pos[ip] > junk[ij] ? 3 > 7 break
        # k = 0
        # pos[ip] = pos[ip] - k = 3

        # ip = 1
        # pos[ip] > junk[ij] ? 4 > 7 break
        # k = 0
        # pos[ip] = pos[ip] - k = 4

        # ip = 2
        # pos[ip] > junk[ij] ? 5 > 7 break
        # k = 0
        # pos[ip] = pos[ip] - k = 5

        # pos = [3, 4, 5]

        # -------------------------------------------------------
        # pos = [3, 8, 11]
        # junk = [7, 9, 10]

        # ip = 0
        # ij = 0
        # pos[ip] > junk[ij] ? 3 > 7 break
        # k = 0
        # pos[ip] = pos[ip] - k = 3 - 0 = 3

        # ip = 1
        # pos[ip] > junk[ij] ? 8 > 7 continue
        # k = 1
        # ij = 1
        # pos[ip] > junk[ij] ? 8 > 9 break
        # k = 1
        # pos[ip] = pos[ip] - k = 8 - 1 = 7

        # ip = 2
        # pos[ip] > junk[ij] ? 11 > 9 continue
        # k = 2
        # ij = 2
        # pos[ip] > junk[ij] ? 11 > 10 continue
        # k = 3
        # ij = 3
        # ij < len(junk) > 3 < 3 break
        # k = 3
        # pos[ip] = pos[ip] - k = 11 - 3 = 8

        # junk = [7, 9, 10]
        # src_pos = [3, 8, 11]
        # dst_pos = [3, 7, 8]



        k = 0
        ij = 0
        # 如果存在干扰项
        if len(junk):
            # decrease positions of positives based on the number of
            # junk images appearing before them
            ip = 0
            # 判断正样本下标第ip个
            while (ip < len(pos)):
                # 遍历junk列表
                # 判断正样本下标是否大于干扰项下标，也就是如果该正样本排名低于干扰项排名，那么继续
                # 如果出现第ij个干扰项排名高于第ip个正样本，那么跳出
                while (ij < len(junk) and pos[ip] > junk[ij]):
                    # pos[ip]：第ip个正样本的排名
                    # junk[j]：第ij个干扰项的排名
                    # pos[ip] > junk[ij]: 正样本排序低于干扰项
                    # pos[ip] < junk[ij]: 正样本排序高于干扰项
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        # compute ap
        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap

        # compute precision @ k
        pos += 1  # get it to 1-based
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j]);
            prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]

    map = map / (nq - nempty)
    pr = pr / (nq - nempty)

    return map, aps, pr, prs
