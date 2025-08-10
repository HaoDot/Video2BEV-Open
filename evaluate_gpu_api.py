# -*- coding: utf-8 -*-
import scipy.io
import torch
import numpy as np
#import time
import os

#######################################################################
# Evaluate
def evaluate(qf,ql,gf,gl):
    """

    Args:
        qf: query feature
        ql: query label
        gf: gallery feature
        gl: gallery label

    Returns:

    """
    query = qf.view(-1,1)
    # print(query.shape)
    # gf:[class,c];query[c]
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index: 检索后的相似度从大到小排序
    index = np.argsort(score)  #from small to large
    index = index[::-1]

    # HAO‘S debug
    # print('estimated category:{}, right id is :{}th, query id:{}, gallery id:{}'.format(index[0],(np.argwhere((index==ql))[0,0]+1), ql, np.argwhere(gl == ql)[0, 0]))
    # if abs(index[0]-ql)<=2 and (index[0]!=ql):
    #     print('estimated category:{}, right id is :{}th, query id:{}, gallery id:{}'.format(index[0],(np.argwhere((index==ql))[0,0]+1), ql, np.argwhere(gl == ql)[0, 0]))
    if (index[0]!=ql):
        try:
            print('estimated category:{}, right id is :{}th, query id:{}, gallery id:{}'.format(index[0],(np.argwhere((index==ql))[0,0]+1), ql, np.argwhere(gl == ql)[0, 0]))
        except:
            print()
    acc=0
    if (index[0]== ql) and (ql == np.argwhere(gl == ql)[0, 0]):
        acc = 1
    else:
        acc=0
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl==ql)
    # # debug
    # if (index[0]+1) == ql or (index[0] -1) == ql:
    #     tmp = index[0]
    #     index[np.argwhere(index == ql)[0, 0]] = tmp
    #     index[0] = ql
    # query和gallery匹配上的索引
    good_index = query_index
    #print(good_index)
    #print(index[0:10])
    junk_index = np.argwhere(gl==-1)

    # index: 相似度索引(从高到低)
    # good_index: query和gallery匹配上的索引
    # junk_index: query和gallery没匹配上的索引
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp,acc


def compute_mAP(index, good_index, junk_index):
    # index: 相似度索引(从高到低)
    # good_index: query和gallery匹配上的索引
    # junk_index: query和gallery没匹配上的索引
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index in index
    # 匹配上的有多少个
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    # 匹配上的索引
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    # CMC曲线的计算方法就是，把每个query的AccK曲线相加，再除以query的总数，即平均AccK曲线。
    # CMC是个阶跃函数
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        # 只用匹配上的样本，只能用他来算precision和recall(https://wrong.wang/blog/20190223-reid%E4%BB%BB%E5%8A%A1%E4%B8%AD%E7%9A%84cmc%E5%92%8Cmap/image_5_hu7f41be7e562defe9f107615778106342_33264_1320x0_resize_q75_h2_lanczos_3.webp)
        # d_recall的意义是delta-recall,即recall_k - recall_{k-1}
        # 而recall = 匹配上的ID个数（len(good_index)）/该ID图片总数（定值）,所以delta_recall = 1/该ID图片总数（定值）
        d_recall = 1.0/ngood
        # precision = 检索到的同ID图片个数/查询结果总个数
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        # 用梯形面积之和算AP（P-R曲线下的面积）
        # precision
        # ^(old_precision)
        # | |\
        # | | \
        # | |  \(precision)
        # | |   |
        # | |   |
        # | |   |
        # --------------->recall
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc

######################################################################
def evaluate_root(query_name,gallery_name,output_path):
    current_dir = os.getcwd()  # 获取当前工作目录
    absolute_path = os.path.abspath(current_dir)  # 将相对路径转换为绝对路径
    result = scipy.io.loadmat(os.path.join(absolute_path,'pytorch_result.mat'))
    query_feature = torch.FloatTensor(result['query_f'])
    query_label = result['query_label'][0]
    gallery_feature = torch.FloatTensor(result['gallery_f'])
    gallery_label = result['gallery_label'][0]
    multi = os.path.isfile('multi_query.mat')

    if multi:
        m_result = scipy.io.loadmat('multi_query.mat')
        mquery_feature = torch.FloatTensor(m_result['mquery_f'])
        mquery_label = m_result['mquery_label'][0]
        mquery_feature = mquery_feature.cuda()

    query_feature = query_feature.cuda()
    gallery_feature = gallery_feature.cuda()

    with open(os.path.join(output_path,'result.txt'),'a+') as f:
        f.writelines(['query_name:{}\n'.format(query_name)])
        f.writelines(['gallery_name:{}\n'.format(gallery_name)])
        f.writelines(['query_shape:{}\n'.format(query_feature.shape)])
        f.writelines(['gallery_shape:{}\n'.format(gallery_feature.shape)])
    f.close()
    print(query_feature.shape)
    print(gallery_feature.shape)
    #print(gallery_feature[0,:])
    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    acc_all = 0.
    #print(query_label)
    for i in range(len(query_label)):
        # https://wrong.wang/blog/20190223-reid%E4%BB%BB%E5%8A%A1%E4%B8%AD%E7%9A%84cmc%E5%92%8Cmap/
        [ap_tmp, CMC_tmp], acc = evaluate(query_feature[i],query_label[i],gallery_feature,gallery_label)
        acc_all = acc_all + acc
        if CMC_tmp[0]==-1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
        #print(i, CMC_tmp[0])

    CMC = CMC.float()
    CMC = CMC/len(query_label) #average CMC
    print(round(len(gallery_label)*0.01))
    print('Recall@1:%.2f Recall@5:%.2f Recall@10:%.2f Recall@top1:%.2f AP:%.2f'%(CMC[0]*100,CMC[4]*100,CMC[9]*100, CMC[round(len(gallery_label)*0.01)]*100, ap/len(query_label)*100))
    with open(os.path.join(output_path,'result.txt'),'a+') as f:
        f.writelines(['{}\n'.format(round(len(gallery_label)*0.01))])
        f.writelines(['Recall@1:%.2f Recall@5:%.2f Recall@10:%.2f Recall@top1:%.2f AP:%.2f\n'%(CMC[0]*100,CMC[4]*100,CMC[9]*100, CMC[round(len(gallery_label)*0.01)]*100, ap/len(query_label)*100)])
        f.writelines(['ACC:{}%\n'.format(100*(acc_all/len(query_label)))])
    f.close()

# multiple-query evaluation is not used.
#CMC = torch.IntTensor(len(gallery_label)).zero_()
#ap = 0.0
#if multi:
#    for i in range(len(query_label)):
#        mquery_index1 = np.argwhere(mquery_label==query_label[i])
#        mquery_index2 = np.argwhere(mquery_cam==query_cam[i])
#        mquery_index =  np.intersect1d(mquery_index1, mquery_index2)
#        mq = torch.mean(mquery_feature[mquery_index,:], dim=0)
#        ap_tmp, CMC_tmp = evaluate(mq,query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
#        if CMC_tmp[0]==-1:
#            continue
#        CMC = CMC + CMC_tmp
#        ap += ap_tmp
#        #print(i, CMC_tmp[0])
#    CMC = CMC.float()
#    CMC = CMC/len(query_label) #average CMC
#    print('multi Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))

