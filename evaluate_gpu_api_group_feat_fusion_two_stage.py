# -*- coding: utf-8 -*-
import scipy.io
import torch
import numpy as np
#import time
import os
from PIL import Image
import torchvision

#######################################################################
def pad_color(src_tsr, color, pad_width = 4,gt=False):
    vertical_strip = torch.zeros_like(src_tsr[:, :pad_width, :])
    horizontal_strip = torch.zeros([pad_width, (pad_width * 2 + src_tsr.shape[1]), 3])
    if color:
        # green
        vertical_strip[:,:,1] = 255
        horizontal_strip[:,:,1] = 255
    else:
        if gt:
            # red
            vertical_strip[:, :, 2] = 255
            horizontal_strip[:, :, 2] = 255
        else:
            # red
            vertical_strip[:, :, 0] = 255
            horizontal_strip[:, :, 0] = 255
    src_tsr_ = torch.cat([vertical_strip,src_tsr,vertical_strip],dim=1)
    src_tsr_ = torch.cat([horizontal_strip, src_tsr_, horizontal_strip], dim=0)
    return src_tsr_

def visualization_img_q_g(query_path, gallery_path_l, matching_result, gt_abs_path):
    """
    visualization samples: query and gallery
    matched gallery sample : pad green
    missed gallery sample : pad red
    Args:
        query_path: should one sample
        gallery_path_l: should be a list of sample path
        matching_result：bool, denotes matched (True) or missed (False)

    Returns:

    """
    # query pad green
    q_tsr = torch.from_numpy(np.array(Image.open(query_path)))
    q_tsr_pad = pad_color(q_tsr, True)

    gallery_pad_l = []
    for (gallery_sample_path, matching_result_i) in zip(gallery_path_l, matching_result):
        g_tsr = torch.from_numpy(np.array(Image.open(gallery_sample_path)))
        if g_tsr.shape != q_tsr.shape:
            g_tsr = g_tsr.to(torch.float32).resize_as_(q_tsr.to(torch.float32))
        assert g_tsr.shape == q_tsr.shape
        if matching_result_i:
            # pad green
            g_tsr_pad = pad_color(g_tsr,matching_result_i)
        else:
            # pad red
            g_tsr_pad = pad_color(g_tsr,matching_result_i)
        if len(gallery_pad_l)!=0:
            assert g_tsr_pad.shape==gallery_pad_l[-1].shape
        gallery_pad_l.append(g_tsr_pad)

    gt_tsr = torch.from_numpy(np.array(Image.open(gt_abs_path)))
    # pad blue
    gt_tsr = pad_color(gt_tsr, False,gt=True)
    if gt_tsr.shape!=gallery_pad_l[-1].shape:
        gt_tsr = torch.from_numpy(np.array(torchvision.transforms.Resize(size=(gallery_pad_l[-1].shape[0],gallery_pad_l[-1].shape[1]))(Image.fromarray(np.uint8(gt_tsr)))))
    gallery_pad_l.append(gt_tsr)

    # list: query, zero, [gallery]
    white_ = torch.zeros_like(q_tsr_pad) + 255
    gallery_pad_l.insert(0,white_)
    gallery_pad_l.insert(0, q_tsr_pad)
    # torchvision
    v_img = torchvision.utils.make_grid((torch.stack(gallery_pad_l, dim=0)).permute(0,3,1,2), nrow=len(gallery_pad_l), padding=2, pad_value=255).permute(1,2,0)
    return v_img

# Evaluate
def evaluate(qf_l,ql,gf,gl, query_path_group, gallery_abs_path_l, cur_result_analysis_path,**kwargs):
    """

    Args:
        qf: query feature
        ql: query label
        gf: gallery feature
        gl: gallery label

    Returns:

    """
    # score_l = []
    # index_l = []
    # for qf in qf_l:
    #     query = qf.view(-1,1)
    #     # print(query.shape)
    #     # gf:[class,c];query[c]
    #     score_ = torch.mm(gf,query)
    #     # score_l.append(score_)
    #     score_l.append(
    #         (score_-torch.min(score_))/(torch.max(score_)-torch.min(score_)))
    #     print('current max is {}'.format(torch.max(score_)))
    #     # score_l.append((score_==torch.max(score_)).type(torch.float))
    # # score = torch.cat(score_l,dim=1).mean(dim=1)
    # score = torch.cat(score_l,dim=1).sum(dim=1)
    # score = (torch.sum(torch.softmax(gf @ qf_l.transpose(1, 0), dim=0), dim=1))
    model = kwargs.get('model')
    score = gf @ torch.mean(qf_l,dim=0)
    # score = (gf @ torch.mean(qf_l,dim=0))/model.logit_scale

    score = score.detach().cpu().numpy()

    # predict index: 检索后的相似度从大到小排序
    index = np.argsort(score)  #from small to large
    index = index[::-1]


    cur_result_analysis_path = cur_result_analysis_path + '_g_{:04d}'.format(index[0])
    # if not os.path.exists(cur_result_analysis_path):
    #     os.makedirs(cur_result_analysis_path)
    visualization_index = index[:5]
    matched_id = visualization_index == ql
    to_v_gallery_abs_path_l = []
    for v_idx in visualization_index:
        g_abs_path = gallery_abs_path_l[v_idx]
        #  select the most representative sample
        to_v_gallery_abs_path_l.append(g_abs_path)
    query_path = query_path_group[int((torch.argmax(qf_l,dim=0))[0].detach().cpu())]

    q_zero_g = visualization_img_q_g(query_path, to_v_gallery_abs_path_l, matched_id,
                                     gallery_abs_path_l[ql])

    sample_reid_name = os.path.basename(cur_result_analysis_path)
    for idx, g_id in enumerate(visualization_index):
        if idx == 0:
            continue
        sample_reid_name = sample_reid_name + '_{:04d}'.format(g_id)
    # Image.fromarray(np.uint8(q_zero_g)).convert('RGB').save(os.path.join(cur_result_analysis_path,'{}.jpg'.format(sample_reid_name)))
    if not os.path.exists(os.path.dirname(cur_result_analysis_path)):
        os.makedirs(os.path.dirname(cur_result_analysis_path))
    if (index[0] != ql):
        print('estimated category:{}, right id is :{}th, query id:{}, gallery id:{}'.format(index[0], (
                np.argwhere((index == ql))[0, 0] + 1), ql, np.argwhere(gl == ql)[0, 0]))
        Image.fromarray(np.uint8(q_zero_g)).convert('RGB').save(
            os.path.join(os.path.dirname(cur_result_analysis_path), '{}.jpg'.format(sample_reid_name)))
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


def evaluate_gallery_group(qf, ql, gf_l, gl, query_path, gallery_abs_path_group, cur_result_analysis_path,query_name,gallery_name,topk_two_stage,model,**kwargs):

    # # query = qf.view(-1, 1)
    # gf_number_l = []
    # for i in gf_l:
    #     gf_number_l.append(i.shape[0])
    # # print(query.shape)
    # # score = (torch.sum((torch.cat(gf_l,dim=0) @ qf), dim=1))
    # score_l = torch.split((torch.cat(gf_l, dim=0) @ qf), split_size_or_sections=gf_number_l)
    # tmp_l = []
    # for score_group in score_l:
    #     # voting_score = (torch.sum(torch.softmax(score_group, dim=0), dim=0))
    #     voting_score = (torch.sum(score_group, dim=0))
    #     tmp_l.append((voting_score).unsqueeze(0))#.unsqueeze(1)
    #
    # score = torch.cat(tmp_l,dim=0).cpu()#.unsqueeze(0)
    tmp_l = []
    # 因为 video的帧数不一样，不能统一的取mean
    for gf_ in gf_l:
        tmp_l.append(torch.mean(gf_,dim=0))
    # scalar: 就一个数
    score = torch.stack(tmp_l) @ qf

    # score = (torch.stack(tmp_l) @ qf)/model.logit_scale

    score = score.detach().cpu().numpy()
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]

    # good index
    query_index = np.argwhere(gl == ql)
    good_index = query_index

    cur_result_analysis_path = cur_result_analysis_path+'_g_{:04d}'.format(gl[index[0]])
    # if not os.path.exists(cur_result_analysis_path):
    #     os.makedirs(cur_result_analysis_path)
    visualization_index = index[:5]
    # tmp = []
    matched_id = [gl[i] for i in visualization_index] == ql
    to_v_gallery_abs_path_l = []
    for v_idx in visualization_index:
        # v_idx = gl[v_idx_]
        g_abs_path_l = gallery_abs_path_group[v_idx]
        #  select the most representative sample
        g_v_abs_path = g_abs_path_l[int((torch.argmax(gf_l[v_idx],dim=0))[0].detach().cpu())]
        to_v_gallery_abs_path_l.append(g_v_abs_path)

    if ql not in gl:
        print('query label {} is not in gallery label'.format(ql))
        return [-1], [-1]
    gt_idx_v = gl.index(ql)
    q_zero_g = visualization_img_q_g(query_path, to_v_gallery_abs_path_l, matched_id,
                                     gallery_abs_path_group[gt_idx_v][0])

    sample_reid_name = os.path.basename(cur_result_analysis_path)
    for idx, g_id in enumerate(visualization_index):
        if idx==0:
            continue
        sample_reid_name = sample_reid_name+'_{:04d}'.format(g_id)
    # Image.fromarray(np.uint8(q_zero_g)).convert('RGB').save(os.path.join(cur_result_analysis_path,'{}.jpg'.format(sample_reid_name)))
    if (gl[int(index[0])] != ql):
        print('estimated category:{}, right id is :{}th, query id:{}, gallery id:{}'.format(index[0], (
                np.argwhere(([gl[i] for i in index] == ql))[0, 0] + 1), ql, gl[np.argwhere(gl == ql)[0, 0]]))
        Image.fromarray(np.uint8(q_zero_g)).convert('RGB').save(
            os.path.join(os.path.dirname(cur_result_analysis_path), '{}.jpg'.format(sample_reid_name)))

    # print(good_index)
    # print(index[0:10])
    junk_index = np.argwhere(gl == -1)

    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


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

def compose_query_group(query_feature,query_label,query_img_path_l,gallery_feature,gallery_label,gallery_img_path_l):
    """

    Args:
        query_feature:
        query_label:
        query_img_path_l:
        gallery_feature:
        gallery_label:
        gallery_img_path_l:

    Returns:

    """
    query_name_d = {}
    # find out how many images are in one class
    for query_img_path in query_img_path_l:
        query_name = query_img_path.split('/')[-2]
        if query_name not in query_name_d.keys():
            query_name_d[query_name] = 1
        else:
            query_name_d[query_name] += 1
    # index all the features in one class and collect them in list
    query_feature_group = []
    query_label_group = []
    query_abs_path_group = []
    for i in range(len(sorted(list(query_name_d.keys())))):
        if (i == 0):
            start_idx = 0
            # current_query_name = sorted(list(query_name_d.keys()))[i]
        else:
            start_idx += query_name_d[sorted(list(query_name_d.keys()))[i-1]]
        # current_query_name = sorted(list(query_name_d.keys()))[i]
        query_label_group.append(query_label[start_idx])
        query_feature_group.append(query_feature[start_idx : start_idx+query_name_d[sorted(list(query_name_d.keys()))[i]]])
        query_abs_path_group.append(query_img_path_l[start_idx : start_idx+query_name_d[sorted(list(query_name_d.keys()))[i]]])
        assert len(query_feature_group[-1]) == query_name_d[sorted(list(query_name_d.keys()))[i]]
        assert len(query_feature_group[-1]) == len(query_abs_path_group[-1])
    return query_feature_group, query_label_group, query_abs_path_group

def compose_gallery_group(query_feature,query_label,query_img_path_l,gallery_feature,gallery_label,gallery_img_path_l):
    """

    Args:
        query_feature:
        query_label:
        query_img_path_l:
        gallery_feature:
        gallery_label:
        gallery_img_path_l:

    Returns:

    """
    gallery_name_d = {}
    # find out how many images are in one class
    for gallery_img_path in gallery_img_path_l:
        gallery_name = gallery_img_path.split('/')[-2]
        if gallery_name not in gallery_name_d.keys():
            gallery_name_d[gallery_name] = 1
        else:
            gallery_name_d[gallery_name] += 1
    # index all the features in one class and collect them in list
    gallery_feature_group = []
    gallery_label_group = []
    gallery_abs_path_group = []
    for i in range(len(sorted(list(gallery_name_d.keys())))):
        if i ==0:
            start_idx = 0
        else:
            start_idx += gallery_name_d[sorted(list(gallery_name_d.keys()))[i-1]]
        gallery_label_group.append(gallery_label[start_idx])
        gallery_feature_group.append(gallery_feature[start_idx : start_idx+gallery_name_d[sorted(list(gallery_name_d.keys()))[i]]])
        assert len(gallery_feature_group[-1]) == gallery_name_d[sorted(list(gallery_name_d.keys()))[i]]
        gallery_abs_path_group.append(
            gallery_img_path_l[start_idx : start_idx+gallery_name_d[sorted(list(gallery_name_d.keys()))[i]]])
        assert len(gallery_feature_group[-1]) == len(gallery_abs_path_group[-1])
    return gallery_feature_group, gallery_label_group, gallery_abs_path_group

######################################################################
def evaluate_root(query_name,query_img_path_l,gallery_name,gallery_img_path_l,output_path,topk_two_stage,model,opt,result_analysis=True):
    current_dir = os.getcwd()  # 获取当前工作目录
    absolute_path = os.path.abspath(current_dir)  # 将相对路径转换为绝对路径

    import h5py, time
    since = time.time()
    if opt.itm:
        with h5py.File(os.path.join(absolute_path, 'pytorch_result.h5'), 'r') as h5file:
            # result = scipy.io.loadmat(os.path.join(absolute_path, 'pytorch_result.mat'))
            # query_feature = torch.from_numpy(result['query_f'])
            # gallery_feature = torch.from_numpy(result['gallery_f'])
            # if opt.itm:
            #     query_feature = torch.from_numpy(np.array(h5file['query_f_full_seq'])[:, 0, :])
            #     query_feature = query_feature / query_feature.norm(dim=-1, keepdim=True)
            #     gallery_feature = torch.from_numpy(np.array(h5file['gallery_f_full_seq'])[:, 0, :])
            #     gallery_feature = gallery_feature / gallery_feature.norm(dim=-1, keepdim=True)
            # else:
            #     result = scipy.io.loadmat(os.path.join(absolute_path, 'pytorch_result.mat'))
            #     query_feature = torch.from_numpy(result['query_f'])
            #     gallery_feature = torch.from_numpy(result['gallery_f'])
            # fnorm = torch.norm(query_feature, p=2, dim=1, keepdim=True)
            # query_feature = query_feature.div(fnorm.expand_as(query_feature))
            query_feature_full_seq = torch.from_numpy(np.array(h5file['query_f_full_seq']))

            # fnorm = torch.norm(gallery_feature, p=2, dim=1, keepdim=True)
            # gallery_feature = gallery_feature.div(fnorm.expand_as(gallery_feature))
            gallery_feature_full_seq = torch.from_numpy(np.array(h5file['gallery_f_full_seq']))
    else:
        # result = scipy.io.loadmat(os.path.join(absolute_path, 'pytorch_result_{}_{}.mat'.format(query_name.split('_')[-1],gallery_name.split('_')[-1])))
        result = scipy.io.loadmat(os.path.join(absolute_path, 'pytorch_result.mat'))
        query_feature = torch.from_numpy(result['query_f'])
        gallery_feature = torch.from_numpy(result['gallery_f'])
        query_feature_full_seq = query_feature
        gallery_feature_full_seq = gallery_feature
    # npz_result = np.load(os.path.join(absolute_path, 'pytorch_result.npz'))
    result = scipy.io.loadmat(os.path.join(absolute_path, 'pytorch_result.mat'))
    # result = scipy.io.loadmat(os.path.join(absolute_path, 'pytorch_result_{}_{}.mat'.format(query_name.split('_')[-1],gallery_name.split('_')[-1])))
    query_feature = torch.from_numpy(result['query_f'])
    gallery_feature = torch.from_numpy(result['gallery_f'])

    query_label = result['query_label'][0]

    gallery_label = result['gallery_label'][0]
    time_elapsed = time.time() - since
    print('Loading data from disk in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    multi = os.path.isfile('multi_query.mat')

    if result_analysis:
        if not os.path.exists(os.path.join(output_path,'result_analysis_{}_{}'.format(query_name,gallery_name))):
            os.makedirs(os.path.join(output_path,'result_analysis_{}_{}'.format(query_name,gallery_name)))
        result_analysis_path = os.path.join(os.path.join(output_path,'result_analysis_{}_{}'.format(query_name,gallery_name)))
    if multi:
        m_result = scipy.io.loadmat('multi_query.mat')
        mquery_feature = torch.FloatTensor(m_result['mquery_f'])
        mquery_label = m_result['mquery_label'][0]
        mquery_feature = mquery_feature.cuda()

    query_feature = query_feature.cuda()
    gallery_feature = gallery_feature.cuda()

    with open(os.path.join(output_path,'result.txt'),'a+') as f:
        print('\n')
        f.writelines(['query_name:{}\n'.format(query_name)])
        f.writelines(['gallery_name:{}\n'.format(gallery_name)])
        f.writelines(['query_shape:{}\n'.format(query_feature.shape)])
        f.writelines(['gallery_shape:{}\n'.format(gallery_feature.shape)])
    f.close()
    print(query_feature.shape)
    print(gallery_feature.shape)
    #print(gallery_feature[0,:])

    #print(query_label)
    if 'bev' in query_name:
        CMC = torch.IntTensor(len(gallery_label)).zero_()
        ap = 0.0
        acc_all = 0.
        query_feature_group, query_label_group, query_abs_path_group = compose_query_group(query_feature,query_label,query_img_path_l,gallery_feature,gallery_label,gallery_img_path_l)
        if opt.itm:
            query_feature_full_seq_group, query_label_group, query_abs_path_group = compose_query_group(query_feature_full_seq,query_label,query_img_path_l,
                                                                                                    gallery_feature_full_seq,gallery_label,gallery_img_path_l)
        else:
            query_feature_full_seq_group = query_feature_group
        for i in range(len(query_label_group)):
            # https://wrong.wang/blog/20190223-reid%E4%BB%BB%E5%8A%A1%E4%B8%AD%E7%9A%84cmc%E5%92%8Cmap/
            query_path = query_abs_path_group[i][0]
            query_name = os.path.dirname(query_path).split('/')[-2]
            cur_result_analysis_path = os.path.join(result_analysis_path, 'q_{}'.format(query_name))

            [ap_tmp, CMC_tmp], acc = evaluate(qf_l = query_feature_group[i], ql = query_label_group[i], gf = gallery_feature,
                                              gl = gallery_label, query_path_group = query_abs_path_group[i], gallery_abs_path_l = gallery_img_path_l, cur_result_analysis_path = cur_result_analysis_path,
                                              topk_two_stage=topk_two_stage,query_name=query_name, query_feature_full_seq=query_feature_full_seq_group[i],gallery_feature_full_seq=gallery_feature_full_seq,model=model)
            acc_all = acc_all + acc
            if CMC_tmp[0] == -1:
                continue
            CMC = CMC + CMC_tmp
            ap += ap_tmp
            # print(i, CMC_tmp[0])

        CMC = CMC.float()
        CMC = CMC / len(query_label_group)  # average CMC
        print(round(len(gallery_label) * 0.01))
        print('Recall@1:%.2f Recall@5:%.2f Recall@10:%.2f Recall@top1:%.2f AP:%.2f' % (
        CMC[0] * 100, CMC[4] * 100, CMC[9] * 100, CMC[round(len(gallery_label) * 0.01)] * 100,
        ap / len(query_label_group) * 100))
        with open(os.path.join(output_path, 'result.txt'), 'a+') as f:
            f.writelines(['{}\n'.format(round(len(gallery_label) * 0.01))])
            f.writelines(['Recall@1:%.2f Recall@5:%.2f Recall@10:%.2f Recall@top1:%.2f AP:%.2f\n' % (
            CMC[0] * 100, CMC[4] * 100, CMC[9] * 100, CMC[round(len(gallery_label) * 0.01)] * 100,
            ap / len(query_label_group) * 100)])
            f.writelines(['ACC:{}%\n'.format(100 * (acc_all / len(query_label)))])
        f.close()
    elif 'bev' in gallery_name:
        gallery_feature_group, gallery_label_group, gallery_abs_path_group = compose_gallery_group(query_feature, query_label, query_img_path_l,
                                                                     gallery_feature, gallery_label, gallery_img_path_l)
        gallery_feature_full_seq_group, _, _ = compose_gallery_group(query_feature,
                                                                                                   query_label,
                                                                                                   query_img_path_l,
                                                                                                   gallery_feature_full_seq,
                                                                                                   gallery_label,
                                                                                                   gallery_img_path_l)
        CMC = torch.IntTensor(len(gallery_label_group)).zero_()
        ap = 0.0
        acc_all = 0.
        for i in range(len(query_label)):
            # if i == 338:
            #     pass
            query_path = query_img_path_l[i]
            if query_label[i] == int((query_path).split('/')[-2]):
                pass
            else:
                raise Exception('qurey_label != query img name')
            query_name = os.path.dirname(query_path).split('/')[-1]
            cur_result_analysis_path = os.path.join(result_analysis_path,'q_{}'.format(query_name))
            # , , , , , , , , model
            ap_tmp, CMC_tmp = evaluate_gallery_group(qf = query_feature[i], ql = query_label[i], gf_l = gallery_feature_group,
                                                     gl = gallery_label_group, query_path = query_path, gallery_abs_path_group = gallery_abs_path_group,
                                                     cur_result_analysis_path = cur_result_analysis_path,query_name = query_name,gallery_name = gallery_name,topk_two_stage = topk_two_stage,
                                                     model= model,query_feature_full_seq=query_feature_full_seq[i],gallery_feature_full_seq=gallery_feature_full_seq_group)
            if CMC_tmp[0] == -1:
                continue
            CMC = CMC + CMC_tmp
            ap += ap_tmp
            # print(i, CMC_tmp[0])

        CMC = CMC.float()
        CMC = CMC / len(query_label)  # average CMC
        print(round(len(gallery_label) * 0.01))
        print('Recall@1:%.2f Recall@5:%.2f Recall@10:%.2f Recall@top1:%.2f AP:%.2f' % (
        CMC[0] * 100, CMC[4] * 100, CMC[9] * 100, CMC[round(len(gallery_label) * 0.01)] * 100,
        ap / len(query_label) * 100))
        with open(os.path.join(output_path, 'result.txt'), 'a+') as f:
            f.writelines(['{}\n'.format(round(len(gallery_label) * 0.01))])
            f.writelines(['Recall@1:%.2f Recall@5:%.2f Recall@10:%.2f Recall@top1:%.2f AP:%.2f\n' % (
            CMC[0] * 100, CMC[4] * 100, CMC[9] * 100, CMC[round(len(gallery_label) * 0.01)] * 100,
            ap / len(query_label) * 100)])
            f.writelines(['ACC:{}%\n'.format(100 * (acc_all / len(query_label)))])
        f.close()
    elif 'drone' in query_name:
        CMC = torch.IntTensor(len(gallery_label)).zero_()
        ap = 0.0
        acc_all = 0.
        query_feature_group, query_label_group, query_abs_path_group = compose_query_group(query_feature,query_label,query_img_path_l,gallery_feature,gallery_label,gallery_img_path_l)
        for i in range(len(query_label_group)):
            # https://wrong.wang/blog/20190223-reid%E4%BB%BB%E5%8A%A1%E4%B8%AD%E7%9A%84cmc%E5%92%8Cmap/
            query_path = query_abs_path_group[i][0]
            query_name = os.path.dirname(query_path).split('/')[-1]
            cur_result_analysis_path = os.path.join(result_analysis_path, 'q_{}'.format(query_name))
            [ap_tmp, CMC_tmp], acc = evaluate(query_feature_group[i], query_label_group[i], gallery_feature,
                                              gallery_label, query_abs_path_group[i], gallery_img_path_l, cur_result_analysis_path)
            acc_all = acc_all + acc
            if CMC_tmp[0] == -1:
                continue
            CMC = CMC + CMC_tmp
            ap += ap_tmp
            # print(i, CMC_tmp[0])

        CMC = CMC.float()
        CMC = CMC / len(query_label_group)  # average CMC
        print(round(len(gallery_label) * 0.01))
        print('Recall@1:%.2f Recall@5:%.2f Recall@10:%.2f Recall@top1:%.2f AP:%.2f' % (
        CMC[0] * 100, CMC[4] * 100, CMC[9] * 100, CMC[round(len(gallery_label) * 0.01)] * 100,
        ap / len(query_label_group) * 100))
        with open(os.path.join(output_path, 'result.txt'), 'a+') as f:
            f.writelines(['{}\n'.format(round(len(gallery_label) * 0.01))])
            f.writelines(['Recall@1:%.2f Recall@5:%.2f Recall@10:%.2f Recall@top1:%.2f AP:%.2f\n' % (
            CMC[0] * 100, CMC[4] * 100, CMC[9] * 100, CMC[round(len(gallery_label) * 0.01)] * 100,
            ap / len(query_label_group) * 100)])
            f.writelines(['ACC:{}%\n'.format(100 * (acc_all / len(query_label)))])
        f.close()
    elif 'drone' in gallery_name:
        gallery_feature_group, gallery_label_group, gallery_abs_path_group = compose_gallery_group(query_feature, query_label, query_img_path_l,
                                                                     gallery_feature, gallery_label, gallery_img_path_l)
        CMC = torch.IntTensor(len(gallery_label_group)).zero_()
        ap = 0.0
        acc_all = 0.
        for i in range(len(query_label)):
            query_path = query_img_path_l[i]
            query_name = os.path.dirname(query_path).split('/')[-1]
            cur_result_analysis_path = os.path.join(result_analysis_path,'q_{}'.format(query_name))
            ap_tmp, CMC_tmp = evaluate_gallery_group(query_feature[i], query_label[i], gallery_feature_group, gallery_label_group, query_path, gallery_abs_path_group, cur_result_analysis_path)
            if CMC_tmp[0] == -1:
                continue
            CMC = CMC + CMC_tmp
            ap += ap_tmp
            # print(i, CMC_tmp[0])

        CMC = CMC.float()
        CMC = CMC / len(query_label)  # average CMC
        print(round(len(gallery_label) * 0.01))
        print('Recall@1:%.2f Recall@5:%.2f Recall@10:%.2f Recall@top1:%.2f AP:%.2f' % (
        CMC[0] * 100, CMC[4] * 100, CMC[9] * 100, CMC[round(len(gallery_label) * 0.01)] * 100,
        ap / len(query_label) * 100))
        with open(os.path.join(output_path, 'result.txt'), 'a+') as f:
            f.writelines(['{}\n'.format(round(len(gallery_label) * 0.01))])
            f.writelines(['Recall@1:%.2f Recall@5:%.2f Recall@10:%.2f Recall@top1:%.2f AP:%.2f\n' % (
            CMC[0] * 100, CMC[4] * 100, CMC[9] * 100, CMC[round(len(gallery_label) * 0.01)] * 100,
            ap / len(query_label) * 100)])
            f.writelines(['ACC:{}%\n'.format(100 * (acc_all / len(query_label)))])
        f.close()

    else:
        raise Exception('only support bev and drone group')

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

