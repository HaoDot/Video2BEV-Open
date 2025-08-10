# -*- coding: utf-8 -*-
import scipy.io
import torch
import numpy as np
# import time
import os
from PIL import Image
import torchvision
import cv2
import copy
import torchvision.transforms as transforms
import torch.nn.functional as F

#######################################################################
def pad_color(src_tsr, color, pad_width=4, gt=False):
    vertical_strip = torch.zeros_like(src_tsr[:, :pad_width, :])
    horizontal_strip = torch.zeros([pad_width, (pad_width * 2 + src_tsr.shape[1]), 3])
    if color:
        # green
        vertical_strip[:, :, 1] = 255
        horizontal_strip[:, :, 1] = 255
    elif color==None:
        src_tsr_ = torch.cat([vertical_strip, src_tsr, vertical_strip], dim=1)
        src_tsr_ = torch.cat([horizontal_strip, src_tsr_, horizontal_strip], dim=0)


        src_tsr_ = F.interpolate(src_tsr.permute(2, 0, 1).unsqueeze(0).float(), size=(src_tsr_.shape[0],src_tsr_.shape[0]), mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)
        return src_tsr_
    else:
        if gt:
            # blue
            vertical_strip[:, :, 2] = 128
            vertical_strip[:, :, 0] = 128
            horizontal_strip[:, :, 2] = 128
            horizontal_strip[:, :, 0] = 128
        else:
            # red
            vertical_strip[:, :, 2] = 255
            horizontal_strip[:, :, 2] = 255
    src_tsr_ = torch.cat([vertical_strip, src_tsr, vertical_strip], dim=1)
    src_tsr_ = torch.cat([horizontal_strip, src_tsr_, horizontal_strip], dim=0)
    return src_tsr_


def add_text(image, mssg):
    # image = cv2.putText(image, "{:02f}".format(float(mssg)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
    image = cv2.putText(image, "{:02f}".format(float(mssg)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.1, (255, 0, 0), 4, cv2.LINE_AA)
    return image


def visualization_img_q_g(query_path, gallery_path_l, matching_result, gt_abs_path, score_topk, g_equal_q_score):
    """
    visualization samples: query and gallery
    matched gallery sample : pad green
    missed gallery sample : pad red
    Args:
        query_path: should one sample
        gallery_path_l: should be a list of sample path
        matching_result：bool, denotes matched (True) or missed (False)
        gt_abs_path:
        score_topk:
        g_equal_q_score:

    Returns:

    """
    # query pad green
    q_tsr = torch.from_numpy(np.array(Image.open(query_path).resize((512,512))))
    q_tsr_pad = pad_color(q_tsr, None)
    # q_tsr_pad = q_tsr

    gallery_pad_l = []
    for idx, (gallery_sample_path, matching_result_i) in enumerate(zip(gallery_path_l, matching_result)):
        # g_tsr = torch.from_numpy(add_text(np.array(Image.open(gallery_sample_path).resize((512,512))), score_topk[idx]))
        g_tsr = torch.from_numpy((np.array(Image.open(gallery_sample_path).resize((512,512)))))
        if g_tsr.shape != q_tsr.shape:
            g_tsr = g_tsr.to(torch.float32).resize_as_(q_tsr.to(torch.float32))
        assert g_tsr.shape == q_tsr.shape
        if matching_result_i:
            # pad green
            g_tsr_pad = pad_color(g_tsr, matching_result_i)
        else:
            # pad red
            g_tsr_pad = pad_color(g_tsr, matching_result_i)
        if len(gallery_pad_l) != 0:
            assert g_tsr_pad.shape == gallery_pad_l[-1].shape
        gallery_pad_l.append(g_tsr_pad)

    # gt_tsr = torch.from_numpy(add_text(np.array(Image.open(gt_abs_path).resize((512,512))), '{}'.format(g_equal_q_score)))
    gt_tsr = torch.from_numpy((np.array(Image.open(gt_abs_path).resize((512,512)))))
    # pad blue
    gt_tsr = pad_color(gt_tsr, False, gt=True)
    if gt_tsr.shape != gallery_pad_l[-1].shape:
        gt_tsr = torch.from_numpy(np.array(
            torchvision.transforms.Resize(size=(gallery_pad_l[-1].shape[0], gallery_pad_l[-1].shape[1]))(
                Image.fromarray(np.uint8(gt_tsr)))))
    gallery_pad_l.append(gt_tsr)

    # list: query, zero, [gallery]
    white_ = torch.zeros_like(q_tsr_pad) + 255
    gallery_pad_l.insert(0, white_)
    gallery_pad_l.insert(0, q_tsr_pad)
    # torchvision
    v_img = torchvision.utils.make_grid((torch.stack(gallery_pad_l, dim=0)).permute(0, 3, 1, 2),
                                        nrow=len(gallery_pad_l), padding=2, pad_value=255).permute(1, 2, 0)
    return v_img


# Evaluate
def evaluate(qf_l, ql, gf, gl, query_path_group, gallery_abs_path_l, cur_result_analysis_path, **kwargs):
    """

    Args:
        qf: query feature
        ql: query label
        gf: gallery feature
        gl: gallery label

    Returns:

    """


    txt_path = kwargs.get('txt_path')
    score = gf @ torch.mean(qf_l, dim=0)
    # # score = score.squeeze(1).cpu()
    score = score.cpu().numpy()
    # score = torch.nn.functional.softmax(score).cpu().numpy()
    # predict index: 检索后的相似度从大到小排序
    index = np.argsort(score)  # from small to large
    index = index[::-1]


    ##################################################
    if not kwargs.get('first_stage_only'):
        # TODO: select Top-K (topk_sim=score[K-idx])
        topk_two_stage = kwargs['topk_two_stage']
        index_topk = index[:topk_two_stage]
        # # TODO: ITM matching (cls token) result (score)
        # # NOTE: ITM的head是两个纬度的，这里只care 第二个纬度 (https://github.com/salesforce/BLIP/blob/3a29b7410476bf5f2ba0955827390eb6ea1f4f9d/train_retrieval.py#L154C9-L154C14)
        # # TODO: fusion head 是只为drone(bev)设计的，注意这里选择gallery or query是drone(bev)
        if 'bev' in kwargs['query_name'] or 'drone' in kwargs['query_name']:
            # satellite:KV, drone or bev:Q
            query_feature_full_seq, gallery_feature_full_seq = \
                kwargs['query_feature_full_seq'], kwargs['gallery_feature_full_seq']




            # TODO: use individual image in the video, then average the whole scores of itm head
            # satellite; 无需分组
            gallery_feature_top_k = (gallery_feature_full_seq[list(index_topk), :])
            score_2nd_l = []
            for frame_idx in range(len(query_feature_full_seq)):
                frame_query_feature_full_seq = query_feature_full_seq[frame_idx].unsqueeze(0)
                if len(frame_query_feature_full_seq.shape)==4:
                    frame_query_feature_full_seq = frame_query_feature_full_seq.repeat(topk_two_stage,1, 1, 1)
                else:
                    frame_query_feature_full_seq = frame_query_feature_full_seq.repeat(topk_two_stage, 1, 1)
                model = kwargs['model']
                with torch.no_grad():
                    # follow itm score in blip colab
                    score_2nd = torch.nn.functional.softmax(model.itm_head_3(
                        model.model_3_fusion(
                            frame_query_feature_full_seq.cuda(),
                            gallery_feature_top_k.cuda())[0]
                        # [:, 0, :]), dim=1)[:, 1]
                    ), dim=1)[:, 1]


                    score_2nd_l.append(score_2nd)
            with torch.no_grad():
                score_2nd = torch.mean(torch.stack(score_2nd_l, dim=0), dim=0)
                # score_1st_stage = torch.from_numpy(score[index[:topk_two_stage]]).cuda()
                # score_1st_stage = (score_1st_stage - score_1st_stage.min()) / (score_1st_stage.max() - score_1st_stage.min())
                # TODO: (score + topk_sim) 为最后的matching 结果 (index)
                # score_all = score_2nd + score_1st_stage
                # score_all = score_2nd + 0.002 * score_1st_stage
                score_all = copy.deepcopy(score)
                score_all[list(index_topk)] = score_2nd.cpu().numpy()
                # 只用第二阶段的分数进行排序
                index_2nd = np.argsort(np.array((score_2nd).data.cpu()))
                index_2nd = index_2nd[::-1]
                # print('Query:{} :Top {} score from ITM{}, original:{}, re-ranking: top{}:{}\n'.format(ql, topk_two_stage, (score_2nd.data),  gl[index_topk], topk_two_stage, gl[index_topk][index_2nd]))
                with open(txt_path, 'a+') as f:
                    f.write(
                        'Query:{} :Top {} score from ITM{}, original:{}, re-ranking: top{}:{}\n'.format(ql, topk_two_stage,
                                                                                                        (score_2nd.data),
                                                                                                        gl[index_topk],
                                                                                                        topk_two_stage,
                                                                                                        gl[index_topk][
                                                                                                            index_2nd]))

                assert np.max(index_2nd) == (len(index[:topk_two_stage]) - 1)
                index[:topk_two_stage] = index[:topk_two_stage][index_2nd]
        else:
            raise Exception('ERROR: query is {}, not drone or bev'.format(kwargs['query_name']))

        cur_result_analysis_path = cur_result_analysis_path + '_g_{:04d}'.format(index[0])
        visualization_index = index[:5]
        matched_id = gl[visualization_index] == ql
        to_v_gallery_abs_path_l = []
        for v_idx in visualization_index:
            g_abs_path = gallery_abs_path_l[v_idx]
            #  select the most representative sample
            to_v_gallery_abs_path_l.append(g_abs_path)
        query_path = query_path_group[int((torch.argmax(qf_l, dim=0))[0].detach().cpu())]

        score_topk = score_2nd[list(index_2nd)][:5]

        if np.where(ql == gl)[0][0] in list(index_topk):
            g_equal_q_score = score_all[np.where(ql == gl)[0][0]]
        else:
            g_equal_q_score = -1
        q_zero_g = visualization_img_q_g(query_path, to_v_gallery_abs_path_l,
                                         matched_id, gallery_abs_path_l[np.where(ql == gl)[0][0]], score_topk,
                                         g_equal_q_score)

    if kwargs.get('first_stage_only'):
        visualization_index = index[:5]
        sample_reid_name = os.path.basename(cur_result_analysis_path)
        for idx, g_id in enumerate(visualization_index):
            if idx == 0:
                continue
            sample_reid_name = sample_reid_name + '_{:04d}'.format(gl[g_id])
        query_path = query_path_group[int((torch.argmax(qf_l, dim=0))[0].detach().cpu())]
        to_v_gallery_abs_path_l = []
        for v_idx in visualization_index:
            g_abs_path = gallery_abs_path_l[v_idx]
            #  select the most representative sample
            to_v_gallery_abs_path_l.append(g_abs_path)
        matched_id = gl[visualization_index] == ql
        if np.where(ql == gl)[0][0] in list(index[:5]):
            g_equal_q_score = score[np.where(ql == gl)[0][0]]
        else:
            g_equal_q_score = -1
        q_zero_g = visualization_img_q_g(query_path, to_v_gallery_abs_path_l,
                                         matched_id, gallery_abs_path_l[np.where(ql == gl)[0][0]], score[:5],
                                         g_equal_q_score)
    else:
        visualization_index = index[:5]
        sample_reid_name = os.path.basename(cur_result_analysis_path)
        for idx, g_id in enumerate(visualization_index):
            if idx == 0:
                continue
            sample_reid_name = sample_reid_name + '_{:04d}'.format(gl[g_id])

    if not os.path.exists(os.path.dirname(cur_result_analysis_path)):
        os.makedirs(os.path.dirname(cur_result_analysis_path))


    if (index[0] != ql):
        print('estimated category:{}, right id is :{}th, query id:{}, gallery id:{}'.format(index[0], (
                np.argwhere((index == ql))[0, 0] + 1), ql, np.argwhere(gl == ql)[0, 0]))
        Image.fromarray(np.uint8(q_zero_g)).convert('RGB').save(
            os.path.join(os.path.dirname(cur_result_analysis_path), '{}.jpg'.format(sample_reid_name)))
    else:
        pass
        # Image.fromarray(np.uint8(q_zero_g)).convert('RGB').save(
        #     os.path.join(os.path.dirname(cur_result_analysis_path), '{}.jpg'.format(sample_reid_name)))


    ##################################################
    acc = 0

    # good index
    query_index = np.argwhere(gl == ql)

    # query和gallery匹配上的索引
    good_index = query_index
    # print(good_index)
    # print(index[0:10])
    junk_index = np.argwhere(gl == -1)

    # index: 相似度索引(从高到低)
    # good_index: query和gallery匹配上的索引
    # junk_index: query和gallery没匹配上的索引

    CMC_tmp = compute_mAP(index, good_index, junk_index)

    return CMC_tmp, acc


def evaluate_gallery_group(qf, ql, gf_l, gl, query_path, gallery_abs_path_group, cur_result_analysis_path, query_name,
                           gallery_name, topk_two_stage, model, **kwargs):

    txt_path = kwargs.get('txt_path')
    tmp_l = []
    # 因为 video的帧数不一样，不能统一的取mean
    for gf_ in gf_l:
        tmp_l.append(torch.mean(gf_, dim=0))
    # scalar: 就一个数
    score = torch.stack(tmp_l) @ qf
    # score = torch.nn.functional.softmax(score,dim=0).cpu().numpy()
    score = score.cpu().numpy()
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]


    #################################################################
    if not kwargs.get('first_stage_only'):
        index_topk = index[:topk_two_stage]

        if ('bev' in gallery_name) or ('drone' in gallery_name):
            # gallery:KV, query:Q
            query_feature_full_seq, gallery_feature_full_seq = \
                kwargs['query_feature_full_seq'], kwargs['gallery_feature_full_seq']
            gallery_feat_top_k = [gallery_feature_full_seq[u] for u in index_topk]
            query_feature_repeat_k = (query_feature_full_seq.unsqueeze(0))  # .repeat(topk_two_stage, 1, 1)
            with torch.no_grad():
                score_2nd_gallery = []
                for gallery_feat in gallery_feat_top_k:
                    score_2nd_ = []
                    for frame_idx in range(gallery_feat.shape[0]):
                        score_2nd = torch.nn.functional.softmax(model.itm_head_3(
                            model.model_3_fusion(
                                gallery_feat[frame_idx].unsqueeze(0).cuda(),
                                query_feature_repeat_k.cuda())[0]
                            # [:,0,:]),dim=1)[:,1]
                        ), dim=1)[:, 1]


                        score_2nd_.append(score_2nd)
                    score_2nd_gallery.append(sum(score_2nd_) / len(score_2nd_))
                score_2nd = torch.stack(score_2nd_gallery, dim=0).squeeze(1)
                # print('Top {} score from ITM{}'.format(topk_two_stage,score_2nd.data))
                # score_1st_stage = torch.from_numpy(score[index[:topk_two_stage]]).cuda()
                # TODO: (score + topk_sim) 为最后的matching 结果 (index)
                # score_all = score_2nd + score_1st_stage
                score_all = copy.deepcopy(score)
                score_all[list(index_topk)] = score_2nd.cpu().numpy()
                # 只用第二阶段的分数进行排序
                index_2nd = np.argsort(np.array((score_2nd).data.cpu()))
                index_2nd = index_2nd[::-1]
                # print('Query:{} :Top {} score from ITM{}, original:{}, re-ranking: top{}:{}\n'.format(ql, topk_two_stage,
                #                                                                                       (score_2nd.data),
                #                                                                                       [gl[i] for i in index_topk],
                #                                                                                       topk_two_stage,
                #                                                                                       [[gl[i] for i in index_topk][ii] for ii in index_2nd]))
                with open(txt_path, 'a+') as f:
                    f.write(
                        'Query:{} :Top {} score from ITM{}, original:{}, re-ranking: top{}:{}\n'.format(ql, topk_two_stage,
                                                                                                        (score_2nd.data),
                                                                                                        [gl[i] for i in
                                                                                                         index_topk],
                                                                                                        topk_two_stage,
                                                                                                        [[gl[i] for i in
                                                                                                          index_topk][ii]
                                                                                                         for ii in
                                                                                                         index_2nd]))

                assert np.max(index_2nd) == (len(index[:topk_two_stage]) - 1)
                index[:topk_two_stage] = index[:topk_two_stage][index_2nd]
        else:
            raise Exception('No bev or drone in the gallery')

        cur_result_analysis_path = cur_result_analysis_path + '_g_{:04d}'.format(gl[index[0]])
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
            g_v_abs_path = g_abs_path_l[int((torch.argmax(gf_l[v_idx], dim=0))[0].detach().cpu())]
            to_v_gallery_abs_path_l.append(g_v_abs_path)

        if ql not in gl:
            print('query label {} is not in gallery label'.format(ql))
            return [-1], [-1]
        gt_idx_v = gl.index(ql)
        if np.where(ql == gl)[0][0] in list(index_topk):
            g_equal_q_score = score_all[np.where(ql == gl)[0][0]]
        else:
            g_equal_q_score = -1

        # visualization_img_q_g(query_path, to_v_gallery_abs_path_l,
        #                       matched_id, gallery_abs_path_l[np.where(ql == gl)[0][0]], score_topk, g_equal_q_score)
        score_topk = score_2nd[list(index_2nd)][:5]
        q_zero_g = visualization_img_q_g(query_path, to_v_gallery_abs_path_l, matched_id,
                                         gallery_abs_path_group[gt_idx_v][0], score_topk, g_equal_q_score)

        sample_reid_name = os.path.basename(cur_result_analysis_path)
        for idx, g_id in enumerate(visualization_index):
            if idx == 0:
                continue
            sample_reid_name = sample_reid_name + '_{:04d}'.format(gl[g_id])
        # Image.fromarray(np.uint8(q_zero_g)).convert('RGB').save(os.path.join(cur_result_analysis_path,'{}.jpg'.format(sample_reid_name)))
    if not os.path.exists(os.path.dirname(cur_result_analysis_path)):
        os.makedirs(os.path.dirname(cur_result_analysis_path))

    if kwargs.get('first_stage_only'):
        score_topk = score[list(index)][:5]
        visualization_index = index[:5]
        # tmp = []
        matched_id = [gl[i] for i in visualization_index] == ql
        to_v_gallery_abs_path_l = []
        for v_idx in visualization_index:
            g_abs_path_l = gallery_abs_path_group[v_idx]
            #  select the most representative sample
            g_v_abs_path = g_abs_path_l[int((torch.argmax(gf_l[v_idx], dim=0))[0].detach().cpu())]
            to_v_gallery_abs_path_l.append(g_v_abs_path)
        gt_idx_v = gl.index(ql)
        if np.where(ql == gl)[0][0] in list(index[:5]):
            g_equal_q_score = score[np.where(ql == gl)[0][0]]
        else:
            g_equal_q_score = -1
        q_zero_g = visualization_img_q_g(query_path, to_v_gallery_abs_path_l, matched_id,
                                         gallery_abs_path_group[gt_idx_v][0], score_topk, g_equal_q_score)
        sample_reid_name = os.path.basename(cur_result_analysis_path)
        for idx, g_id in enumerate(visualization_index):
            if idx == 0:
                continue
            sample_reid_name = sample_reid_name + '_{:04d}'.format(gl[g_id])
    if (gl[int(index[0])] != ql):
        print('estimated category:{}, right id is :{}th, query id:{}, gallery id:{}'.format(index[0], (
                np.argwhere(([gl[i] for i in index] == ql))[0, 0] + 1), ql, gl[np.argwhere(gl == ql)[0, 0]]))
        Image.fromarray(np.uint8(q_zero_g)).convert('RGB').save(
            os.path.join(os.path.dirname(cur_result_analysis_path), '{}.jpg'.format(sample_reid_name)))
    else:
        # pass
        Image.fromarray(np.uint8(q_zero_g)).convert('RGB').save(
            os.path.join(os.path.dirname(cur_result_analysis_path), '{}.jpg'.format(sample_reid_name)))

    #################################################################
    junk_index = np.argwhere(gl == -1)
    # good index
    query_index = np.argwhere(gl == ql)
    good_index = query_index
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    # index: 相似度索引(从高到低)
    # good_index: query和gallery匹配上的索引
    # junk_index: query和gallery没匹配上的索引
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index in index
    # 匹配上的有多少个
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    # 匹配上的索引
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    # CMC曲线的计算方法就是，把每个query的AccK曲线相加，再除以query的总数，即平均AccK曲线。
    # CMC是个阶跃函数
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        # 只用匹配上的样本，只能用他来算precision和recall(https://wrong.wang/blog/20190223-reid%E4%BB%BB%E5%8A%A1%E4%B8%AD%E7%9A%84cmc%E5%92%8Cmap/image_5_hu7f41be7e562defe9f107615778106342_33264_1320x0_resize_q75_h2_lanczos_3.webp)
        # d_recall的意义是delta-recall,即recall_k - recall_{k-1}
        # 而recall = 匹配上的ID个数（len(good_index)）/该ID图片总数（定值）,所以delta_recall = 1/该ID图片总数（定值）
        d_recall = 1.0 / ngood
        # precision = 检索到的同ID图片个数/查询结果总个数
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
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
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc


def compose_query_group(query_feature, query_label, query_img_path_l, gallery_feature, gallery_label,
                        gallery_img_path_l):
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
            start_idx += query_name_d[sorted(list(query_name_d.keys()))[i - 1]]
        # current_query_name = sorted(list(query_name_d.keys()))[i]
        query_label_group.append(query_label[start_idx])
        query_feature_group.append(
            query_feature[start_idx: (start_idx + query_name_d[sorted(list(query_name_d.keys()))[i]])])
        query_abs_path_group.append(
            query_img_path_l[start_idx: start_idx + query_name_d[sorted(list(query_name_d.keys()))[i]]])
        assert len(query_feature_group[-1]) == query_name_d[sorted(list(query_name_d.keys()))[i]]
        assert len(query_feature_group[-1]) == len(query_abs_path_group[-1])
    return query_feature_group, query_label_group, query_abs_path_group


def compose_gallery_group(query_feature, query_label, query_img_path_l, gallery_feature, gallery_label,
                          gallery_img_path_l):
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
        if i == 0:
            start_idx = 0
        else:
            start_idx += gallery_name_d[sorted(list(gallery_name_d.keys()))[i - 1]]
        gallery_label_group.append(gallery_label[start_idx])
        gallery_feature_group.append(
            gallery_feature[start_idx: start_idx + gallery_name_d[sorted(list(gallery_name_d.keys()))[i]]])
        assert len(gallery_feature_group[-1]) == gallery_name_d[sorted(list(gallery_name_d.keys()))[i]]
        gallery_abs_path_group.append(
            gallery_img_path_l[start_idx: start_idx + gallery_name_d[sorted(list(gallery_name_d.keys()))[i]]])
        assert len(gallery_feature_group[-1]) == len(gallery_abs_path_group[-1])
    return gallery_feature_group, gallery_label_group, gallery_abs_path_group


######################################################################
def evaluate_root(query_name, query_img_path_l, gallery_name, gallery_img_path_l, output_path, topk_two_stage, model,
                  opt, result_analysis=True):
    current_dir = os.getcwd()  # 获取当前工作目录
    absolute_path = os.path.abspath(current_dir)  # 将相对路径转换为绝对路径
    # result = scipy.io.loadmat(os.path.join(absolute_path,'pytorch_result.mat'))
    # import hdf5storage
    # result = hdf5storage.loadmat(os.path.join(absolute_path,'pytorch_result.mat'))
    import h5py, time
    since = time.time()
    # if opt.itm:
    if not opt.first_stage_only:
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
        result = scipy.io.loadmat(os.path.join(absolute_path, 'pytorch_result.mat'))
        query_feature = torch.from_numpy(result['query_f'])
        gallery_feature = torch.from_numpy(result['gallery_f'])
        query_feature_full_seq = query_feature
        gallery_feature_full_seq = gallery_feature
    # npz_result = np.load(os.path.join(absolute_path, 'pytorch_result.npz'))
    result = scipy.io.loadmat(os.path.join(absolute_path, 'pytorch_result.mat'))
    query_feature = torch.from_numpy(result['query_f'])
    gallery_feature = torch.from_numpy(result['gallery_f'])
    # # query_feature = torch.FloatTensor(npz_result['query_f_full_seq'][:,0,:])
    # query_feature = torch.from_numpy(npz_result['query_f_full_seq'][:,0,:])
    # # query_feature = query_feature / query_feature.norm(dim=-1, keepdim=True)
    # query_feature_full_seq = torch.from_numpy(npz_result['query_f_full_seq'])
    query_label = result['query_label'][0]
    # gallery_feature = torch.FloatTensor(npz_result['gallery_f_full_seq'][:,0,:])
    # gallery_feature = torch.from_numpy(npz_result['gallery_f_full_seq'][:,0,:])
    # # gallery_feature = gallery_feature / gallery_feature.norm(dim=-1, keepdim=True)
    # gallery_feature_full_seq = torch.from_numpy(npz_result['gallery_f_full_seq'])
    gallery_label = result['gallery_label'][0]
    time_elapsed = time.time() - since
    print('Loading data from disk in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    multi = os.path.isfile('multi_query.mat')

    if result_analysis:
        if not os.path.exists(os.path.join(output_path, 'result_analysis_{}_{}'.format(query_name, gallery_name))):
            os.makedirs(os.path.join(output_path, 'result_analysis_{}_{}'.format(query_name, gallery_name)))
        result_analysis_path = os.path.join(
            os.path.join(output_path, 'result_analysis_{}_{}'.format(query_name, gallery_name)))
    if multi:
        m_result = scipy.io.loadmat('multi_query.mat')
        mquery_feature = torch.FloatTensor(m_result['mquery_f'])
        mquery_label = m_result['mquery_label'][0]
        mquery_feature = mquery_feature.cuda()

    query_feature = query_feature.cuda()
    gallery_feature = gallery_feature.cuda()

    with open(os.path.join(output_path, 'result.txt'), 'a+') as f:

        f.writelines(['query_name:{}\n'.format(query_name)])
        f.writelines(['gallery_name:{}\n'.format(gallery_name)])
        f.writelines(['query_shape:{}\n'.format(query_feature.shape)])
        f.writelines(['gallery_shape:{}\n'.format(gallery_feature.shape)])
    f.close()
    print(query_feature.shape)
    print(gallery_feature.shape)
    txt_path = os.path.join(output_path, 'result.txt')

    # print(query_label)
    if 'bev' in query_name:
        CMC = torch.IntTensor(len(gallery_label)).zero_()
        ap = 0.0
        acc_all = 0.
        query_feature_group, query_label_group, query_abs_path_group = compose_query_group(query_feature, query_label,
                                                                                           query_img_path_l,
                                                                                           gallery_feature,
                                                                                           gallery_label,
                                                                                           gallery_img_path_l)
        if not opt.first_stage_only:
            query_feature_full_seq_group, query_label_group, query_abs_path_group = compose_query_group(
                query_feature_full_seq, query_label, query_img_path_l,
                gallery_feature_full_seq, gallery_label, gallery_img_path_l)
        else:
            query_feature_full_seq_group = query_feature_group
        for i in range(len(query_label_group)):
            # https://wrong.wang/blog/20190223-reid%E4%BB%BB%E5%8A%A1%E4%B8%AD%E7%9A%84cmc%E5%92%8Cmap/
            # query_path = query_abs_path_group[i][0]
            # query_name = os.path.dirname(query_path).split('/')[-2]
            cur_result_analysis_path = os.path.join(result_analysis_path, 'q_{:04d}'.format(query_label_group[i]))

            # txt_path = os.path.join(output_path, 'result.txt')

            [ap_tmp, CMC_tmp], acc = evaluate(qf_l=query_feature_group[i], ql=query_label_group[i], gf=gallery_feature,
                                              gl=gallery_label, query_path_group=query_abs_path_group[i],
                                              gallery_abs_path_l=gallery_img_path_l,
                                              cur_result_analysis_path=cur_result_analysis_path,
                                              topk_two_stage=topk_two_stage, query_name=query_name,
                                              query_feature_full_seq=query_feature_full_seq_group[i],
                                              gallery_feature_full_seq=gallery_feature_full_seq, model=model,
                                              txt_path=txt_path,
                                              first_stage_only=opt.first_stage_only)
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

        gallery_feature_group, gallery_label_group, gallery_abs_path_group = compose_gallery_group(query_feature,
                                                                                                   query_label,
                                                                                                   query_img_path_l,
                                                                                                   gallery_feature,
                                                                                                   gallery_label,
                                                                                                   gallery_img_path_l)
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
            # query_name = os.path.dirname(query_path).split('/')[-1]
            cur_result_analysis_path = os.path.join(result_analysis_path, 'q_{}'.format(os.path.dirname(query_path).split('/')[-1]))
            # , , , , , , , , model
            ap_tmp, CMC_tmp = evaluate_gallery_group(qf=query_feature[i], ql=query_label[i], gf_l=gallery_feature_group,
                                                     gl=gallery_label_group, query_path=query_path,
                                                     gallery_abs_path_group=gallery_abs_path_group,
                                                     cur_result_analysis_path=cur_result_analysis_path,
                                                     query_name=query_name, gallery_name=gallery_name,
                                                     topk_two_stage=topk_two_stage,
                                                     model=model, query_feature_full_seq=query_feature_full_seq[i],
                                                     gallery_feature_full_seq=gallery_feature_full_seq_group,
                                                     txt_path=txt_path,
                                                     first_stage_only=opt.first_stage_only)
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

        query_feature_group, query_label_group, query_abs_path_group = compose_query_group(query_feature, query_label,
                                                                                           query_img_path_l,
                                                                                           gallery_feature,
                                                                                           gallery_label,
                                                                                           gallery_img_path_l)
        if not opt.first_stage_only:
            query_feature_full_seq_group, query_label_group, query_abs_path_group = compose_query_group(
                query_feature_full_seq, query_label, query_img_path_l,
                gallery_feature_full_seq, gallery_label, gallery_img_path_l)
        else:
            query_feature_full_seq_group = query_feature_group
        for i in range(len(query_label_group)):
            # https://wrong.wang/blog/20190223-reid%E4%BB%BB%E5%8A%A1%E4%B8%AD%E7%9A%84cmc%E5%92%8Cmap/
            query_path = query_abs_path_group[i][0]
            if query_label_group[i] == int((query_path).split('/')[-2]):
                pass
            else:
                raise Exception('qurey_label != query img name')
            # query_name = os.path.dirname(query_path).split('/')[-1]
            cur_result_analysis_path = os.path.join(result_analysis_path, 'q_{}'.format(os.path.dirname(query_path).split('/')[-1]))

            [ap_tmp, CMC_tmp], acc = evaluate(qf_l=query_feature_group[i], ql=query_label_group[i], gf=gallery_feature,
                                              gl=gallery_label, query_path_group=query_abs_path_group[i],
                                              gallery_abs_path_l=gallery_img_path_l,
                                              cur_result_analysis_path=cur_result_analysis_path,
                                              topk_two_stage=topk_two_stage, query_name=query_name,
                                              query_feature_full_seq=query_feature_full_seq_group[i],
                                              gallery_feature_full_seq=gallery_feature_full_seq, model=model,
                                              txt_path=txt_path,
                                              first_stage_only=opt.first_stage_only)
            # [ap_tmp, CMC_tmp], acc = evaluate(query_feature_group[i], query_label_group[i], gallery_feature,
            #                                   gallery_label, query_abs_path_group[i], gallery_img_path_l,
            #                                   cur_result_analysis_path)
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
        gallery_feature_group, gallery_label_group, gallery_abs_path_group = compose_gallery_group(query_feature,
                                                                                                   query_label,
                                                                                                   query_img_path_l,
                                                                                                   gallery_feature,
                                                                                                   gallery_label,
                                                                                                   gallery_img_path_l)
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
            query_path = query_img_path_l[i]
            query_name = os.path.dirname(query_path).split('/')[-1]
            cur_result_analysis_path = os.path.join(result_analysis_path, 'q_{}'.format(query_name))
            ap_tmp, CMC_tmp = evaluate_gallery_group(qf=query_feature[i], ql=query_label[i], gf_l=gallery_feature_group,
                                                     gl=gallery_label_group, query_path=query_path,
                                                     gallery_abs_path_group=gallery_abs_path_group,
                                                     cur_result_analysis_path=cur_result_analysis_path,
                                                     query_name=query_name, gallery_name=gallery_name,
                                                     topk_two_stage=topk_two_stage,
                                                     model=model, query_feature_full_seq=query_feature_full_seq[i],
                                                     gallery_feature_full_seq=gallery_feature_full_seq_group,
                                                     txt_path=txt_path,
                                                     first_stage_only=opt.first_stage_only)
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
# CMC = torch.IntTensor(len(gallery_label)).zero_()
# ap = 0.0
# if multi:
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

