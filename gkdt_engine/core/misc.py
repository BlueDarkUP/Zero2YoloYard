import torch
import torch.nn as nn

def compute_openkd_heatmap_loss(cfg, model, loss_func, query_kps, query_kp_mask, heatmaps_list,
                                support_kp_mask=None, kps_texts_mask=None):
    '''
    Compute the heatmap loss for episodes
    :param cfg:
    :param model: OpenKDModel
    :param loss_func:
    :param query_kps: S x B2 x N x 2 (TODO: used for GT)
    :param query_kp_mask: S x B2 x N (TODO: used for GT) (Note it requires original query_kp_mask)
    :param heatmaps_list: [{'obj': tensor, 'text': tensor, 'image': tensor}, ...], a list of dict

    Below masks used for determining valid text/vision-induced heatmaps.
    :param support_kp_mask: S x B1 x N or None. If None, it means all kps are valid.
    :param kps_texts_mask: None or tensor S x N x T2. If None, it means all texts are valid.
    :return:
    '''
    # ! add here, handle DDP wrapped model
    model_module = model.module if hasattr(model, 'module') else model


    S, B, N = query_kp_mask.shape
    multi_group_supervision = cfg.LOSS.MULTI_GROUP_SUPERVISION  # True or False
    query_kp_mask= query_kp_mask.long()  # S x B x N

    loss = 0
    count = 0
    for s in range(S):  # S episodes
        per_query_kps = query_kps[s]              # B2 x N x 2 (used for GT)
        per_query_kp_mask = query_kp_mask[s]      # B2 x N     (used for GT)
        per_heatmaps_set = heatmaps_list[s]       # {'text': tensor, 'image': tensor}
        if (len(per_heatmaps_set['image']) > 0) and (support_kp_mask is not None):
            per_support_kp_mask = support_kp_mask[s]  # B1 x N
        else:
            per_support_kp_mask = None
        if (len(per_heatmaps_set['text']) > 0) and (kps_texts_mask is not None):
            per_kps_texts_mask = kps_texts_mask[s]  # N x T2
        else:
            per_kps_texts_mask = None

        # Changed here to handle DDP wrapper's model. (For DDP wrapper, it is safe to access functions via model.module)
        # as we may access/modify the model parameters which may have impacts in DDP
        heatmaps_fused, fused_mask_sum, heatmaps_collect, masks_collect = model_module.openkd_heatmap_fuse(
            per_heatmaps_set,
            per_support_kp_mask,
            per_kps_texts_mask
        )
        if multi_group_supervision == True:  # (each kp_heatmap * obj_heamap) then fuse
            for g, each_group_heatmap in enumerate(heatmaps_collect):
                # mask_tmp = masks_collect[g] * per_query_kp_mask  #.long()
                # loss_tmp = loss_func(each_group_heatmap, per_query_kps, mask_tmp)
                mask_tmp = masks_collect[g]
                loss_tmp = loss_func(each_group_heatmap, per_query_kps, per_query_kp_mask, mask_tmp)
                loss += loss_tmp
                if mask_tmp.sum() > 0:  # if no valid kps, no need to count
                    count += 1
        else:  # False. (fused all kp_heatmap) * obj_heamap
            # mask_tmp = (fused_mask_sum > 0).long() * per_query_kp_mask #.long()
            # loss_tmp = loss_func(heatmaps_fused, per_query_kps, mask_tmp)
            mask_tmp = (fused_mask_sum > 0).long()
            loss_tmp = loss_func(heatmaps_fused, per_query_kps, per_query_kp_mask, mask_tmp)
            loss += loss_tmp
            if mask_tmp.sum() > 0:  # if no valid kps, no need to count
                count += 1
    if count > 0:
        loss /= count  # count must >0
    return loss

def split_main_aux_heatmaps(predict_heatmaps_list, num_main_kps=11, num_aux_kps=18):
    '''
    :param predict_heatmaps_list: a list of S episodes' heatmaps_set. Each heatmaps_set = {'obj': [], 'text': [], 'image': []}
    :param num_main_kps:
    :param num_aux_kps:
    :return: main_heatmaps_set, aux_heatmaps_set
    '''
    main_heatmaps_list = []
    aux_heatmaps_list = []
    total_num = num_main_kps + num_aux_kps
    for each_heatmaps_set in predict_heatmaps_list:
        main_tmp = {'obj': [], 'text': [], 'image': []}
        aux_tmp = {'obj': [], 'text': [], 'image': []}
        if len(each_heatmaps_set['obj']) > 0:
            main_tmp['obj'] = each_heatmaps_set['obj']  # B2 x 1 x H x W
            aux_tmp['obj'] = each_heatmaps_set['obj']
        if len(each_heatmaps_set['text']) > 0:
            assert (each_heatmaps_set['text']).shape[1] == total_num, 'The number of heatmaps should match.'
            main_tmp['text'] = each_heatmaps_set['text'][:, :num_main_kps]  # B2 x N x H x W
            aux_tmp['text'] = each_heatmaps_set['text'][:, num_main_kps:]   # B2 x A x H x W
        if len(each_heatmaps_set['image']) > 0:
            assert (each_heatmaps_set['image']).shape[1] == total_num, 'The number of heatmaps should match.'
            main_tmp['image'] = each_heatmaps_set['image'][:, :num_main_kps]  # B2 x N x H x W
            aux_tmp['image'] = each_heatmaps_set['image'][:, num_main_kps:]  # B2 x A x H x W
        main_heatmaps_list.append(main_tmp)
        aux_heatmaps_list.append(aux_tmp)

    return main_heatmaps_list, aux_heatmaps_list