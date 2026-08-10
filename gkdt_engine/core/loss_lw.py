import torch
import torch.nn as nn
from utils.heatmap import putGaussianMaps
import matplotlib.pyplot as plt

def get_loss_function(cfg):
    loss_func = nn.MSELoss(reduction='sum')  # Used in keypoint heatmap regression
    # loss_func = nn.L1Loss(reduction='sum')
    # loss_func = nn.NLLLoss(ignore_index=-1)  # Used in animal class classification
    return loss_func

class DirectCoordLoss(nn.Module):
    def __init__(self, cfg):
        super(DirectCoordLoss, self).__init__()
        self.loss_type = 'direct_coord'
        self.edge_length = cfg.LOSS.DIRECT_COORD.EDGE_LENGTH

    def forward(self, coords, kp_labels, kp_mask):
        '''
        input arguments
        coords: B x N x 2 (each row is x, y, -1<=x<1, -1<=y<1)
        kp_labels: B x N x 2 (each row is x, y, -1<=x<1, -1<=y<1)
        kp_mask:   B x N
        '''
        B, N, _ = coords.shape
        num_valid_kps = torch.sum(kp_mask)
        if num_valid_kps <= 0:
            # set 0 in cuda device in order to be compatible with subsequent code
            final_loss = torch.tensor(0., requires_grad=True).cuda()
        else:
            if self.edge_length == 1:  # coord range is 0~1
                coords = coords / 2 + 0.5
                kp_labels = kp_labels / 2 + 0.5
            loss = ((coords - kp_labels) ** 2).sum(-1) * kp_mask
            final_loss = torch.sum(loss) / num_valid_kps
        return final_loss

class HeatmapLoss(nn.Module):
    def __init__(self, cfg):
        super(HeatmapLoss, self).__init__()
        self.cfg = cfg
        self.loss_type = cfg.LOSS.TYPE
        # if self.loss_type == 'MSE':
        #     self.loss_func = nn.MSELoss(reduction='mean')
        # elif self.loss_type == 'sigmoid-bce':
        #     self.loss_func = nn.BCELoss(reduction='mean')
        # elif self.loss_type == 'cross-entropy':
        #     self.loss_func = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)
        # else:
        #     raise NotImplementedError

    def forward(self, heatmaps_predict, kp_labels, kp_mask):
        '''
        input arguments
        heatmaps_predict: B x N x H x W
        kp_labels: B x N x 2 (each row is x, y, -1<=x<1, -1<=y<1)
        kp_mask:   B x N
        '''
        B, N, H, W = heatmaps_predict.shape
        l     = self.cfg.DATASET.SQUARE_IMAGE_LENGTH  # 384 (image space)
        sigma = self.cfg.DATASET.SIGMA  # 14 (image space)
        # stride= self.cfg.MODEL.ENCODER.DOWNSIZE_FACTOR  # 32
        stride = l // W
        assert (stride * W) == l, "Pay attention. The image length should be divisible by (upscaled) feature map length."
        final_loss = 0
        num_valid_kps = torch.sum(kp_mask)
        if num_valid_kps <= 0:
            # set 0 in cuda device in order to be compatible with subsequent code
            final_loss = torch.tensor(0., requires_grad=True).cuda()
        else:
            for b in range(B):
                for n in range(N):
                    if kp_mask[b, n] > 0:
                        center = ((kp_labels[b, n] / 2 + 0.5) * l).clamp(0, l - 1)  # original image length 384
                        # sigma (in original image scale) 14; heatmap grid size 12 x 12; stride 32
                        heatmap_gt, _ = putGaussianMaps(center, sigma, H, W, stride)
                        # loss = self.loss_func(heatmaps_predict[b, n], heatmap_gt)
                        # loss = torch.nn.functional.mse_loss(heatmaps_predict[b, n], heatmap_gt, reduction='mean')
                        loss = ((heatmaps_predict[b, n] - heatmap_gt) ** 2).mean()
                        final_loss += loss
            final_loss /= num_valid_kps


            # if self.loss_type in ['MSE', 'sigmoid-bce']:
            #     for b in range(B):
            #         for n in range(N):
            #             if kp_mask[b, n] > 0:
            #                 center = ((kp_labels[b, n] / 2 + 0.5) * l).clamp(0, l-1)  # original image length 384
            #                 # sigma (in original image scale) 14; heatmap grid size 12 x 12; stride 32
            #                 heatmap_gt, _ = putGaussianMaps(center, sigma, H, W, stride)
            #                 # loss = self.loss_func(heatmaps_predict[b, n], heatmap_gt)
            #                 # loss = torch.nn.functional.mse_loss(heatmaps_predict[b, n], heatmap_gt, reduction='mean')
            #                 loss = ((heatmaps_predict[b, n] - heatmap_gt) ** 2).mean()
            #                 final_loss += loss
            #     final_loss /= num_valid_kps
            # elif self.loss_type in ['cross-entropy']:
            #     # grid classification
            #     predict_grids = heatmaps_predict.reshape(B*N, -1)  # (B*N) x (H*W)
            #
            #     # compute grid groundtruth and deviation
            #     grid_length = W  # H == W
            #     gridxy = (kp_labels /2 + 0.5) * grid_length  # coordinate -1~1 --> 0~self.grid_length, B2 x N x 2
            #     gridxy_quantized = gridxy.long().clamp(0, grid_length - 1)  # B2 x N x 2
            #     # Method 1, deviation range: -1~1
            #     # label_deviations = (gridxy - (gridxy_quantized + 0.5)) * 2  # we hope the deviation ranges -1~1, B2 x N x 2
            #     # Method 2, deviation range: 0~1
            #     # label_deviations = (gridxy - gridxy_quantized)  # we hope the deviation ranges 0~1, B2 x N x 2
            #     label_grids = gridxy_quantized[:, :, 1] * grid_length + gridxy_quantized[:, :, 0]  # 0 ~ grid_length * grid_length - 1, B2 x N
            #
            #     kp_mask_int = (kp_mask>0).long()
            #     label_grids = label_grids * kp_mask_int + (-1) * (1-kp_mask_int)  # set ignore_index=-1 for invalid keypoints
            #     label_grids2 = label_grids.reshape(-1)
            #     final_loss = self.loss_func(predict_grids, label_grids2)
            # else:
            #     raise NotImplementedError

        return final_loss

class HeatmapLossV2(nn.Module):
    def __init__(self, cfg):
        super(HeatmapLossV2, self).__init__()
        self.cfg = cfg
        self.loss_type = cfg.LOSS.TYPE

    def forward(self, heatmaps_predict, kp_labels, kp_mask, prompt_kp_mask):
        '''
        MSE Loss. If kp is valid, GT heatmap is constructed; if kp is invalid, GT heatmap is set to 0 (negative samples learning).
        
        input arguments
        heatmaps_predict: B x N x H x W
        kp_labels: B x N x 2 (each row is x, y, -1<=x<1, -1<=y<1)
        kp_mask:   B x N
        prompt_kp_mask: 1 x N or B x N
        '''
        # heatmaps_predict = nn.functional.relu(heatmaps_predict)  # may lead to mode collapse

        B, N, H, W = heatmaps_predict.shape
        l     = self.cfg.DATASET.SQUARE_IMAGE_LENGTH  # 384 (image space)
        sigma = self.cfg.DATASET.SIGMA  # 14 (image space)
        # stride= self.cfg.MODEL.ENCODER.DOWNSIZE_FACTOR  # 32
        stride = l // W
        assert (stride * W) == l, "Pay attention. The image length should be divisible by (upscaled) feature map length."
        final_loss = 0
        prompt_kp_mask = prompt_kp_mask.expand(B, N) if (prompt_kp_mask.shape[0] == 1) else prompt_kp_mask  # B x N
        num_valid_prompt_kps = torch.sum(prompt_kp_mask)
        device = heatmaps_predict.device
        if num_valid_prompt_kps <= 0:
            # set 0 in cuda device in order to be compatible with subsequent code
            final_loss = torch.tensor(0., requires_grad=True).cuda(device=device)
        else:
            for b in range(B):
                for n in range(N):
                    if prompt_kp_mask[b, n] > 0:
                        if kp_mask[b, n] > 0:
                            center = ((kp_labels[b, n] / 2 + 0.5) * l).clamp(0, l - 1)  # original image length 384
                            # sigma (in original image scale) 14; heatmap grid size 12 x 12; stride 32
                            heatmap_gt, _ = putGaussianMaps(center, sigma, H, W, stride)
                        else:
                            heatmap_gt = torch.zeros((H, W)).cuda(device=device)  # negative samples learning
                        # loss = torch.nn.functional.mse_loss(heatmaps_predict[b, n], heatmap_gt, reduction='mean')
                        loss = ((heatmaps_predict[b, n] - heatmap_gt) ** 2).mean()
                        final_loss += loss
            final_loss /= num_valid_prompt_kps

        return final_loss

def generate_batch_heatmaps(kp_labels, kp_mask, im_length=384, sigma=14, hm_width=48, is_cuda=True):
    '''
    input arguments
    kp_labels: B x N x 2 (each row is x, y, -1<=x<1, -1<=y<1)
    kp_mask:   B x N
    '''
    B, N = kp_mask.shape
    l     = im_length  # 384 (image space)
    sigma = sigma  # 14 (image space)
    W = hm_width  # (heatmap space)
    # stride= self.cfg.MODEL.ENCODER.DOWNSIZE_FACTOR  # 32
    stride = l // W
    assert (stride * W) == l, "Pay attention. The image length should be divisible by (upscaled) feature map length."
    heatmaps = torch.zeros((B, N, W, W)).cuda()
    num_valid_kps = torch.sum(kp_mask)
    if num_valid_kps <= 0:
        pass
    else:
        for b in range(B):
            for n in range(N):
                if kp_mask[b, n] > 0:
                    center = ((kp_labels[b, n] / 2 + 0.5) * l).clamp(0, l - 1)  # original image length 384
                    # sigma (in original image scale) 14; heatmap grid size 12 x 12; stride 32
                    heatmap_gt, _ = putGaussianMaps(center, sigma, W, W, stride)
                    heatmaps[b, n] = heatmap_gt
    return heatmaps