import numpy as np
from utils.utils import mean_confidence_interval

class NEMetric(object):
    def __init__(self):
        '''
        Computing Normalized Error (NE) of predicted keypoints
        '''
        self.ne_list = []  # normalized error list

    def compute_ne_and_update(self, predictions: np.ndarray, groundtruth: np.ndarray, kp_mask: np.ndarray, edges: np.ndarray):
        '''
        predictions: B x N x 2 (B images, each image with N keypoints)
        groundtruth: B x N x 2
        kp_mask:     B x N
        edges:       B x 2 (bounding box's edges, or image's width and height)
        '''
        square_diff = np.sum((predictions - groundtruth) ** 2, axis=2)  # B x N
        longer_edge = np.max(edges, axis=1)  # B
        longer_edge = longer_edge.reshape(-1, 1)  # B x 1

        ne = np.sqrt(square_diff) / longer_edge  # B x N
        
        ne = ne.reshape(-1)
        result_mask = kp_mask.astype(np.bool_).reshape(-1)
        
        ne = ne[result_mask]
        ne_mean_this_episode = np.sum(ne) / max(len(ne), 1)
        
        # record results of this iteration
        self.ne_list.append(ne_mean_this_episode)

        return ne_mean_this_episode
    
    def get_mean_ne_result(self, ne_list=None):
        ne_list_tmp = self.ne_list if (ne_list is None) else ne_list
        ne_mean, ne_interval = mean_confidence_interval(ne_list_tmp)
        return ne_mean, ne_interval

