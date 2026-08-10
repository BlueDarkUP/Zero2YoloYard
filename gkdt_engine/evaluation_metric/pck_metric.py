import numpy as np
import scipy.stats
from utils.utils import mean_confidence_interval, mean_confidence_interval_multiple
from evaluation_metric.coco_eval_funs import compute_recall_ap
import copy

class PCKMetric(object):
    def __init__(self, pck_thresholds=[0.10, 0.20]):
        '''
        Computing Percentage of Correct Keypoints (PCK) @ thresholds

        pck_thresholds: a list of thresholds
        type: 'bbox' or 'img' ()
        '''
        self.pck_thresholds = np.array(pck_thresholds)
        self.acc_list = [[] for _ in range(len(pck_thresholds))]
        self.tps, self.fps = [[] for _ in range(len(pck_thresholds))], [[] for _ in range(len(pck_thresholds))]
    
    def compute_pck_and_update(self, predictions: np.ndarray, groundtruth: np.ndarray, kp_mask: np.ndarray, edges: np.ndarray):
        '''
        predictions: B x N x 2 (B images, each image with N keypoints)
        groundtruth: B x N x 2
        kp_mask:     B x N
        edges:       B x 2 (bounding box's edges, or image's width and height)
        '''
        square_diff = np.sum((predictions - groundtruth) ** 2, axis=2)  # B x N
        longer_edge = np.max(edges, axis=1)  # B
        longer_edge = longer_edge.reshape(-1, 1)  # B x 1

        result_mask = kp_mask.astype(np.bool_)

        tps_this_iter = []
        fps_this_iter = []
        acc_this_iter = []
        for thr in self.pck_thresholds:
            judges = (square_diff <= (thr * longer_edge) ** 2)
            judges = judges.reshape(-1)
            # masking
            judges = judges[result_mask.reshape(-1)]

            tps = judges.astype(int).tolist()  # a list
            fps = (1 - judges).astype(int).tolist()  # a list
            acc = np.sum(judges) / max(len(judges), 1)  # a value
            
            tps_this_iter.append(tps)
            fps_this_iter.append(fps)
            acc_this_iter.append(acc)

        # record results of this iteration
        for i in range(len(self.pck_thresholds)):
            self.tps[i].extend(tps_this_iter[i])
            self.fps[i].extend(fps_this_iter[i])
            self.acc_list[i].append(acc_this_iter[i])

        # return single-term results
        return acc_this_iter, tps_this_iter, fps_this_iter
    
    def get_mean_accuracy_result(self, acc_list=None):
        acc_list_tmp = self.acc_list if (acc_list is None) else acc_list
        acc_mean, interval = mean_confidence_interval_multiple(acc_list_tmp)
        return acc_mean, interval

    def get_recall_ap(self):
        # below statistics may be less used
        recall, AP = compute_recall_ap(self.tps, self.fps, len(self.tps[0]))
        return recall, AP
    
    