import os
import json
import tempfile
import numpy as np
import argparse

import sys
sys.path.append('.')  # append pwd into system path so that it could find python modules
print(os.getcwd())

from datasets.pycocotools.coco import COCO
from datasets.pycocotools.cocoeval import COCOeval
from evaluation_metric.pck_metric import PCKMetric

def compute_keypoint_similarity(kp1, kp2, mask=None):
    """
    Compute keypoint similarity between two instances (lower is better for matching).
    Uses normalized Euclidean distance.
    
    Args:
        kp1: N x 2 array of keypoints
        kp2: N x 2 array of keypoints
        mask: N array indicating valid keypoints (optional)
    
    Returns:
        float: normalized similarity score (0 means identical, 1 means very different)
    """
    diff = np.sum((kp1 - kp2) ** 2, axis=1)  # N
    
    if mask is not None:
        mask = mask.astype(bool)
    else:
        mask = np.ones(len(kp1), dtype=bool)
    
    valid_diff = diff[mask]
    if len(valid_diff) == 0:
        return float('inf')
    
    similarity = np.sqrt(np.mean(valid_diff))  # RMSE
    return similarity

def compute_bbox_iou(bbox1, bbox2):
    """
    Compute IoU (Intersection over Union) between two bounding boxes.
    
    Args:
        bbox1: [x, y, w, h] format
        bbox2: [x, y, w, h] format
    
    Returns:
        float: IoU score (0 to 1, higher is better)
    """
    # Convert from [x, y, w, h] to [x1, y1, x2, y2]
    x1_min, y1_min, w1, h1 = bbox1
    x2_min, y2_min, w2, h2 = bbox2
    
    x1_max = x1_min + w1
    y1_max = y1_min + h1
    x2_max = x2_min + w2
    y2_max = y2_min + h2
    
    # Compute intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Compute union
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    
    iou = inter_area / union_area
    return iou

def find_best_matching_detection(gt_ann, dt_anns, match_method=1):
    """
    Find the best matching detection for a GT annotation.
    
    Args:
        gt_ann: GT annotation dictionary
        dt_anns: List of DT annotations
        match_method: 1 for keypoint distance, 0 for bbox IoU
    
    Returns:
        tuple: (best_matching_annotation, matching_score)
    """
    if len(dt_anns) == 0:
        return None, float('inf')
    
    best_match = None
    if match_method == 1:  # init keypoint distance
        best_score = float('inf') 
    else: # bbox IoU
        best_score = -1.0
    
    gt_keypoints = np.array(gt_ann['keypoints']).reshape(-1, 3)
    gt_kp_coords = gt_keypoints[:, :2].astype(np.float32)
    gt_kp_mask = gt_keypoints[:, 2].astype(np.int32)
    gt_bbox = gt_ann.get('bbox', [])
    
    for dt_ann in dt_anns:
        if match_method == 1:
            # Match based on keypoint distance (lower is better)
            dt_keypoints = np.array(dt_ann['keypoints']).reshape(-1, 3)
            dt_kp_coords = dt_keypoints[:, :2].astype(np.float32)
            score = compute_keypoint_similarity(gt_kp_coords, dt_kp_coords, gt_kp_mask)
            
            if score < best_score:
                best_score = score
                best_match = dt_ann
        else:
            # Match based on bbox IoU (higher is better)
            dt_bbox = dt_ann.get('bbox', [])
            
            if len(gt_bbox) == 4 and len(dt_bbox) == 4:
                score = compute_bbox_iou(gt_bbox, dt_bbox)
                if score > best_score:
                    best_score = score
                    best_match = dt_ann
    
    return best_match, best_score

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='计算关键点检测的AP指标')
    parser.add_argument('--pred_file', 
                       type=str,
                       default="/project/vonneumann1/cl2025/GKD/experiments/expert_benchmark/pred_file/main4_neg_samples_bilinear_tvm.json",
                       help='模型预测结果的JSON文件路径')
    parser.add_argument('--original_gt_file', 
                       type=str,
                    #    default="/project/vonneumann1/cl2025/GKD/experiments/evaluation_for_coco/humanart_result/validation_humanart.json",
                       default="/project/vonneumann1/cl2025/keypoint_datasets/human_pose/coco/annotations/person_keypoints_val2017.json",
                       help='原始COCO标注文件路径')
    parser.add_argument('--pck_threshold',
                       nargs='+',
                       default=[0.2, 0.1, 0.05],
                       help='PCK评估的距离阈值')
    parser.add_argument('--match_method',
                       type=int,
                       default=1,
                       help='GT和DT匹配方法: 1 for keypoint similarity, 0 for bbox IoU')
    parser.add_argument('--output_file', 
                       type=str,
                       default="/project/vonneumann1/cl2025/GKD/experiments/evaluation_for_coco/PCK_result/result.json",
                       help='结果保存文件路径')
    parser.add_argument('--score_threshold', 
                       type=float,
                       default=0.05,
                       help='score threshold for filtering predictions')
    
    return parser.parse_args()

def compute_keypoints_pck(result_file, original_gt_file, pck_threshold=[0.1, 0.05], match_method=1, score_threshold=0.05):
    """
    计算关键点检测的PCK（使用完整的原始GT文件）
    
    参数:
        result_file (str): 模型预测结果的JSON文件路径
        original_gt_file (str): 原始COCO标注文件路径
        pck_threshold (float): PCK评估的距离阈值
        match_method (int): GT和DT匹配方法 (1: keypoint similarity, 0: bbox IoU)
    返回:
        dict: 包含关键点评估指标的字典
    """
    # 加载COCO格式的GT
    coco_gt = COCO(original_gt_file)

    print(f"原始GT文件统计:")
    print(f"  - 包含图像: {len(coco_gt.dataset['images'])} 张")
    print(f"  - 包含标注: {len(coco_gt.dataset['annotations'])} 条")
    
    # -------------------------- 处理预测文件 --------------------------
    # 读取预测文件
    with open(result_file, 'r') as f:
        pred_data = json.load(f)
    
    # 判断预测文件格式：如果是列表格式，直接使用；如果是字典格式，提取annotations字段
    if isinstance(pred_data, list):
        # 新的格式：直接是预测列表
        pred_annotations = pred_data
    elif isinstance(pred_data, dict) and 'annotations' in pred_data:
        # 旧的格式：包含annotations字段的字典
        pred_annotations = pred_data['annotations']
    else:
        raise ValueError("预测结果文件格式不正确，应该是列表格式或包含annotations字段的字典格式")
    
    print(f"从预测文件中提取到 {len(pred_annotations)} 个预测结果")
    
    # Filter predictions by score threshold
    pred_annotations = [ann for ann in pred_annotations if ann.get('score', 0) > score_threshold]
    print(f"过滤后剩余 {len(pred_annotations)} 个预测结果 (score > {score_threshold})")
    
    # 提取所有unique的image_id（用于统计信息）
    pred_image_ids = list(set([ann['image_id'] for ann in pred_annotations]))
    print(f"预测结果涉及 {len(pred_image_ids)} 个不同的图像")

    # -------------------------- 准备预测数据 --------------------------
    # 准备符合COCO评估要求的预测数据
    coco_predictions = []
    for ann in pred_annotations:
        if 'image_id' not in ann:
            raise ValueError("每个预测必须包含image_id字段")
        if 'keypoints' not in ann:
            raise ValueError("每个预测必须包含keypoints字段")
        
        pred = {
            "image_id": ann['image_id'],
            "category_id": ann.get('category_id', 1),  # 如果有category_id字段则使用，否则默认为person
            "keypoints": ann['keypoints'],
            "score": ann.get('score', 1.0), # 如果有score字段则使用，否则默认为1.0
            "bbox": ann.get('bbox', [])  # 如果有bbox字段则使用，否则默认为[]
        }
        coco_predictions.append(pred)
    
    # 使用临时文件存储预测数据
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        json.dump(coco_predictions, tmp_file)
        tmp_path = tmp_file.name
    
    try:
        # 加载预测结果并评估
        coco_dt = coco_gt.loadRes(tmp_path)
        
        # Initialize PCK metric
        pck_metric = PCKMetric(pck_thresholds=pck_threshold)
        
        # Get all image IDs from GT
        all_image_ids = coco_gt.getImgIds()
        
        # Iterate through each GT annotation instance
        for img_id in all_image_ids:
            # Get image info for edge information
            img_info = coco_gt.loadImgs(ids=[img_id])[0]
            img_width = img_info['width']
            img_height = img_info['height']
            edges = np.array([[img_width, img_height]])

            # Get GT annotations for this image
            gt_ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
            gt_anns = coco_gt.loadAnns(ids=gt_ann_ids)
            
            # Get predicted annotations for this image
            dt_ann_ids = coco_dt.getAnnIds(imgIds=[img_id])
            dt_anns = coco_dt.loadAnns(ids=dt_ann_ids)
            
            # Match GT and DT annotations by category
            for gt_ann in gt_anns:
                # Get GT keypoints
                gt_keypoints = np.array(gt_ann['keypoints']).reshape(-1, 3)  # N x 3 (x, y, visibility)
                gt_kp_coords = gt_keypoints[:, :2].astype(np.float32)  # N x 2
                gt_kp_mask = gt_keypoints[:, 2].astype(np.int32)  # N (visibility mask)
                
                # Find matching predictions (same image and category)
                category_id = gt_ann.get('category_id')
                matching_dt_anns = [ann for ann in dt_anns if ann.get('category_id') == category_id]
                
                # Find the best matching detection
                best_dt_ann, match_score = find_best_matching_detection(gt_ann, matching_dt_anns, match_method)
                
                if best_dt_ann is None:
                    # No predictions for this instance, record zeros
                    dt_kp_coords = np.zeros_like(gt_kp_coords)
                else:
                    # Use the best matching detection
                    dt_keypoints = np.array(best_dt_ann['keypoints']).reshape(-1, 3)  # N x 3
                    dt_kp_coords = dt_keypoints[:, :2].astype(np.float32)  # N x 2
                
                # Prepare batch for PCK computation (expand dims for batch)
                predictions_batch = dt_kp_coords[np.newaxis, :, :]  # 1 x N x 2
                groundtruth_batch = gt_kp_coords[np.newaxis, :, :]  # 1 x N x 2
                valid_kp_mask_batch = gt_kp_mask[np.newaxis, :]  # 1 x N
                edges_batch = edges  # 1 x 2
                
                # Compute PCK and update metric
                pck_metric.compute_pck_and_update(
                    predictions_batch,
                    groundtruth_batch,
                    valid_kp_mask_batch,
                    edges_batch
                )
        
        # Get mean accuracy results
        # acc_mean, interval = pck_metric.get_mean_accuracy_result()  # Since the number of batch for test images is 1, the statistic is not accurate
        recall = [np.sum(pck_metric.tps[idx]) / max(len(pck_metric.tps[idx]), 1) for idx in range(len(pck_threshold))]
        
        # 整理结果
        pck_result_names = [f'PCK@{thr:.2f}' for thr in pck_threshold]
        
        return {
            "recall": dict(zip(pck_result_names, recall)),
            'gt_path': original_gt_file,
            'stats': {  # 统计信息
                'original_images': len(coco_gt.dataset['images']),
                'original_annotations': len(coco_gt.dataset['annotations']),
                'prediction_count': len(pred_annotations),
                'prediction_images': len(pred_image_ids)
            }
        }
    finally:
        # 清理临时文件
        try:
            os.unlink(tmp_path)
        except:
            pass


def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.pred_file):
        print(f"错误: 预测文件不存在: {args.pred_file}")
        return
    
    if not os.path.exists(args.original_gt_file):
        print(f"错误: GT文件不存在: {args.original_gt_file}")
        return
    
    print("="*60)
    print("关键点检测PCK评估工具")
    print("="*60)
    print(f"预测文件: {args.pred_file}")
    print(f"GT文件: {args.original_gt_file}")
    print(f"匹配方法: {'keypoint similarity' if args.match_method == 1 else 'bbox IoU'}")
    print("="*60)
    
    # 执行评估
    results = compute_keypoints_pck(args.pred_file, args.original_gt_file, args.pck_threshold, args.match_method, args.score_threshold)
    
    # 打印主要指标
    print("\nPCK关键点检测评估结果:")
    print("="*50)
    for name, value in results['recall'].items():
        print(f"recall: {name}: {value:.4f}")
    print("="*50)
    
    # 确定输出文件路径
    if args.output_file:
        result_save_path = args.output_file
        # 创建目录（如果不存在）
        output_dir = os.path.dirname(result_save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    else:
        # 默认路径：预测文件所在目录的results.json
        output_dir = os.path.dirname(args.pred_file)
        result_save_path = os.path.join(output_dir, "results.json")
    
    # 保存完整结果
    with open(result_save_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\n完整评估结果已保存至: {result_save_path}")

if __name__ == "__main__":
    '''
    # eval PCK example
    original_gt_file=/project/vonneumann1/cl2025/keypoint_datasets/human_pose/coco/annotations/person_keypoints_val2017.json
    output_pred_file=/project/vonneumann1/cl2025/GKD/experiments/expert_benchmark/pred_file/main4_neg_samples_bilinear_tvm.json
    python evaluation_metric/coco_eval_pck.py  \
    --pred_file  ${output_pred_file} \
    --original_gt_file  ${original_gt_file}
    '''

    main()