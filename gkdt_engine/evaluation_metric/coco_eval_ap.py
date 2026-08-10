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

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='计算关键点检测的AP指标')
    parser.add_argument('--pred_file', 
                       type=str,
                       default="/project/vonneumann1/cl2025/GKD/experiments/evaluation_for_coco/humanart_result/main4_neg_samples_bilinear_tvm/output_validation_humanart.json",
                       help='模型预测结果的JSON文件路径')
    parser.add_argument('--original_gt_file', 
                       type=str,
                       default="/project/vonneumann1/cl2025/GKD/experiments/evaluation_for_coco/humanart_result/validation_humanart.json",
                       #default="/project/vonneumann1/cl2025/keypoint_datasets/human_pose/coco/annotations/person_keypoints_val2017.json",
                       help='原始COCO标注文件路径')
    parser.add_argument('--output_file', 
                       type=str,
                       default="/project/vonneumann1/cl2025/GKD/experiments/evaluation_for_coco/AP_result/result.json",
                       help='结果保存文件路径（默认：预测文件所在目录的evaluation_results.json）')
    
    return parser.parse_args()

def compute_keypoints_ap(result_file, original_gt_file):
    """
    计算关键点检测的AP（使用完整的原始GT文件）
    
    参数:
        result_file (str): 模型预测结果的JSON文件路径
        original_gt_file (str): 原始COCO标注文件路径
    
    返回:
        dict: 包含关键点评估指标的字典
    """
    # 加载COCO格式的GT
    coco_gt = COCO(original_gt_file)

    print(f"原始GT文件统计:")
    print(f"  - 包含图像: {len(coco_gt.dataset['images'])} 张")
    print(f"  - 包含标注: {len(coco_gt.dataset['annotations'])} 条")

    gt_categories = coco_gt.dataset.get('categories', [])
    gt_num_keypoints = None
    if gt_categories and 'keypoints' in gt_categories[0]:
        gt_num_keypoints = len(gt_categories[0]['keypoints'])

    desired_kpt_num = 17
    if gt_num_keypoints is not None:
        desired_kpt_num = min(gt_num_keypoints, desired_kpt_num)

    if gt_num_keypoints is not None and gt_num_keypoints > desired_kpt_num:
        for ann in coco_gt.dataset.get('annotations', []):
            if 'keypoints' in ann:
                ann['keypoints'] = ann['keypoints'][: desired_kpt_num * 3]
                if 'num_keypoints' in ann:
                    ann['num_keypoints'] = min(ann['num_keypoints'], desired_kpt_num)
        if gt_categories and 'keypoints' in gt_categories[0]:
            gt_categories[0]['keypoints'] = gt_categories[0]['keypoints'][: desired_kpt_num]
        coco_gt.createIndex()

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
    
    # 提取所有unique的image_id（用于统计信息）
    pred_image_ids = list({ann['image_id'] for ann in pred_annotations})
    print(f"预测结果涉及 {len(pred_image_ids)} 个不同的图像")

    # -------------------------- 准备预测数据 --------------------------
    # 准备符合COCO评估要求的预测数据
    coco_predictions = []
    for ann in pred_annotations:
        if 'image_id' not in ann:
            raise ValueError("每个预测必须包含image_id字段")
        if 'keypoints' not in ann:
            raise ValueError("每个预测必须包含keypoints字段")
        
        keypoints = ann['keypoints']
        keypoints = keypoints[: desired_kpt_num * 3]

        pred = {
            "image_id": ann['image_id'],
            "category_id": ann.get('category_id', 1),  # 默认为person
            "keypoints": keypoints,
            "score": ann.get('score', 1.0)
        }
        coco_predictions.append(pred)
    
    # 使用临时文件存储预测数据
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        json.dump(coco_predictions, tmp_file)
        tmp_path = tmp_file.name
    
    try:
        # 加载预测结果并评估
        coco_dt = coco_gt.loadRes(tmp_path)
        coco_eval = COCOeval(coco_gt, coco_dt, 'keypoints')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # 整理结果
        stats_names = ['AP', 'AP@0.5', 'AP@0.75', 'AP (M)', 'AP (L)',
                      'AR', 'AR@0.5', 'AR@0.75', 'AR (M)', 'AR (L)']
        
        return {
            'metrics': dict(zip(stats_names, coco_eval.stats)),
            'precision': coco_eval.eval['precision'].tolist(),
            'recall': coco_eval.eval['recall'].tolist(),
            'gt_path': original_gt_file,  # 修改为原始GT文件路径
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
    print("关键点检测AP评估工具")
    print("="*60)
    print(f"预测文件: {args.pred_file}")
    print(f"GT文件: {args.original_gt_file}")
    print("="*60)
    
    # 执行评估
    results = compute_keypoints_ap(args.pred_file, args.original_gt_file)
    
    # 打印主要指标
    print("\n关键点检测评估结果:")
    print("="*50)
    for name, value in results['metrics'].items():
        print(f"{name}: {value:.4f}")
    print("="*50)
    print(f"评估基于原始GT文件: {results['gt_path']}")
    
    # 确定输出文件路径
    if args.output_file:
        result_save_path = args.output_file
        # 创建目录（如果不存在）
        output_dir = os.path.dirname(result_save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    else:
        # 默认路径：预测文件所在目录的evaluation_results.json
        output_dir = os.path.dirname(args.pred_file)
        result_save_path = os.path.join(output_dir, "evaluation_results.json")
    
    # 保存完整结果
    with open(result_save_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\n完整评估结果已保存至: {result_save_path}")

if __name__ == "__main__":
    '''
    # eval AP example
    original_gt_file=/project/vonneumann1/cl2025/keypoint_datasets/human_pose/coco/annotations/person_keypoints_val2017.json
    output_pred_file=/project/vonneumann1/cl2025/GKD/experiments/expert_benchmark/pred_file/main4_neg_samples_bilinear_tvm.json
    python evaluation_metric/coco_eval_ap.py  \
    --pred_file  ${output_pred_file} \
    --original_gt_file  ${original_gt_file}
    '''
    main()