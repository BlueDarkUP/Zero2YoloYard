import json
import os
from typing import Dict, List

def init_coco_prediction_file(output_file: str) -> Dict:
    """初始化COCO格式的预测文件结构"""
    pred_data = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }
    return pred_data

def add_coco_predictions(
    pred_data: Dict,
    gkd_dataset,
    episode_indexes,
    predictions_o,
    original_gt_dict: Dict,
    maxvals,
    original_gt_data: Dict
) -> None:
    """
    向COCO预测数据中添加新的预测结果
    只有当原始标注中num_keypoints不为0时才添加
    
    参数:
        pred_data: 已初始化的COCO预测数据结构
        gkd_dataset: GKD数据集对象
        episode_indexes: 当前episode的索引列表
        predictions_o: 预测的关键点坐标(原始图像尺度)
        original_gt_dict: 原始COCO标注字典(id到标注的映射)
    """
    assert len(episode_indexes) == 1, "当前仅支持单个episode的处理"
    assert len(episode_indexes) == predictions_o.shape[0], "预测数量与episode索引数量不匹配"
    curr_episode_index = episode_indexes[0].item()

    # 获取当前episode对应的local_id
    current_index_list = gkd_dataset.episode_list[curr_episode_index]
    global_to_local_map = gkd_dataset.global_to_local_id_map
    local_ids = [global_to_local_map[index]['local_id'] for index in current_index_list]
    
    # 构建类别映射，便于按真实category_id取keypoints数量
    categories = original_gt_data.get('categories', [])
    category_map = {cat.get('id'): cat for cat in categories if isinstance(cat, dict)}
    
    # print("predictions_o.shape:",predictions_o.shape)
    # print("len(local_ids):",len(local_ids))
    # 转换为numpy数组
    predictions_np = predictions_o.numpy()  # B x N x 2
    
    # 为每个预测生成COCO格式的标注
    for i, local_id in enumerate(local_ids):  # i-th query image (or instance) in the episode
        original_ann = original_gt_dict[local_id]
        category_id = original_ann.get('category_id')
        image_id = original_ann.get('image_id')
        category = category_map.get(category_id, {})

        # 创建预测结果条目
        pred_ann = {
            "image_id": image_id,
            "category_id": category_id,  # 使用真实类别ID
            "keypoints": [],
            "bbox": original_ann.get('bbox', []),
            "score": original_ann.get('score', 1.0)
        }
        
        # 填充关键点数据
        num_kpts = predictions_np.shape[1]
        pred_kpts = [0.0] * (num_kpts * 3)
        for j in range(num_kpts):
            x, y = predictions_np[i, j]
            x, y = float(x), float(y)
            keypoint_score = float(maxvals[i,j])

            # 否则正常填充坐标
            pred_kpts[j*3+0] = round(x, 2)
            pred_kpts[j*3+1] = round(y, 2)
            pred_kpts[j*3+2] = round(keypoint_score, 2)
            
        pred_ann['keypoints'] = pred_kpts
        pred_data['annotations'].append(pred_ann)


def save_coco_predictions(pred_data: Dict, output_file: str) -> None:
    """保存COCO格式的预测结果到文件并打印统计信息"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 保存JSON文件
    with open(output_file, 'w') as f:
        json.dump(pred_data, f)
    
    # 打印统计信息
    total_annotations = len(pred_data['annotations'])
    total_images = len(pred_data['images'])
    category_names = [cat['name'] for cat in pred_data['categories']]
    print(f"已保存COCO格式预测结果到: {output_file}")
    print(f"总图像数量: {total_images}")
    print(f"总标注数量: {total_annotations}")
    print(f"类别: {category_names}")