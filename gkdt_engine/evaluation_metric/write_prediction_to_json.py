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
        "categories": [{
            "id": 1,
            "name": "person",
            "supercategory": "person",
            "keypoints": [
                "nose","left_eye","right_eye","left_ear","right_ear",
                "left_shoulder","right_shoulder","left_elbow","right_elbow",
                "left_wrist","right_wrist","left_hip","right_hip",
                "left_knee","right_knee","left_ankle","right_ankle"
            ]
        }]
    }
    return pred_data

def add_coco_predictions(
    pred_data: Dict,
    gkd_dataset,
    episode_indexes,
    predictions_o,
    original_gt_dict: Dict,
    maxvals 
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
    # 获取当前episode对应的local_id
    current_index_list = gkd_dataset.episode_list[episode_indexes]
    global_to_local_map = gkd_dataset.global_to_local_id_map
    local_ids = [global_to_local_map[index]['local_id'] for index in current_index_list]
    
    # print("predictions_o.shape:",predictions_o.shape)
    # print("len(local_ids):",len(local_ids))
    # 转换为numpy数组
    predictions_np = predictions_o.numpy()
    
    # 为每个预测生成COCO格式的标注
    for i in range(len(local_ids)):
        valid_keypoint_num = 0
        sum_keypoint_score = 0.0
        local_id = local_ids[i]
        if local_id in original_gt_dict:
            original_ann = original_gt_dict[local_id]

            if original_ann.get('num_keypoints', 0) != 0:
                # 创建预测结果条目
                pred_ann = {
                    "image_id": original_ann['image_id'],
                    "category_id": 1,  # person
                    "keypoints": [],
                    "score": 1.0,
                    "num_keypoints": original_ann['num_keypoints']
                }
                
                # 填充关键点数据
                pred_kpts = [0.0] * 51
                for j in range(17):
                    x, y = predictions_np[i, j]

                    keypoint_score = float(maxvals[i,j])
                    valid_keypoint_num += 1

                    # 检查原始标注中该关键点是否可见
                    if len(original_ann['keypoints']) > j*3+2 and original_ann['keypoints'][j*3+2] == 0: #(1)通过GT数据来选择哪些关键点是valid的
                    #if len(original_ann['keypoints']) > j*3+2 and keypoint_score < 0.1 : #(2)通过keypoint_score阈值判断其是否是valid的，不需要使用GT数据
                        # 如果visibility=0，强制将坐标设为0
                        pred_kpts[j*3] = 0.0
                        pred_kpts[j*3+1] = 0.0
                        pred_kpts[j*3+2] = 0
                    else:
                        # 否则正常填充坐标
                        pred_kpts[j*3] = float(x)
                        pred_kpts[j*3+1] = float(y)

                        sum_keypoint_score += keypoint_score
                        valid_keypoint_num += 1
                        
                        if len(original_ann['keypoints']) > j*3+2:
                            pred_kpts[j*3+2] = original_ann['keypoints'][j*3+2]
                        else:
                            pred_kpts[j*3+2] = 2 if (x > 0 and y > 0) else 0
                
                avrgae_valid_keypoint_score = sum_keypoint_score / valid_keypoint_num if valid_keypoint_num > 0 else 0

                pred_ann["score"] = avrgae_valid_keypoint_score

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
    print(f"已保存COCO格式预测结果到: {output_file}")
    print(f"总标注数量: {total_annotations}")