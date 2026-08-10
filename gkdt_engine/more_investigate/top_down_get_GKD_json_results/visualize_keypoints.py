import json
import os
import cv2
import random
import numpy as np
from pathlib import Path
import argparse


def get_colors(num_keypoints):
    """生成关键点的颜色列表"""
    colors = []
    for i in range(num_keypoints):
        hue = int(180 * i / num_keypoints)
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, color)))
    return colors


def visualize_keypoints(json_file, image_root, output_dir, max_images=None, kp_score_thresh=0.1, draw_skeleton=True):
    """
    可视化COCO格式JSON文件中的关键点
    
    参数:
        json_file: COCO格式的JSON文件路径
        image_root: 图像根目录
        output_dir: 输出目录
        max_images: 最多可视化多少张图像（None表示全部）
        draw_skeleton: 是否绘制骨架连接
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取JSON文件
    print(f"正在读取JSON文件: {json_file}")
    with open(json_file, 'r') as f:
        coco_data = json.load(f)
    
    # 构建image_id到图像信息的映射
    images_dict = {img['id']: img for img in coco_data['images']}
    
    # 构建image_id到annotations的映射
    image_to_anns = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_to_anns:
            image_to_anns[image_id] = []
        image_to_anns[image_id].append(ann)
    
    # 获取类别信息和骨架信息
    categories = {cat['id']: cat for cat in coco_data.get('categories', [])}
    
    # 获取关键点名称和骨架连接
    if len(categories) > 0:
        category = list(categories.values())[0]
        keypoint_names = category.get('keypoints', [])
        skeleton = category.get('skeleton', [])
        num_keypoints = len(keypoint_names)
        print(f"关键点数量: {num_keypoints}")
        print(f"关键点名称: {keypoint_names}")
    else:
        keypoint_names = []
        skeleton = []
        num_keypoints = 0
    
    # 生成关键点颜色
    kpt_colors = get_colors(num_keypoints) if num_keypoints > 0 else []
    
    print(f"总图像数量: {len(images_dict)}")
    print(f"总标注数量: {len(coco_data['annotations'])}")
    
    # 获取要处理的图像列表
    all_image_ids = list(images_dict.keys())
    # 随机打乱
    random.shuffle(all_image_ids)
    
    # 限制处理数量
    if max_images is not None:
        target_image_ids = all_image_ids[:max_images]
    else:
        target_image_ids = all_image_ids

    # 处理每张图像
    processed_count = 0
    for image_id in target_image_ids:
        # 获取该图像的所有annotations
        anns = image_to_anns.get(image_id, [])
        
        # 如果没有对应的标注，则跳过
        if len(anns) < 2:  # 这里设置为至少2个标注，避免只画一个实例的图像过于简单
            continue

        image_info = images_dict[image_id]
        
        # 构建图像路径
        image_path = os.path.join(image_root, image_info['file_name'])
        
        if not os.path.exists(image_path):
            print(f"警告: 图像不存在 - {image_path}")
            continue
        
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"警告: 无法读取图像 - {image_path}")
            continue
        
        # 获取图像尺寸并计算自适应参数
        height, width = img.shape[:2]
        
        # 根据图像宽度计算关键点大小和线条粗细
        KEYPOINT_RADIUS = max(3, int(width * 0.004))  # 图像宽度的0.4%，最小为3像素
        SKELETON_THICKNESS = max(2, int(width * 0.008))  # 图像宽度的0.8%，最小为2像素
        BORDER_THICKNESS = max(1, int(KEYPOINT_RADIUS * 0.6))  # 外圈粗细
        TEXT_SCALE = max(1, width / 2000)  # 根据图像大小调整文字比例
        TEXT_THICKNESS = max(1, int(width / 2000 * 2))

        # 在图像上绘制每个annotation的关键点
        for ann_idx, ann in enumerate(anns):
            # 获取类别名称
            cat_id = ann.get('category_id')
            cat_name = categories.get(cat_id, {}).get('name', str(cat_id))

            # 生成该instance的骨架随机一致颜色
            skeleton_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            keypoints = ann.get('keypoints', [])
            
            if len(keypoints) == 0:
                continue
            
            # 将关键点转换为 (x, y, v) 的列表
            kpts = []
            for i in range(0, len(keypoints), 3):
                if i + 2 < len(keypoints):
                    x, y, v = keypoints[i], keypoints[i+1], keypoints[i+2]
                    kpts.append((x, y, v))
            
            # 绘制骨架连接
            if draw_skeleton and len(skeleton) > 0:
                for connection in skeleton:
                    kpt1_idx = connection[0] - 1  # COCO格式从1开始
                    kpt2_idx = connection[1] - 1
                    
                    if kpt1_idx < len(kpts) and kpt2_idx < len(kpts):
                        x1, y1, v1 = kpts[kpt1_idx]
                        x2, y2, v2 = kpts[kpt2_idx]
                        
                        # 只有当两个关键点都可见或具有较高置信度时才绘制连接
                        if v1 > kp_score_thresh and v2 > kp_score_thresh:
                            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), 
                                   skeleton_color, SKELETON_THICKNESS, cv2.LINE_AA)
            
            # 绘制关键点
            for kpt_idx, (x, y, v) in enumerate(kpts):
                if v > kp_score_thresh:  # 只绘制可见或具有较高置信度的关键点
                    # 中间黑色外圈白色的圆
                    cv2.circle(img, (int(x), int(y)), KEYPOINT_RADIUS + BORDER_THICKNESS, (255, 255, 255), -1)
                    cv2.circle(img, (int(x), int(y)), KEYPOINT_RADIUS, (0, 0, 0), -1)
            
            # 绘制bbox（可选）
            if 'bbox' in ann:
                bbox = ann['bbox']
                x, y, w, h = bbox
                x, y, w, h = int(x), int(y), int(w), int(h)
                # 绿线bbox
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # 左上方外侧写上该instance的category name (白底黑字)
                (text_w, text_h), baseline = cv2.getTextSize(cat_name, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_THICKNESS)
                
                # 绘制文字背景(白色)
                cv2.rectangle(img, (x, y - text_h - 10), (x + text_w, y), (255, 255, 255), -1)
                
                # 绘制文字(黑色)
                cv2.putText(img, cat_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                           TEXT_SCALE, (0, 0, 0), TEXT_THICKNESS, cv2.LINE_AA)
        
        # 保存图像
        output_filename = os.path.basename(image_info['file_name'])
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, img)
        
        processed_count += 1
        if processed_count % 10 == 0:
            print(f"已处理 {processed_count}/{min(max_images or len(images_dict), len(images_dict))} 张图像")
    
    print(f"\n完成! 总共处理了 {processed_count} 张图像")
    print(f"结果保存在: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='可视化COCO格式JSON中的关键点')
    parser.add_argument('--json_file', type=str, 
                       default='/project/vonneumann1/cl2025/GKD/more_investigate/top_down_get_GKD_json_results/result_json_F_measure/macaque_pose_grounddino_our_results.json',
                    #    default='/project/vonneumann1/cl2025/keypoint_datasets/animal_pose/ap10k/annotations_fskd/ap10k_split_test.json',
                       help='COCO格式的JSON文件路径')
    parser.add_argument('--image_root', type=str,
                       default='/project/vonneumann1/cl2025/keypoint_datasets/animal_pose/macaque_pose/images',
                       help='图像根目录')
    parser.add_argument('--output_dir', type=str,
                       default='/project/vonneumann1/cl2025/GKD/more_investigate/top_down_get_GKD_json_results/image/macaque',
                       help='输出目录')
    parser.add_argument('--max_images', type=int, default=None,
                       help='最多可视化多少张图像（None表示全部）')
    parser.add_argument('--kp_score_thresh', type=float, default=0.05)  # suggest 0.05
    parser.add_argument('--no_skeleton', action='store_true',
                       help='不绘制骨架连接')
    
    args = parser.parse_args()
    
    visualize_keypoints(args.json_file, args.image_root, args.output_dir, 
                       args.max_images, args.kp_score_thresh, draw_skeleton=not args.no_skeleton)


if __name__ == '__main__':
    main()
