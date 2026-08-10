import json
import os
import cv2
import random
import numpy as np
from pathlib import Path
import argparse


def get_colors(num_keypoints):
    """Generate a color list for keypoints."""
    colors = []
    for i in range(num_keypoints):
        hue = int(180 * i / num_keypoints)
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, color)))
    return colors


def visualize_keypoints(json_file, image_root, output_dir, max_images=None, kp_score_thresh=0.05, draw_skeleton=True):
    """
    Visualize keypoints from a COCO-format JSON file.

    Args:
        json_file: Path to a COCO-format JSON file.
        image_root: Root directory for images.
        output_dir: Output directory.
        max_images: Maximum number of images to visualize (None means all).
        draw_skeleton: Whether to draw skeleton connections.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read JSON file
    print(f"Reading JSON file: {json_file}")
    with open(json_file, 'r') as f:
        coco_data = json.load(f)
    
    # Build mapping from image_id to image info
    images_dict = {img['id']: img for img in coco_data['images']}
    
    # Build mapping from image_id to annotations
    image_to_anns = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_to_anns:
            image_to_anns[image_id] = []
        image_to_anns[image_id].append(ann)
    
    # Get category and skeleton info
    categories = {cat['id']: cat for cat in coco_data.get('categories', [])}
    
    # Get keypoint names and skeleton connections
    if len(categories) > 0:
        category = list(categories.values())[0]
        keypoint_names = category.get('keypoints', [])
        skeleton = category.get('skeleton', [])
        num_keypoints = len(keypoint_names)
        print(f"Number of keypoints: {num_keypoints}")
        print(f"Keypoint names: {keypoint_names}")
    else:
        keypoint_names = []
        skeleton = []
        num_keypoints = 0
    
    # Generate keypoint colors
    # kpt_colors = get_colors(num_keypoints) if num_keypoints > 0 else []
    
    print(f"Total number of images: {len(images_dict)}")
    print(f"Total number of annotations: {len(coco_data['annotations'])}")
    
    # Get list of images to process
    all_image_ids = list(images_dict.keys())
    # Shuffle randomly
    # random.shuffle(all_image_ids)
    
    # Limit number of images processed
    if max_images is not None:
        target_image_ids = all_image_ids[:max_images]
    else:
        target_image_ids = all_image_ids

    # Process each image
    processed_count = 0
    for image_id in target_image_ids:
        # Get all annotations for this image
        anns = image_to_anns.get(image_id, [])
        
        # # Skip if there are no annotations
        # if len(anns) < 2:  # Require at least 2 annotations to avoid overly simple single-instance images
        #     continue

        image_info = images_dict[image_id]
        
        # Build image path
        image_path = os.path.join(image_root, image_info['file_name'])
        
        if not os.path.exists(image_path):
            print(f"Warning: image does not exist - {image_path}")
            continue
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: could not read image - {image_path}")
            continue
        
        # Get image dimensions and compute adaptive parameters
        height, width = img.shape[:2]
        
        # Compute keypoint sizes and line thickness based on image width
        KEYPOINT_RADIUS = max(3, int(width * 0.004))  # 0.4% of image width, minimum 3 pixels
        SKELETON_THICKNESS = max(2, int(width * 0.008))  # 0.8% of image width, minimum 2 pixels
        BORDER_THICKNESS = max(1, int(KEYPOINT_RADIUS * 0.6))  # border thickness
        TEXT_SCALE = max(1, width / 2000)  # scale text based on image size
        TEXT_THICKNESS = max(1, int(width / 2000 * 2))

        # Draw keypoints for each annotation on the image
        for ann_idx, ann in enumerate(anns):
            # Get category name
            cat_id = ann.get('category_id')
            cat_name = categories.get(cat_id, {}).get('name', str(cat_id))

            # Generate a random consistent color for this instance skeleton
            skeleton_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            keypoints = ann.get('keypoints', [])
            
            if len(keypoints) == 0:
                continue
            
            # Convert keypoints to a list of (x, y, v) tuples
            kpts = []
            for i in range(0, len(keypoints), 3):
                if i + 2 < len(keypoints):
                    x, y, v = keypoints[i], keypoints[i+1], keypoints[i+2]
                    kpts.append((x, y, v))
            
            # Draw skeleton connections
            if draw_skeleton and len(skeleton) > 0:
                for connection in skeleton:
                    kpt1_idx = connection[0] - 1  # COCO format is 1-indexed
                    kpt2_idx = connection[1] - 1
                    
                    if kpt1_idx < len(kpts) and kpt2_idx < len(kpts):
                        x1, y1, v1 = kpts[kpt1_idx]
                        x2, y2, v2 = kpts[kpt2_idx]
                        
                        # Only draw connection when both keypoints are visible or have high confidence
                        if v1 > kp_score_thresh and v2 > kp_score_thresh:
                            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), 
                                   skeleton_color, SKELETON_THICKNESS, cv2.LINE_AA)
            
            # Draw keypoints
            for kpt_idx, (x, y, v) in enumerate(kpts):
                if v > kp_score_thresh:  # only draw visible or high-confidence keypoints
                    # white outer circle with black inner fill
                    cv2.circle(img, (int(x), int(y)), KEYPOINT_RADIUS + BORDER_THICKNESS, (255, 255, 255), -1)
                    cv2.circle(img, (int(x), int(y)), KEYPOINT_RADIUS, (0, 0, 0), -1)
            
            # Draw bbox (optional)
            if 'bbox' in ann:
                bbox = ann['bbox']
                x, y, w, h = bbox
                x, y, w, h = int(x), int(y), int(w), int(h)
                # green bbox
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # draw category name at top-left outside the box (white background, black text)
                (text_w, text_h), baseline = cv2.getTextSize(cat_name, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_THICKNESS)
                
                # draw text background (white)
                cv2.rectangle(img, (x, y - text_h - 10), (x + text_w, y), (255, 255, 255), -1)
                
                # draw text (black)
                cv2.putText(img, cat_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                           TEXT_SCALE, (0, 0, 0), TEXT_THICKNESS, cv2.LINE_AA)
        
        # Save image
        output_filename = os.path.basename(image_info['file_name'])
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, img)
        
        processed_count += 1
        if processed_count % 10 == 0:
            print(f"Processed {processed_count}/{min(max_images or len(images_dict), len(images_dict))} images")
    
    print(f"\nDone! Processed {processed_count} images in total")
    print(f"Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize keypoints in COCO-format JSON')
    parser.add_argument('--json_file', type=str, 
                       default='/project/vonneumann1/cl2025/GKD/more_investigate/top_down_get_GKD_json_results/result_json_F_measure/macaque_pose_grounddino_our_results.json',
                    #    default='/project/vonneumann1/cl2025/keypoint_datasets/animal_pose/ap10k/annotations_fskd/ap10k_split_test.json',
                       help='Path to COCO-format JSON file')
    parser.add_argument('--image_root', type=str,
                       default='/project/vonneumann1/cl2025/keypoint_datasets/animal_pose/macaque_pose/images',
                       help='Root directory for images')
    parser.add_argument('--output_dir', type=str,
                       default='/project/vonneumann1/cl2025/GKD/more_investigate/top_down_get_GKD_json_results/image/macaque',
                       help='Output directory')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Maximum number of images to visualize (None means all)')
    parser.add_argument('--kp_score_thresh', type=float, default=0.05)  # suggest 0.05
    parser.add_argument('--no_skeleton', action='store_true',
                       help='Do not draw skeleton connections')
    
    args = parser.parse_args()
    
    visualize_keypoints(args.json_file, args.image_root, args.output_dir, 
                       args.max_images, args.kp_score_thresh, draw_skeleton=not args.no_skeleton)


if __name__ == '__main__':
    main()
