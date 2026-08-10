import argparse
import json
from typing import Any, Dict, List


def load_json(path: str) -> Any:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def save_json(data: Dict[str, Any], path: str) -> None:
	with open(path, "w", encoding="utf-8") as f:
		json.dump(data, f, ensure_ascii=False)


def build_keypoints_list(cx: float, cy: float, num_keypoints: int, v: int = 2) -> List[float]:
	keypoints: List[float] = []
	for _ in range(num_keypoints):
		keypoints.extend([float(cx), float(cy), int(v)])
	return keypoints


def convert_list_to_coco_keypoints(
	det_list_path: str,
	coco_gt_path: str,
	output_path: str,
) -> None:
	det_data = load_json(det_list_path)
	if isinstance(det_data, dict) and "annotations" in det_data:
		raise ValueError("Input already looks like COCO annotations. Please provide a list JSON file.")
	if not isinstance(det_data, list):
		raise ValueError("Input must be a JSON list of detections.")

	coco_gt = load_json(coco_gt_path)

	categories = coco_gt.get("categories", [])
	if not categories:
		raise ValueError("COCO GT file has no categories.")

	num_keypoints = len(categories[0].get("keypoints", []))
	if num_keypoints <= 0:
		raise ValueError("COCO GT file has no keypoints definition in categories.")

	annotations: List[Dict[str, Any]] = []
	for idx, det in enumerate(det_data, start=1):
		bbox = det.get("bbox")
		if not bbox or len(bbox) != 4:
			raise ValueError(f"Invalid bbox at index {idx-1}: {bbox}")

		x, y, w, h = bbox
		cx = x + w * 0.5
		cy = y + h * 0.5

		ann: Dict[str, Any] = {
			"id": det.get("id", idx),
			"image_id": det.get("image_id"),
			"category_id": det.get("category_id", 1),
			"bbox": [float(x), float(y), float(w), float(h)],
			"area": float(w) * float(h),
			"iscrowd": 0,
			"num_keypoints": num_keypoints,
			"keypoints": build_keypoints_list(cx, cy, num_keypoints, v=2),
			"segmentation": [],
		}
		if "score" in det:
			ann["score"] = det["score"]
		annotations.append(ann)

	output = {
		"info": coco_gt.get("info", {}),
		"licenses": coco_gt.get("licenses", []),
		"images": coco_gt.get("images", []),
		"categories": categories,
		"annotations": annotations,
	}

	save_json(output, output_path)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Convert detection list JSON to COCO keypoints annotation JSON.")
	parser.add_argument(
		"--det_list",
		required=True,
		help="Path to detection list JSON (each entry has bbox/image_id/category_id/score).",
	)
	parser.add_argument(
		"--coco_gt",
		required=True,
		help="Path to COCO GT keypoints JSON (used to copy info/licenses/images/categories).",
	)
	parser.add_argument(
		"--output",
		required=True,
		help="Path to output COCO keypoints annotation JSON.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	convert_list_to_coco_keypoints(args.det_list, args.coco_gt, args.output)


if __name__ == "__main__":
	main()
