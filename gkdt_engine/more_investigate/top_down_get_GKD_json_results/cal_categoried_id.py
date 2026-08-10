import json
from collections import Counter


def main():
	anno_path = "/project/vonneumann1/cl2025/keypoint_datasets/animal_pose/ap10k/annotations_fskd/ap10k_grounddino_results_v2.json"

	with open(anno_path, "r") as f:
		anno_data = json.load(f)

	annotations = anno_data.get("annotations", [])
	if not annotations:
		print("No annotations found.")
		return

	counter = Counter()
	missing = 0
	for ann in annotations:
		if not isinstance(ann, dict):
			missing += 1
			continue
		cat_id = ann.get("category_id")
		if cat_id is None:
			missing += 1
			continue
		counter[cat_id] += 1

	print("category_id counts:")
	for cat_id, count in sorted(counter.items()):
		print(f"{cat_id}: {count}")

	if missing:
		print(f"missing/invalid category_id entries: {missing}")


if __name__ == "__main__":
	main()
