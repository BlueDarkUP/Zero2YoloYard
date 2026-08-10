import json
data = json.load(open('/project/vonneumann1/cl2025/keypoint_datasets/vehicle/carfusion/annotations/car_keypoints_test.json'))
print(json.dumps(data['images'][0], indent=4))