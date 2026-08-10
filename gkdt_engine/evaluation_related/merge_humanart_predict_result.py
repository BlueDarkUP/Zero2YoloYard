import json
import os
import glob

# 目录路径
directory = "/project/vonneumann1/cl2025/GKD/experiments/evaluation_for_coco/humanart_result/main4_neg_samples_bilinear_tvm/"
output_path = os.path.join(directory, "output_validation_humanart.json")

# 查找所有以output_开头的JSON文件
pattern = os.path.join(directory, "output_*.json")
json_files = glob.glob(pattern)

# 排除输出文件本身
json_files = [f for f in json_files if not f.endswith("output_validation_humanart.json")]

if not json_files:
    print(f"在目录 {directory} 中未找到以'output_'开头的JSON文件")
    exit()

print(f"找到 {len(json_files)} 个文件")

# 初始化合并字典
merged_data = {}

# 逐个读取并合并文件
for file_path in json_files:
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # 合并逻辑：将相同key的value合并为列表
    for key, value in data.items():
        if key in merged_data:
            # 如果已存在，确保当前值是列表
            if not isinstance(merged_data[key], list):
                merged_data[key] = [merged_data[key]]
            # 添加新值
            if isinstance(value, list):
                merged_data[key].extend(value)
            else:
                merged_data[key].append(value)
        else:
            # 如果是第一次遇到这个key，直接存储
            merged_data[key] = value

# 保存合并后的JSON文件
with open(output_path, 'w') as f:
    json.dump(merged_data, f, indent=2)

print(f"合并完成，保存到: {output_path}")
print(f"合并后键数: {len(merged_data)}")