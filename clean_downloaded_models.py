"""
clean_downloaded_models.py
用于清理本地已下载的模型权重，方便测试刚克隆项目时的状态及设置页面的下载逻辑。

特性:
1. 安全防护: 仅删除大权重文件与临时文件 (.tmp / .cache / .safetensors / .pth / .pt / .best)。
2. 保持 Git 状态: 不会删除 LocateAnything-3B 中 git 跟踪的代码和文本配置文件，完美模拟刚 git clone 后的初始状态。
3. 交互确认: 默认列出所有将删除的文件与释放的空间大小，输入 y 确认后才执行；支持 --yes / -y 参数一键确认，支持 --dry-run 仅预览。
4. 状态复查: 删除完成后自动调用 local_model_manager 重新核对所有模型状态并输出自检表格。
"""

import os
import sys
import argparse

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from config import BASE_DIR
import local_model_manager


def find_files_to_remove():
    """扫描出所有已下载的模型权重文件和临时下载缓存"""
    to_delete = []  # [(path, size_bytes, description)]
    
    # 1. 扫描单文件模型 (SAM 2.1, SAM 3, GKDT, Grounding DINO 等)
    registry = local_model_manager.get_model_registry()
    for m in registry:
        rel_path = m["path"]
        full_path = os.path.join(BASE_DIR, rel_path)
        
        if m["type"] == "file":
            # 真实权重文件
            if os.path.isfile(full_path):
                to_delete.append((full_path, os.path.getsize(full_path), f"Model weight: {m['name']}"))
            # 临时下载文件
            tmp_path = full_path + ".tmp"
            if os.path.isfile(tmp_path):
                to_delete.append((tmp_path, os.path.getsize(tmp_path), f"Temp download: {m['name']} (.tmp)"))
                
        elif m["id"] == "locate_anything":
            # LocateAnything-3B: 仅清理 safetensors 权重与下载临时文件，保留 git 跟踪的配置与 py 代码
            if os.path.isdir(full_path):
                for item in os.listdir(full_path):
                    item_path = os.path.join(full_path, item)
                    if item.endswith(".safetensors") or item.endswith(".tmp"):
                        to_delete.append((item_path, os.path.getsize(item_path), f"LocateAnything weight: {item}"))
                    elif item == ".cache" and os.path.isdir(item_path):
                        cache_size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(item_path) for f in fs)
                        to_delete.append((item_path, cache_size, "LocateAnything .cache folder"))
                        
        elif m["id"].startswith("clip_"):
            # CLIP 模型目录 (在 checkpoints/clip/ 下)
            if os.path.isdir(full_path):
                dir_size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(full_path) for f in fs)
                to_delete.append((full_path, dir_size, f"CLIP model dir: {m['name']}"))

    # 2. 检查各输出目录中的权重 (.best / .tmp) 与 .cache 目录
    output_dir = os.path.join(BASE_DIR, "gkdt_engine", "output")
    if os.path.isdir(output_dir):
        for root, dirs, files in os.walk(output_dir):
            if ".cache" in dirs:
                cache_dir = os.path.join(root, ".cache")
                c_size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(cache_dir) for f in fs)
                if not any(item[0] == cache_dir for item in to_delete):
                    to_delete.append((cache_dir, c_size, "GKDT .cache folder"))
            for f in files:
                if f.endswith(".best") or f.endswith(".tmp"):
                    f_path = os.path.join(root, f)
                    if not any(item[0] == f_path for item in to_delete):
                        to_delete.append((f_path, os.path.getsize(f_path), f"GKDT weight/tmp: {f}"))

    return to_delete


def main():
    parser = argparse.ArgumentParser(description="Clean downloaded model weights for testing fresh-clone status.")
    parser.add_argument("-y", "--yes", action="store_true", help="Automatically confirm deletion without prompt.")
    parser.add_argument("--dry-run", action="store_true", help="Scan and list files to delete without actually removing them.")
    args = parser.parse_args()

    print("=" * 80)
    print("Zero2YoloYard - Clean Downloaded Models Utility")
    print("=" * 80)
    
    targets = find_files_to_remove()
    
    if not targets:
        print("未检测到任何已下载的模型文件或缓存，当前项目已是完全未下载状态。")
        return

    total_bytes = sum(t[1] for t in targets)
    total_mb = total_bytes / (1024 * 1024)
    total_gb = total_mb / 1024

    print(f"\n检测到以下 {len(targets)} 项模型权重与缓存文件 (共计约 {total_mb:.1f} MB / {total_gb:.2f} GB):\n")
    print(f"{'#':<3} | {'类型/说明':<32} | {'大小':>10} | {'路径'}")
    print("-" * 80)
    for i, (path, size, desc) in enumerate(targets, 1):
        mb = size / (1024 * 1024)
        rel_p = os.path.relpath(path, BASE_DIR)
        print(f"{i:<3} | {desc:<32} | {mb:>8.1f}MB | {rel_p}")
    print("-" * 80)
    print(f"总计可释放磁盘空间: {total_mb:.1f} MB ({total_gb:.2f} GB)\n")

    if args.dry_run:
        print("[DRY-RUN 模式] 未执行任何删除操作。如需执行，请去掉 --dry-run 参数运行。")
        return

    if not args.yes:
        confirm = input("确定要删除以上所有已下载的模型权重文件吗？(y/N): ").strip().lower()
        if confirm != "y":
            print("操作已取消，未删除任何文件。")
            return

    print("\n正在安全删除模型权重文件...")
    import shutil
    deleted_count = 0
    for path, _, desc in targets:
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif os.path.isfile(path):
                os.remove(path)
            deleted_count += 1
            print(f"  [OK] 已删除: {os.path.relpath(path, BASE_DIR)}")
        except Exception as e:
            print(f"  [FAIL] 删除失败: {path} - {e}")

    print(f"\n删除完成！成功清理 {deleted_count}/{len(targets)} 项。")

    # 重新自检所有模型状态
    print("\n" + "=" * 80)
    print("清理后模型状态自检 (应全部显示 present=False):")
    print("=" * 80)
    print(f"{'模型 ID':<18} | {'存在状态':<8} | {'检测大小':>12} | {'模型名称'}")
    print("-" * 80)
    registry = local_model_manager.get_model_registry()
    for m in registry:
        present, size_mb = local_model_manager.check_model_presence(m)
        status_str = "READY" if present else "NOT_FOUND"
        size_str = f"{size_mb:.1f} MB"
        print(f"{m['id']:<18} | {status_str:<8} | {size_str:>12} | {m['name']}")
    print("-" * 80)
    print("提示: 您现在可以打开 Web 界面设置页面 (Local Model Management) 测试全新的下载与镜像交互！")


if __name__ == "__main__":
    main()
