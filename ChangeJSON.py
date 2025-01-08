#!/usr/bin/env python
#!coding=utf-8
import os
import json

# 定义目标文件夹路径
folder_path = os.path.expanduser('~/Desktop/YOLOv7-Pytorch-Segmentation/datasetChangeJSON')

def replace_battery_labels(data):
    """
    遍历 JSON 数据，将 label 字段中的 battery1-battery5 替换为 battery
    """
    if isinstance(data, dict):
        if "label" in data and data["label"].startswith("battery"):
            print(f"正在替换 label: {data['label']} -> battery")  # 打印替换日志
            data["label"] = "battery"  # 统一替换为 battery
        for key, value in data.items():
            data[key] = replace_battery_labels(value)
    elif isinstance(data, list):
        data = [replace_battery_labels(item) for item in data]
    return data

def process_json_file(file_path):
    """
    读取 JSON 文件，替换 label 值并覆盖保存
    """
    try:
        print(f"开始处理文件: {file_path}")  # 打印当前文件路径
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 替换 label 字段
        modified_data = replace_battery_labels(data)

        # 保存修改后的数据到原文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(modified_data, f, ensure_ascii=False, indent=4)

        print(f"处理完成: {file_path}")
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")

# 遍历文件夹，处理所有 JSON 文件
if __name__ == "__main__":
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.json'):
                file_path = os.path.join(folder_path, file_name)
                process_json_file(file_path)
    else:
        print(f"文件夹不存在: {folder_path}")

