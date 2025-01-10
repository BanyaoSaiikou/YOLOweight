import os
import re
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import simpledialog

def find_second_latest_exp_folder(base_path):
    exp_numbers = []

    for folder in os.listdir(base_path):
        match = re.search(r'exp(\d+)', folder)
        if match:
            exp_number = int(match.group(1))
            exp_numbers.append(exp_number)

    unique_exp_numbers = sorted(set(exp_numbers))

    if len(unique_exp_numbers) >= 2:
        second_latest_exp = unique_exp_numbers[-2]
        for folder in os.listdir(base_path):
            if f'exp{second_latest_exp}' in folder:
                return os.path.join(base_path, folder)

    return None

def list_files_in_folder(folder_path):
    full_paths = []
    if folder_path:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_paths.append(os.path.join(root, file))
    return full_paths

def display_images(reference_image_path):
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口

    second_latest_exp_folder = find_second_latest_exp_folder(base_path)
    if second_latest_exp_folder is None:
        print('没有找到符合条件的文件夹。')
        return

    image_paths = list_files_in_folder(second_latest_exp_folder)

    current_window = None  # 用于存储当前显示的窗口

def display_images(reference_image_path):
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口

    second_latest_exp_folder = find_second_latest_exp_folder(base_path)
    if second_latest_exp_folder is None:
        print('没有找到符合条件的文件夹。')
        return

    image_paths = list_files_in_folder(second_latest_exp_folder)

    current_window = None  # 用于存储当前显示的窗口
    current_image_path = None  # 当前显示的图片路径

    while True:
        for img_path in image_paths:
            try:
                img1 = Image.open(img_path)
                img2 = Image.open(reference_image_path)

                # 创建新窗口来显示图片
                if current_window is not None:
                    current_window.destroy()  # 关闭旧窗口

                current_window = tk.Toplevel(root)
                current_window.title('图片展示')

                # 显示第二张图片在左边
                img2.thumbnail((400, 400))  # 调整大小
                img2_photo = ImageTk.PhotoImage(img2)
                label2 = tk.Label(current_window, image=img2_photo)
                label2.pack(side='left')

                # 显示第一张图片在右边
                img1.thumbnail((400, 400))  # 调整大小
                img1_photo = ImageTk.PhotoImage(img1)
                label1 = tk.Label(current_window, image=img1_photo)
                label1.pack(side='right')

                current_window.update()  # 更新窗口
                
                current_image_path = img_path  # 保存当前图片路径
                reference_image_path_display = reference_image_path  # 保存参考图片路径

                print(f'当前显示的图片路径: {current_image_path}')
                print(f'参考图片路径: {reference_image_path_display}')

                # 等待用户输入
                user_input = simpledialog.askstring("输入", "输入0关闭图片，输入1更新图片，其他任意键继续:")
                if user_input == '0':
                    current_window.destroy()  # 关闭窗口
                    return  # 退出函数
                elif user_input == '1':
                    # 输出当前图片路径
                    print(f'当前显示的图片路径: {current_image_path}')
                    print(f'参考图片路径: {reference_image_path_display}')
                    # 更新文件路径
                    second_latest_exp_folder = find_second_latest_exp_folder(base_path)
                    image_paths = list_files_in_folder(second_latest_exp_folder)
                    break  # 跳出当前循环，重新开始

            except (OSError, EOFError) as e:
                print(f"无法加载图片 {img_path}: {e}")
                continue  # 跳过损坏的图片

    root.quit()  # 退出主循环
    root.destroy()  # 销毁主窗口



    root.quit()  # 退出主循环
    root.destroy()  # 销毁主窗口



base_path = '/home/cwh/Desktop/YOLOv7-Pytorch-Segmentation/runs/predict-seg'
reference_image_path = os.path.expanduser('~/Desktop/workspace/image/RGB/4.png')

display_images(reference_image_path)

