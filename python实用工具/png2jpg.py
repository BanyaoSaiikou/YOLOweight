import os
from PIL import Image

def convert_png_to_jpg(folder_path):
    # 展开波浪号并检查文件夹是否存在
    folder_path = os.path.expanduser(folder_path)
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            # 构造png文件的完整路径
            png_file = os.path.join(folder_path, filename)
            # 构造jpg文件的完整路径，替换文件后缀名为.jpg
            jpg_file = os.path.splitext(png_file)[0] + '.jpg'
            
            # 打开png文件
            with Image.open(png_file) as img:
                # 将png图像转换为RGB格式
                rgb_img = img.convert('RGB')
                # 保存为jpg格式
                rgb_img.save(jpg_file)
                
            # 删除原始的png文件
            os.remove(png_file)
            print(f"Converted {filename} to {os.path.basename(jpg_file)}")

# 替换为您的文件夹路径
folder_path = '~/Downloads/RGBforLAST/5'
convert_png_to_jpg(folder_path)

