import os

# 源文件夹路径
folder_path = '~/Downloads/RGBforLAST/無題のフォルダー'
folder_path = os.path.expanduser(folder_path)

# 设置起始编号
start_number = 1004  # 从46开始编号

# 检查文件夹是否存在
if not os.path.exists(folder_path):
    print(f"文件夹 {folder_path} 不存在！")
    exit()

# 获取所有image_开头的jpg文件并排序
files = []
for filename in os.listdir(folder_path):
    if filename.startswith('image_') and filename.endswith('.jpg'):
        files.append(filename)

# 按照文件名中的数字排序
files.sort(key=lambda x: int(x.replace('image_', '').replace('.jpg', '')))

# 创建一个临时文件夹，用于存储重命名的中间文件
temp_folder = os.path.join(folder_path, 'temp')
if not os.path.exists(temp_folder):
    os.makedirs(temp_folder)

# 首先将文件重命名为临时名称（避免重命名冲突）
for i, filename in enumerate(files, 1):
    old_path = os.path.join(folder_path, filename)
    temp_path = os.path.join(temp_folder, f'temp_{i}.jpg')
    os.rename(old_path, temp_path)

# 然后将临时文件移回原文件夹并重命名为目标名称
for i in range(1, len(files) + 1):
    temp_path = os.path.join(temp_folder, f'temp_{i}.jpg')
    new_path = os.path.join(folder_path, f'image_{i + start_number - 1}.jpg')  # 调整序号
    os.rename(temp_path, new_path)
    print(f"文件已重命名为: image_{i + start_number - 1}.jpg")

# 删除临时文件夹
os.rmdir(temp_folder)

print("文件重命名完成")