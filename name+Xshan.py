import os

# 源文件夹路径
folder_path = '~/Downloads/RGBforLAST/無題のフォルダー'
folder_path = os.path.expanduser(folder_path)

# 检查文件夹是否存在
if not os.path.exists(folder_path):
    print(f"文件夹 {folder_path} 不存在！")
    exit()

# 遍历1到73的数字
for i in range(1, 1500):
    # 原文件名：使用zfill(6)来生成6位数的字符串，如000001
    old_filename = os.path.join(folder_path, f'{str(i).zfill(6)}.jpg')
    # 新文件名：直接使用数字i
    new_filename = os.path.join(folder_path, f'image_{i}.jpg')
    
    if not os.path.exists(old_filename):
        print(f"警告：文件 {old_filename} 不存在！")
        continue
        
    os.rename(old_filename, new_filename)
    print(f"{old_filename} 重命名为 {new_filename}")

print("文件重命名完成")