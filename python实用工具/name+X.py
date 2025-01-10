import os

# 源文件夹路径
folder_path = '~/Downloads/RGBforLAST'


# 扩展波浪号(~)
folder_path = os.path.expanduser(folder_path)

# 起始和结束编号
start_index =000001
end_index = 000073

# 目标起始编号
target_start_index = 1

# 目标文件名的起始编号
target_filename_index = target_start_index + 1

# 遍历文件夹中的文件
for i in range(start_index, end_index + 1):
    #old_filename = os.path.join(folder_path, f'image_{i}.jpg')
    old_filename = os.path.join(folder_path, f'{i}.jpg')
    new_filename = os.path.join(folder_path, f'image_{target_filename_index}.jpg')
    os.rename(old_filename, new_filename)
    print(f"{old_filename} 重命名为 {new_filename}")
    target_filename_index += 1

print("文件重命名完成")
