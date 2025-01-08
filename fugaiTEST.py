import os

def overwrite_files():
    # 定义文件路径
    file_mappings = {
        "/home/cwh/Desktop/expTF/2.jpg": "/home/cwh/Desktop/expTF/0.jpg",
        "/home/cwh/Desktop/expTF/3.jpg": "/home/cwh/Desktop/expTF/1.jpg"
    }
    
    for src_file, dest_file in file_mappings.items():
        try:
            if os.path.exists(src_file):
                # 打开源文件并读取内容
                with open(src_file, 'rb') as src:
                    content = src.read()

                # 写入到目标文件，覆盖目标文件
                with open(dest_file, 'wb') as dest:
                    dest.write(content)

                print(f"{src_file} 已覆盖 {dest_file}")
            else:
                print(f"{src_file} 不存在，无法覆盖 {dest_file}")
        except Exception as e:
            print(f"覆盖文件时出错: {e}")

# 调用函数
overwrite_files()

