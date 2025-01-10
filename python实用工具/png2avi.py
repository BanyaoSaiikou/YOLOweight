import cv2
import os

# 设置图片文件夹路径
image_folder = '~/Downloads/cai_bottom/cai_lefttop2'    #改路径
image_folder = os.path.expanduser(image_folder)  # 展开波浪号(~)

# 设置输出视频文件名
video_name = '~/Downloads/cai_bottom/output4.avi'    #改路径
video_name = os.path.expanduser(video_name)  # 展开波浪号(~)

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
images = sorted(images, key=lambda x: int(x.split('_')[-1].split('.')[0]))  # 按照图片序号进行排序

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# 初始化视频编码器
video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 30, (width,height))  # 设置帧速率为30帧/秒

# 遍历图片文件夹中的所有图片，并将它们写入视频
for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

# 释放视频编码器并关闭所有窗口
cv2.destroyAllWindows()
video.release()

