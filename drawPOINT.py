
from PIL import Image, ImageDraw

# 读取图片
image_path = '/home/cwh/Desktop/workspace/image/RGB/1.png'
img = Image.open(image_path)

# 获取图像的宽度和高度
width, height = img.size

# 指定坐标
x, y =871, 764  # 替换为你想要画点的坐标
#x2, y2 = 633, 779
#x3, y3 =645, 801
#x4, y4 =618, 824
# 创建ImageDraw对象
draw = ImageDraw.Draw(img)

# 设置点的大小（以椭圆的半径表示）
point_radius = 10  # 替换为你想要的点的半径

# 画一个椭圆，模拟粗一点的点
draw.ellipse((x - point_radius, y - point_radius, x + point_radius, y + point_radius), fill="red")
#draw.ellipse((x2 - point_radius, y2 - point_radius, x2 + point_radius, y2 + point_radius), fill="red")
#draw.ellipse((x3 - point_radius, y3 - point_radius, x3 + point_radius, y3 + point_radius), fill="red")
#draw.ellipse((x4 - point_radius, y4 - point_radius, x4 + point_radius, y4 + point_radius), fill="red")
# 保存修改后的图片


# 显示图片（可选）
img.show()
