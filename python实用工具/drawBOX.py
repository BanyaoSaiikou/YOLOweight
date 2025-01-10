import cv2

def draw_box_with_label(image_path, box_coords, label, color=(255, 255, 255)):
    # 读取图像
    image = cv2.imread(image_path)

    # 获取左上角和右下角的坐标
    x_min, y_min, x_max, y_max = box_coords

    # 绘制矩形框
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

    # 字体设置
    font =cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1


    # 获取文本尺寸
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

    # 绘制标签背景矩形
    cv2.rectangle(image, (x_min, y_min - text_height - baseline), 
                  (x_min + text_width, y_min), (0, 0, 0), cv2.FILLED)

    # 添加标签
    cv2.putText(image, label, (x_min, y_min - 5), font, font_scale, color, font_thickness)

    # 显示图像
    cv2.imshow('Image with Box', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 示例使用
image_path = '/home/cwh/Downloads/2024-6-27/之后实验用/segmented_image_20240703_120407.png'
box_coords = (337, 490, 364, 520)  # 左上和右下的坐标
label = 'bottom_knob'
draw_box_with_label(image_path, box_coords, label)
