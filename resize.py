from PIL import Image

image = Image.open("/home/cwh/Desktop/workspace/image/5.png")

resized_img = image.resize((640,480))

resized_img.save("/home/cwh/Desktop/workspace/image/05.png")
