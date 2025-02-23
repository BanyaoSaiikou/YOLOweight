from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import cv2
import numpy as np
import torch
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
detector = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")#This process may take a while as the model is very large and it takes time to download.


path = r"/home/cwh/Downloads/yolo追加训练10-9/10-10green-cube追加数据/image_125.jpg"
image = Image.open(path)

classes = ["middle-right drawer"]
inputs = processor(text=classes, images=image, return_tensors="pt")
outputs = detector(**inputs)
 
 
target_sizes = torch.Tensor([image.size[::-1]])
 
 
predictions = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)

image = np.array(image) # convert it to numpy array
 
 
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # change from BGR to RGB
boxes, scores, labels = predictions[0]["boxes"], predictions[0]["scores"], predictions[0]["labels"]
for box, score, label in zip(boxes, scores, labels):
    cv2.rectangle(image, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), (255, 0, 0), 2)
    cv2.putText(image, f"{classes[label]}",
                (int(box[0]), int(box[1]) - 10),
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
                
cv2.imshow("Detected Image", image)


output_path = "/home/cwh/Downloads/drawer.jpg"
cv2.imwrite(output_path, image)
# 等待 'q' 键关闭窗口
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


