#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
import os
import re
from PIL import Image

# Define paths
rgb_folder = '/home/cwh/Desktop/workspace/image/RGB'  # Path to RGB folder
predict_seg_folder = '/home/cwh/Desktop/YOLOv7-Pytorch-Segmentation/runs/predict-seg'  # Original exp folder
destination_folder = '/home/cwh/Desktop/expTF'        # Destination to save JPG files

def backup_old_file(file_path, backup_prefix="z"):
    """Backup an existing file before overwriting."""
    if os.path.exists(file_path):
        # Find next available backup name
        folder, original_filename = os.path.split(file_path)
        filename, ext = os.path.splitext(original_filename)
        
        for i in range(1, 500):  # Try z1, z2, ..., z99
            backup_filename = f"{backup_prefix}{i}{ext}"
            backup_file_path = os.path.join(folder, backup_filename)
            if not os.path.exists(backup_file_path):
                os.rename(file_path, backup_file_path)
                rospy.loginfo(f"Backed up {file_path} as {backup_file_path}")
                break

def save_image(source_image_path, target_filename):
    """Helper function to convert and save image."""
    destination_image_path = os.path.join(destination_folder, target_filename)

    # Backup old file if it exists
    backup_old_file(destination_image_path)

    if os.path.exists(source_image_path):
        try:
            image = Image.open(source_image_path)
            image = image.convert('RGB')  # Ensure it's in RGB mode
            image.save(destination_image_path, 'JPEG')
            rospy.loginfo(f"Successfully saved {source_image_path} as {destination_image_path}")
        except Exception as e:
            rospy.loginfo(f"Failed to process image: {e}")
    else:
        rospy.loginfo(f"File not found: {source_image_path}")

def save_image_based_on_message(data):
    """Determine which image to save based on the received message."""
    # Define source image path (4.png in RGB folder)
    source_image_path = os.path.join(rgb_folder, '4.png')

    # Define target filenames based on data.data value
    if data.data == 100:
        save_image(source_image_path, '0.jpg')  # Save as 0.jpg
        # Also handle saving 1.jpg from predict_seg_folder
        second_latest_exp_folder = find_second_latest_exp_folder(predict_seg_folder)
        if second_latest_exp_folder:
            source_image_path = os.path.join(second_latest_exp_folder, '4.png')
            save_image(source_image_path, '1.jpg')  # Save as 1.jpg
    elif data.data == 200:
        rospy.sleep(3.0)
        save_image(source_image_path, '2.jpg')  # Save as 2.jpg
        # Also handle saving 3.jpg from predict_seg_folder
        second_latest_exp_folder = find_second_latest_exp_folder(predict_seg_folder)
        if second_latest_exp_folder:
            source_image_path = os.path.join(second_latest_exp_folder, '4.png')
            save_image(source_image_path, '3.jpg')  # Save as 3.jpg

def find_second_latest_exp_folder(base_path):
    exp_numbers = []
    for folder in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, folder)):
            match = re.search(r'exp(\d+)', folder)
            if match:
                exp_number = int(match.group(1))
                exp_numbers.append(exp_number)

    unique_exp_numbers = sorted(set(exp_numbers))
    if len(unique_exp_numbers) >= 2:
        second_latest_exp = unique_exp_numbers[-2]
        for folder in os.listdir(base_path):
            if f'exp{second_latest_exp}' in folder:
                return os.path.join(base_path, folder)
    return None

def callback(data):
    rospy.loginfo(f"Received number: {data.data}")
    save_image_based_on_message(data)

def subscriber():
    rospy.init_node('image_saver_listener', anonymous=True)
    rospy.Subscriber('start_action', Int32, callback)
    rospy.spin()

if __name__ == '__main__':
    try:
        source_image_path = os.path.join(rgb_folder, '4.png')
        save_image(source_image_path, '0.jpg')  # Save as 0.jpg
        # Also handle saving 1.jpg from predict_seg_folder
        second_latest_exp_folder = find_second_latest_exp_folder(predict_seg_folder)
        if second_latest_exp_folder:
            source_image_path = os.path.join(second_latest_exp_folder, '4.png')
            save_image(source_image_path, '1.jpg')  # Save as 1.jpg

        subscriber()
    except rospy.ROSInterruptException:
        pass


