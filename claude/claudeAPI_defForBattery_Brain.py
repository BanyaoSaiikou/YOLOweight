#! /usr/bin/env python
# coding: utf-8
import os
import anthropic
import base64
import ast
import rospy
import time
from std_msgs.msg import String, Int32
import shutil
import os
import re
from PIL import Image
import rospy
from std_msgs.msg import String
#-------------------------------------------------------------------------------
image1_path = "/home/cwh/Desktop/expTF/module1/0.jpg"
image2_path = "/home/cwh/Desktop/expTF//module1/1.jpg"

image3_path = "/home/cwh/Desktop/expTF/module1/2.jpg"
image4_path = "/home/cwh/Desktop/expTF/module1/3.jpg"


rgb_folder = '/home/cwh/Desktop/workspace/image/RGB'  # Path to RGB folder
predict_seg_folder = '/home/cwh/Desktop/YOLOv7-Pytorch-Segmentation/runs/predict-seg'  # Original exp folder
destination_folder = '/home/cwh/Desktop/expTF/module1'        # Destination to save JPG files


actionBEDONE = "Tidy up the desk surface"
History_log = []

#"Put <blue_cube> into bottom drawer."
#print("User Instructions:")

client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key="",
)
    
def generate_image_Descriptions(image1_path):
    # 读取图像并进行编码
    with open(image1_path, "rb") as image_file:
        image1_data = base64.b64encode(image_file.read()).decode("utf-8")
    


def generate_VLM0(actionBEDONE,History_log, image1_path, image2_path,image3_path, image4_path):
    with open(image1_path, "rb") as image_file:
        image1_data = base64.b64encode(image_file.read()).decode("utf-8")
    
    with open(image2_path, "rb") as image_file:
        image2_data = base64.b64encode(image_file.read()).decode("utf-8")

    with open(image3_path, "rb") as image_file:
        image3_data = base64.b64encode(image_file.read()).decode("utf-8")
    
    with open(image4_path, "rb") as image_file:
        image4_data = base64.b64encode(image_file.read()).decode("utf-8")
#-------------------------------------------------------------------------------




    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2000,
        temperature=1,
        system="You are a robot assistant equipped with a Vision-Language Model (VLM) that collaborates with humans to complete core tasks. You do not have direct access to the explicit intentions of the human operator and must infer these intentions based on the following information:\n\n    Core Task: The goal of the core task to be accomplished. \n\n    Human-pre-Action Image:\n    ・The state of the workspace or scene before the human performs any actions, with each item labeled. Labels will help identify and reference specific objects.\n    ・If no actions have been performed yet, the Pre-Action Image represents the initial state of the workspace or scene.\n    ・If actions have already been performed, the Pre-Action Image reflects the state of the workspace or scene after the robot's most recent action recorded in the Action Execution Log.\n\n    Human-post-Action Image:\n    ・The state of the workspace or scene after the human has performed an action, with labeled items to aid in pinpointing changes.\n    ・The Post-Action Image represents the state of the workspace after human intervention, showing the changes made relative to the Pre-Action Image. \n\n    Action Execution Log: Keep a detailed record of all actions performed by humans and robots at each step. This log will serve as a reference for future steps, helping to avoid redundant or ineffective actions. The state of the workspace or scene after the most recent action recorded in the Action Execution Log will be reflected in the Pre-Action Image for subsequent steps.\n\nYour responsibilities are as follows:\n\n    Describe Human Action: Based on the changes between the pre-action and post-action images, describe the action taken by the human that caused the observed change.（Humans can perform multi-step operations on objects）\n\n    Explain Human Action: Explain the reasoning that led you to this conclusion.\n\n    Inferring human intention: Based on the core task, the expected human action (operation standard) is inferred by comparing the images before and after the action, the reasoning history, the action execution log, and observing the characteristics of each object in the image (color, shape, etc.).\n\n    Evaluate and Explain Intent: \"Explain your reasoning process and assess whether this inferred intent is beneficial to achieving the core task. If the inferred intent appears incorrect or deviates from the task objective, provide reasonable suggestions for adjustment. Stay vigilant if a set of actions repeats—consider whether the human might have other intentions and evaluate whether those intentions are reasonable.\n\n    Generate the Next Actions for the Robot:\n        Based on the inferred human intent and the core task, suggest effective, focused actions for the robot to assist the human in completing the core task.\n        Each suggested action should affect only one object at a time and avoid any actions already logged in the action execution history.\n        The robot first action should focus on a single object that needs immediate intervention based on the human's intent.\n\n{\n  \"Human Action Description\": \"Based on the changes between the pre-action and post-action images, Describe the human's action in a clear and concise format, using a structure like 'verb the noun to a specific state'.\",\n  \"Inferred Intent\": \"Based on the core task and image comparisons, infer both the specific intent behind the human's current action and the broader global intent, stated plainly, to describe how all actions contribute to achieving the core task.（Please describe the appearance characteristics of each item details，Predict which systematic approach the human is implementing. and Provide a detailed explanation of the reasoning process.）Stay vigilant if a set of actions repeats—consider whether the human might have other intentions and evaluate whether those intentions are reasonable.If you find that you cannot change the human's mind, you can appropriately obey the human's mind in order to continue the mission.\",\n  \"Reasoning Process\": \"A detailed explanation of the reasoning process.\",\n  \"Is Intent Beneficial to Core Task\": \"Analyze whether the inferred intent is effective and helpful for accomplishing the core task.\",\n  \"Adjustment Suggestions\": \"If the inferred intent is incorrect, provide adjustment suggestions (if any).\",\n  \"Robot First Action\": \"Suggest the robot first action to assist the human in completing the core task.（ must generate a specific action，No additional explanation is needed except for the action content）concise format, using a structure like 'verb the noun to a specific state .\",\n \"Reason for the first action\": \"Provide a detailed explanation of the reasoning process.\"\n\n}\nPrioritize task completion.\nYou can only manipulate objects with labels",
        messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "\nYou will be provided with:\n1.Core Task :\n【\n   Tidy up the desk surface\n】\n\n2.Action Execution Log ：\n【\n\n】\n\n3.Human-pre-action Image （One is the original image, and the other is the image with the visual prompt added）:\n【"
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image1_data
                    }
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image2_data
                    }
                },
                {
                    "type": "text",
                    "text": "】\n\n4.Human-post-action Image （One is the original image, and the other is the image with the visual prompt added）:\n【"
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image3_data
                    }
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image4_data
                    }
                },
                {
                    "type": "text",
                    "text": "】\nPlease describe the appearance characteristics of each object(each battery) details first.（Pay attention to shape, color, text, etc.Say one by one.）,Refer to the object appearance descriptionand then analyze the task."
                }
            ]
        }
    ]
)

    print(message.content[0].text)
    print("----------------------")

    
    try:
        # 按行分割文本
        lines = (message.content[0].text).split('\n')
        
        # 初始化变量
        Human_Action_Description = ""
        Robot_First_Action = ""
        
        # 查找标记的不同可能形式
        human_action_markers = ["Human Action Description:", "Human Action Description"]
        robot_action_markers = ["Robot First Action:", "Robot First Action"]
        
        # 遍历每一行
        for i, line in enumerate(lines):
            # 检查是否找到人类动作描述
            for marker in human_action_markers:
                if marker in line:
                    # 获取下一行的内容（如果存在）
                    if i + 1 < len(lines) and "Inferred Intent" not in lines[i + 1]:
                        Human_Action_Description = lines[i + 1].strip()
                    break
            
            # 检查是否找到机器人动作
            for marker in robot_action_markers:
                if marker in line:
                    # 获取下一行的内容（如果存在）
                    if i + 1 < len(lines) and "Reason for" not in lines[i + 1]:
                        Robot_First_Action = lines[i + 1].strip()
                    break
        
        print(f"Extracted Human Action: {Human_Action_Description}")
        print(f"Extracted Robot Action: {Robot_First_Action}")
        
        return (message.content[0].text), Human_Action_Description, Robot_First_Action
        
    except Exception as e:
        print(f"Error in extract_sections: {str(e)}")
        traceback.print_exc()  # 打印详细错误信息
        return (message.content[0].text), '', ''


#！！！！！！！！！！！！！！！！！！！！！！！！！！！！11111
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
def save_image_based_on_message(data1):
    """Determine which image to save based on the received message."""
    # Define source image path (4.png in RGB folder)
    source_image_path = os.path.join(rgb_folder, '4.png')

    # Define target filenames based on data.data value
    if data1 == 100:
        save_image(source_image_path, '/home/cwh/Desktop/expTF/module1/0.jpg')  # Save as 0.jpg
        # Also handle saving 1.jpg from predict_seg_folder
        second_latest_exp_folder = find_second_latest_exp_folder(predict_seg_folder)
        if second_latest_exp_folder:
            source_image_path = os.path.join(second_latest_exp_folder, '4.png')
            save_image(source_image_path, '/home/cwh/Desktop/expTF/module1/1.jpg')  # Save as 1.jpg
    elif data1 == 200:
        rospy.sleep(3.0)
        save_image(source_image_path, '/home/cwh/Desktop/expTF/module1/2.jpg')  # Save as 2.jpg
        # Also handle saving 3.jpg from predict_seg_folder
        second_latest_exp_folder = find_second_latest_exp_folder(predict_seg_folder)
        if second_latest_exp_folder:
            source_image_path = os.path.join(second_latest_exp_folder, '4.png')
            save_image(source_image_path, '/home/cwh/Desktop/expTF/module1/3.jpg')  # Save as 3.jpg

def save_image(source_path, target_path):
    """
    Save an image from source_path to target_path, ensuring proper format conversion if necessary.
    
    Args:
        source_path (str): The path of the source image file.
        target_path (str): The path where the image should be saved.
    """
    try:
        # 检查源文件是否存在
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file does not exist: {source_path}")
        
        # 获取目标文件扩展名（jpg 或 png 等）
        target_ext = os.path.splitext(target_path)[-1].lower()

        # 如果目标文件已存在，则删除它
        if os.path.exists(target_path):
            os.remove(target_path)
        
        # 如果源文件是 PNG，且目标需要保存为 JPG，则使用 Pillow 进行格式转换
        if source_path.endswith('.png') and target_ext == '.jpg':
            with Image.open(source_path) as img:
                # 将图片转换为 RGB 模式（去除透明通道）
                rgb_img = img.convert('RGB')
                rgb_img.save(target_path, format='JPEG')
        else:
            # 否则直接复制文件
            shutil.copy(source_path, target_path)
        
        print(f"Image successfully saved to {target_path}")
    
    except Exception as e:
        print(f"Error occurred while saving image: {e}")

# 全局变量
final_received = False


def final_callback(msg):
    global final_received
    print("Received message:", msg.data)  # 调试信息
    if msg.data.strip().lower() == "final":
        final_received = True
        print("接收到 'final' 消息，程序将继续执行...")
    else:
        print(f"接收到消息: {msg.data}，但不是 'final'")

def reset_final_status():
    global final_received
    final_received = False

# 主函数
if __name__ == '__main__':
    try:
        rospy.init_node('final_subscriber_node', anonymous=True)
        intent_publisher = rospy.Publisher('/intent_topic', String, queue_size=10)#！！！！！！！！！改topic名
        # 定义订阅器
        rospy.Subscriber('final_topic12138', String, final_callback)
        rospy.sleep(1)  # 给订阅者初始化时间
        # 初始化变量
        step_count = 0
        step_count2 = 1
        inputq = 999

        while True:
            try:
                user_input = int(input("请拍摄："))
                # 第一次保存图片
                save_image_based_on_message(100)
                user_input = int(input("请执行人类步骤。输入一个数字继续（输入 999 退出）："))
                save_image_based_on_message(200)
                
                Description = generate_VLM0(actionBEDONE, History_log, image1_path, image2_path, image3_path, image4_path)
               ########
               #########发布5s
                if Description and len(Description) > 2:
                        intent_msg = String()
                        intent_msg.data = Description[2]
                                # 设置发布持续时间
                        start_time = rospy.Time.now()
                        duration = rospy.Duration(5)  # 5秒
        
                        rospy.loginfo("开始发布intent，持续5秒...")
        
                        # 持续发布5秒
                        while (rospy.Time.now() - start_time) < duration:
                            intent_publisher.publish(intent_msg)
                            #rospy.loginfo(f"Published content: {Description[2]}")
                            rospy.sleep(0.1)  # 控制发布频率
            
                        rospy.loginfo("Intent发布完成")
               #########发布5s
               ################
        
                #print("Generated Image Description: \n", Description)
                #print("Human Action Description: \n", Description[1])
                #print("Inferred Intent: \n", Description[2])
                #print("Reasoning Process: \n", Description[3])
                #print("Is Intent Beneficial to Core Task: \n", Description[4])
                #print("Robot First Action: \n", Description[5])
                #print("Reason for the first action: \n", Description[6])
                
                History_log.append(f"Step{step_count}(Human): {Description[1]}")
                History_log.append(f"Step{step_count2}(Robot): {Description[2]}")
                print(f"当前历史日志: {History_log}")
                
                step_count += 2
                step_count2 += 2
                
                print("等待finish信息")
                should_continue = True
                #while not rospy.is_shutdown() and should_continue:
                #    if final_received:
                #        rospy.loginfo("处理接收到的 'final' 信号...")
                #        should_continue = False  # 设置标志使循环结束
                #    else:
                #        rospy.sleep(0.1) # 短暂休眠避免过度占用CPU
                
                if user_input == inputq:
                    print("程序结束！")
                    break
                else:
                    print(f"你输入的是：{user_input}，程序继续执行...")
            except ValueError:
                print("无效输入，请输入一个数字！")
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        print(f"Error occurred: {e}")




