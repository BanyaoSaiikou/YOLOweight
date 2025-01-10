#! /usr/bin/env python
# coding: utf-8
import os

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

def publish_final_message():
    # 创建发布者
    pub = rospy.Publisher('final_topic12138', String, queue_size=10)
    rospy.sleep(0.1)  # 短暂等待确保连接建立
    
    # 创建并发送消息
    msg = String()
    msg.data = "final"
    pub.publish(msg)
    print("Published 'final' message")

if __name__ == '__main__':
    try:
        rospy.init_node('number_input_publisher', anonymous=True)
        
        while True:
            try:
                # 获取用户输入
                user_input = input("请输入一个数字（输入'q'退出）：")
                
                # 检查是否退出
                if user_input.lower() == 'q':
                    print("程序结束")
                    break
                
                # 尝试转换为数字
                number = float(user_input)
                print(f"你输入的数字是：{number}")
                
                # 发送'final'消息
                publish_final_message()
                
            except ValueError:
                print("无效输入，请输入一个数字！")
            
    except rospy.ROSInterruptException:
        pass
