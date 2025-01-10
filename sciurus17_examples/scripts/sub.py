#!/usr/bin/env python

import rospy
from std_msgs.msg import String

def callback(data):
    # subscribe的callback函数
    rospy.loginfo("Received message: %s", data.data)
    # 在这里添加你的可视化代码

def listener():
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("chatter", String, callback)

    rospy.spin()  # 保持节点运行

if __name__ == '__main__':
    listener()
    
    
    
    
