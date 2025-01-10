#!/usr/bin/env python

import rospy
from std_msgs.msg import Float64MultiArray

def callback(data):
    rospy.loginfo("Received coordinates: %s", data.data)

def listener():
    # 初始化ROS节点
    rospy.init_node('coordinate_listener', anonymous=True)

    # 订阅名为 'coordinates' 的话题，消息类型为 Float64MultiArray，当收到消息时调用 callback 函数
    rospy.Subscriber('coordinates', Float64MultiArray, callback)

    # 循环等待消息
    rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass

