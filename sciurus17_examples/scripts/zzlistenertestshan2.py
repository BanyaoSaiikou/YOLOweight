#!/usr/bin/env python
import rospy
from std_msgs.msg import String, Int32

# 用于存储接收到的动作序列
last_action_sequence = None

def robot_actions_callback(data):
    global last_action_sequence  # 使用全局变量
    last_action_sequence = data.data  # 存储最新的动作序列

def user_check_callback(data):
    # 处理接收到的用户确认数字
    rospy.loginfo(f"接收到的用户确认数字: {data.data}")
    
    # 只有在接收到 10 或 20 时输出动作序列
    if data.data in [10, 20] and last_action_sequence is not None:
        rospy.loginfo(f"上一个接收到的动作序列: {last_action_sequence}")

def subscribe_robot_actions():
    # 初始化 ROS 节点
    rospy.init_node('robot_action_subscriber', anonymous=True)
    
    # 订阅 'robot_actions_topic' 话题，消息类型为 String
    rospy.Subscriber('robot_actions_topic', String, robot_actions_callback)
    
    # 订阅 'user_check_topic' 话题，消息类型为 Int32
    rospy.Subscriber('user_check_topic', Int32, user_check_callback)
    
    # 保持节点运行，直到节点关闭
    rospy.spin()

if __name__ == '__main__':
    try:
        subscribe_robot_actions()
    except rospy.ROSInterruptException:
        pass

