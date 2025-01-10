#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32

def publish_start_action():
    # 初始化 ROS 节点
    rospy.init_node('start_action_publisher', anonymous=True)
    
    # 创建一个发布者，发布到 'start_action' 话题
    pub = rospy.Publisher('start_action', Int32, queue_size=10)
    
    while not rospy.is_shutdown():
        # 等待用户输入
        user_input = input("请输入 1 来发布消息 (输入 'q' 退出): ")
        
        if user_input == '1':
            # 发布数字 1
            start_action_value = 1
            rospy.loginfo(f"发布数字: {start_action_value}")
            pub.publish(start_action_value)
        elif user_input.lower() == 'q':
            # 输入 'q' 退出循环
            print("退出发布...")
            break
        else:
            print("无效输入，请输入 1 或 'q' 退出。")

if __name__ == '__main__':
    try:
        publish_start_action()
    except rospy.ROSInterruptException:
        pass

