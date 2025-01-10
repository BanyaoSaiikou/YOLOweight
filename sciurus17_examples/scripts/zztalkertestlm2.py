#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32

def publish_user_check():
    # 初始化ROS节点
    rospy.init_node('user_check_publisher', anonymous=True)

    # 创建发布者，向'user_check_topic'发布Int32类型的消息
    pub = rospy.Publisher('user_check_topic', Int32, queue_size=10)

    # 设置发布频率
    rate = rospy.Rate(1)  # 1 Hz

    while not rospy.is_shutdown():
        try:
            # 输入10或20
            user_input = input("请输入 10 或 20 来发布到 'user_check_topic': ")
            user_input = int(user_input)
            
            if user_input in [10, 20]:
                # 创建消息并发布
                msg = Int32()
                msg.data = user_input
                pub.publish(msg)
                rospy.loginfo(f"已发布消息: {msg.data}")
            else:
                print("请输入有效的数字（10 或 20）")

        except ValueError:
            print("无效输入，请输入数字")
        
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_user_check()
    except rospy.ROSInterruptException:
        pass

