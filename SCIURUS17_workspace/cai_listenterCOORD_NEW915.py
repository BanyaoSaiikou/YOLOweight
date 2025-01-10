#!/usr/bin/env python
# coding=utf-8
# license removed for brevity
import rospy
from std_msgs.msg import Float64MultiArray
import ast
import rospy
from std_msgs.msg import String
import rospy
import math
import sys
import pickle
import copy
import socket
import cv2
import os
import glob
import numpy as np

#from scipy import spatial
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Vector3, Point
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

# for NeckYawPitch
import actionlib
from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
    JointTrajectoryControllerState
)
from trajectory_msgs.msg import JointTrajectoryPoint

# for Stacker
import moveit_commander
import actionlib
from tf.transformations import quaternion_from_euler
from control_msgs.msg import GripperCommandAction, GripperCommandGoal

class DepthTo3D(object):
    def __init__(self):
        self._bridge = CvBridge()
        self._image_sub = rospy.Subscriber("/sciurus17/camera/color/image_raw", Image, self._image_callback, queue_size=1)
        self._depth_sub = rospy.Subscriber("/sciurus17/camera/aligned_depth_to_color/image_raw", Image, self._depth_callback, queue_size=1)
        self._info_sub = rospy.Subscriber("/sciurus17/camera/aligned_depth_to_color/camera_info", CameraInfo, self._info_callback, queue_size=1)

    def _info_callback(self, info):
        #print info	
        self.cam_intrinsics = info.K
    
    def _depth_callback(self, ros_image):

        try:
            input_image = self._bridge.imgmsg_to_cv2(ros_image, "passthrough")
        except CvBridgeError as e:
            rospy.logerr(e)
            return
        self.depth_img = input_image

    def _image_callback(self, ros_image):
        try:
            input_image = self._bridge.imgmsg_to_cv2(ros_image, "bgr8")
            self.rgb_img = input_image
            #cv2.circle(input_image, (320,250), 2, (255, 255, 0), -1)
   #cv2.imshow('image', input_image)
            #cv2.waitKey(1)
        except CvBridgeError as e:
            rospy.logerr(e)
        return
#----------------
converted_list = None
subscriber = None
def callback(data):
    global converted_list  # 声明使用全局变量
    global subscriber
    # 将接收到的字符串转换为列表
    converted_list = ast.literal_eval(data.data)
    
    #rospy.spin()
    
    # 接收到一次后关闭节点
    #rospy.signal_shutdown("Received one message, shutting down.")
    #print("接收到一次,关闭节点")
    # 取消订阅
    if subscriber:
        subscriber.unregister()
        subscriber = None
        #print("接收到一次消息，取消订阅")
    

def robot_action_subscriber():
    # 初始化ROS节点
    
    global subscriber
    # 订阅 robot_actions_topic 话题
    subscriber = rospy.Subscriber('YOLOv7_topic', String, callback)
    

    # 保持节点运行，直到接收到退出信号
    #rospy.spin()

def convert_to_center_point(data_list):
    result = []
    
    # 遍历原始列表
    for item in data_list:
        label = item[0]  # 标签名
        top_left = item[1]  # 左上角坐标
        bottom_right = item[2]  # 右下角坐标
        
        # 计算中心点坐标
        center_x = int((top_left[0] + bottom_right[0]) / 2)
        center_y = int((top_left[1] + bottom_right[1]) / 2)
        center_point = [center_x, center_y]
        
        # 将 label 和中心点坐标组合并添加到结果列表
        result.append([label, center_point])
    
    return result
def update_objects(data_list):
    # 初始化各个变量为空


    bottle_cap = []
    bottle = []
    # 遍历列表，根据标签名更新相应的变量
    for item in data_list:
        label = item[0][0]  # 获取标签名
        coords = item[1]  # 获取坐标



        if label == 'bottle_cap':
            bottle_cap = coords
        elif label == 'bottle':
            bottle = coords

    return {


        "bottle_cap": bottle_cap,
        "bottle": bottle,
    }
    

    # 遍历列表，根据标签名更新相应的变量
    for item in data_list:
        label = item[0][0]  # 获取标签名
        coords = item[1]  # 获取坐标


        if label == 'bottle_cap':
            bottle_cap = coords
        elif label == 'bottle':
            bottle = coords

    return bottle_cap, bottle
#----------------
i=0
def coordinates_callback(msg):
    #global i
    #if i%2==0:
        #print("space",i)
        #i+=1
    #elif i%2==1:
        #print("knob",i)
        #i+=1
    
    global list1
    list1 = msg.data
    print("obj:",list1)
    #rospy.loginfo("Received coordinates: %s", list1)
    #print("Contents of list1:", list1)
    x = (list1[1] + list1[3]) // 2
    y = (list1[2] + list1[4]) // 2
    print("Center:", "(", x, ",", y, ")")
    #if list1[0]==7:

    #    print("CenterXYZ:",pixel_to_3d_point(int(x),int(y)))
    #print("--------------")
    print("CenterXYZ:",pixel_to_3d_point(int(x),int(y)))
def create_marker(x, y, z, marker_id):
    """
    创建一个 Marker 消息
    """
    marker = Marker()
    marker.action = Marker.ADD
    marker.pose.position.x = x
    marker.pose.position.y = y
    marker.pose.position.z = z
    marker.id = marker_id
    marker.type = Marker.SPHERE  # 可以根据需要更改 Marker 类型
    return marker
def publish_markers(pub, marker_data):
    """
    发布 MarkerArray 消息
    """
    marker_array_msg = MarkerArray()
    for marker_id, (x, y, z) in marker_data.items():
        #print("!!!!!!!!!",marker_id)
        marker = create_marker(x, y, z, marker_id)
        print("!!!!!!!!!",marker.id)
        marker_array_msg.markers.append(marker)
    #print(marker_array_msg)
    pub.publish(marker_array_msg)
def main():
    global converted_list
    pub = rospy.Publisher('/sciurus17/example/my_point', MarkerArray, queue_size=2)
    rospy.init_node('YOLOv7_topic_subscriber', anonymous=True)

    while not rospy.is_shutdown():
        try:
            robot_action_subscriber()

            if converted_list is not None:
                # 调用函数并获得结果
                results = update_objects(convert_to_center_point(converted_list))
                print(results)
                
                marker_data = {}

                # 使用 try-except 捕获坐标缺失的情况，或检查键是否存在
                try:
                    if "bottle_cap" in results:
                        X1, Y1, D1 = pixel_to_3d_point(results["bottle_cap"][0], results["bottle_cap"][1])
                        marker_data[0] = (X1, Y1, D1)
                    else:
                        print("'bottle_cap' not found")

                    if "bottle" in results:
                        X2, Y2, D2 = pixel_to_3d_point(results["bottle"][0], results["bottle"][1])
                        marker_data[1] = (X2, Y2, D2)
                    else:
                        print("'bottom_space' not found")

                
                except Exception as e:
                    print("Error occurred: ")

                # 如果 marker_data 不为空，发布 MarkerArray 消息
                if marker_data:
                    publish_markers(pub, marker_data)

            else:
                print("No data received.")
                
        except rospy.ROSInterruptException:
            pass
        
        rospy.sleep(1)  # 确保每个循环中有暂停


        

def pixel_to_3d_point(x, y):

    if depth_to_3d.depth_img is None:
        rospy.logerr("深度图像不可用。")
        return None

    if not isinstance(x, int) or not isinstance(y, int):
        rospy.logerr("像素坐标必须为整数。")
        return None

    if x < 0 or x >= depth_to_3d.depth_img.shape[1] or y < 0 or y >= depth_to_3d.depth_img.shape[0]:
        rospy.logerr("像素坐标超出范围。")
        return None
    #x=int((x*640)/1920)
    #y=int((y*480)/1080)
    

    #print(depth_to_3d.cam_intrinsics)
    depth = depth_to_3d.depth_img[y][x] / 1000.0
    #for i in range(1920):
     #   print("depth",i,":", depth_to_3d.depth_img[430][i] / 1000.0)
    #print("test",x,y,depth)  
    X = (x - depth_to_3d.cam_intrinsics[2]) * depth / depth_to_3d.cam_intrinsics[0]
    Y = (y - depth_to_3d.cam_intrinsics[5]) * depth / depth_to_3d.cam_intrinsics[4]
    print("test",X,Y,depth)     

    return X, Y, depth


if __name__ == '__main__':
    depth_to_3d = DepthTo3D()
    
    main()

