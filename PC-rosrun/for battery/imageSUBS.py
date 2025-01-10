#!/usr/bin/env python
# coding: utf-8
import os
import rospy#####接受从外面Sciurus18传来的图片信息
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


RGB_DIR = "image/RGB"
class ImageSubscriber:
    def __init__(self):
        self._bridge = CvBridge()
        self._image_sub = rospy.Subscriber("output_image", Image, self._image_callback)
        self._out = None
        self._frame_idx = 0

    def _image_callback2(self, ros_image):
        try:
            # 将ROS图像消息转换成OpenCV格式
            cv_image = self._bridge.imgmsg_to_cv2(ros_image, "bgr8")
            
            # 在这里处理接收到的图像数据，比如显示图像、保存图像等
            ##cv2.imshow("Received Image", cv_image)  # 显示图像到窗口
            ##cv2.waitKey(1)
            
            # 将图像数据保存成AVI格式
           ## if self._out is None:
                # 创建VideoWriter对象，定义视频编码器、帧率和视频大小
             ##   fourcc = cv2.VideoWriter_fourcc(*'XVID')
               ## self._out = cv2.VideoWriter('image/GETSUBSCRI/output_video.avi', fourcc, 30.0, (cv_image.shape[1], cv_image.shape[0]))
            # 写入帧
            ##self._out.write(cv_image)
            ##self._frame_idx += 1
        except CvBridgeError as e:
            rospy.logerr(e)
            return
        cv2.imwrite(os.path.join(RGB_DIR, str(self._frame_idx) + '.png'),cv_image)
        self._frame_idx += 1
        if self._frame_idx>30:
            self._frame_idx=0
        self._out.write(cv_image)
        self._image_pub.publish(self._bridge.cv2_to_imgmsg(cv_image, "bgr8"))

    def _image_callback(self, ros_image):
        try:
            # Convert ROS image message to OpenCV format
            cv_image = self._bridge.imgmsg_to_cv2(ros_image, "bgr8")
                        # 在这里处理接收到的图像数据，比如显示图像、保存图像等
            #cv2.imshow("Received Image", cv_image)  # 显示图像到窗口
            #cv2.waitKey(1)
            # Save received image as PNG
            cv2.imwrite(os.path.join(RGB_DIR, str(self._frame_idx) + '.png'), cv_image)
            
            self._frame_idx += 1
            if self._frame_idx > 30:
                self._frame_idx = 0
                
        except CvBridgeError as e:
            rospy.logerr(e)
            return

def main():
    rospy.init_node("image_subscriber")
    image_subscriber = ImageSubscriber()
    rospy.spin()

if __name__ == '__main__':
    main()

