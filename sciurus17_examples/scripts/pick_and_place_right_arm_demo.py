#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 RT Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rospy
import moveit_commander
import geometry_msgs.msg
import rosnode
import actionlib
from tf.transformations import quaternion_from_euler
from control_msgs.msg import (GripperCommandAction, GripperCommandGoal)

def move_hand(obj):
    # アーム初期ポーズを表示
    arm_initial_pose = arm.get_current_pose().pose

    # 何かを掴んでいた時のためにハンドを開く
    gripper_goal.command.position = 0.9
    gripper.send_goal(gripper_goal)
    gripper.wait_for_result(rospy.Duration(1.0))

    # SRDFに定義されている"home"の姿勢にする
    arm.set_named_target("r_arm_waist_init_pose")
    arm.go()
    gripper_goal.command.position = 0.0
    gripper.send_goal(gripper_goal)
    gripper.wait_for_result(rospy.Duration(1.0))
    
    
    x1=obj[0]
    y1=obj[1]
    z1=obj[2]

        # ハンドを開く
    gripper_goal.command.position = 0.7
    gripper.send_goal(gripper_goal)
    gripper.wait_for_result(rospy.Duration(1.0))
    
        # 初期姿勢
    target_pose = geometry_msgs.msg.Pose()
    target_pose.position.x = 0.25
    target_pose.position.y = 0.0
    target_pose.position.z = 0.3
    q = quaternion_from_euler(0.0, 0.0, 3.14/2.0)  # 上方から掴みに行く場合
    target_pose.orientation.x = q[0]
    target_pose.orientation.y = q[1]
    target_pose.orientation.z = q[2]
    target_pose.orientation.w = q[3]
    arm.set_pose_target(target_pose)  # 目標ポーズ設定
    arm.go()  # 実行




 # 掴む準備をする z轴对齐
    target_pose = geometry_msgs.msg.Pose()
    target_pose.position.x = 0.25
    target_pose.position.y = 0.0
    target_pose.position.z = z1#Z1　Z1　Z1
    q = quaternion_from_euler(0.0, 0.0, 3.14/2.0)  # 上方から掴みに行く場合
    target_pose.orientation.x = q[0]
    target_pose.orientation.y = q[1]
    target_pose.orientation.z = q[2]
    target_pose.orientation.w = q[3]
    arm.set_pose_target(target_pose)  # 目標ポーズ設定
    arm.go()  # 実行
    
# 掴む準備をする xy轴对齐
    target_pose = geometry_msgs.msg.Pose()
    target_pose.position.x = x1#X１　X１ X１
    target_pose.position.y = y1#Y１　Y１　Y１
    target_pose.position.z = z1#Z1 Z1 Z1
    q = quaternion_from_euler(0.0, 0.0, 3.14/2.0)  # 上方から掴みに行く場合
    target_pose.orientation.x = q[0]
    target_pose.orientation.y = q[1]
    target_pose.orientation.z = q[2]
    target_pose.orientation.w = q[3]
    arm.set_pose_target(target_pose)  # 目標ポーズ設定
    arm.go()  # 実行
def main():
    drawer_bottom_knob=[0.5, 0.0, 0.1]

##################################################################################################################################################################
#
    move_hand(drawer_bottom_knob)
    print('success')
##################################################################################################################################################################

#def grasp-object()
 # ハンドを閉じる
    gripper_goal.command.position = 0.4
    gripper.send_goal(gripper_goal)
    gripper.wait_for_result(rospy.Duration(1.0))

##################################################################################################################################################################
#def open_by_slide(open/close)
    target_pose = geometry_msgs.msg.Pose()
    target_pose.position.x = 0.45#X１　X１ X１
    target_pose.position.y = 0.0#Y１　Y１　Y１
    target_pose.position.z = 0.1#Z1 Z1 Z1
    q = quaternion_from_euler(0.0, 0.0, 3.14/2.0)  # 上方から掴みに行く場合
    target_pose.orientation.x = q[0]
    target_pose.orientation.y = q[1]
    target_pose.orientation.z = q[2]
    target_pose.orientation.w = q[3]
    arm.set_pose_target(target_pose)  # 目標ポーズ設定
    arm.go()  # 実行
##################################################################################################################################################################
 #def release-object()
    # ハンドを開く
    gripper_goal.command.position = 0.7
    gripper.send_goal(gripper_goal)
    gripper.wait_for_result(rospy.Duration(1.0))
    
    


##################################################################################################################################################################
#def detach_from_plane(cube)

# 掴みに行く
  

    print("done")


if __name__ == '__main__':

    try:
        if not rospy.is_shutdown():
            
            rospy.init_node("sciurus17_pick_and_place_controller")
            robot = moveit_commander.RobotCommander()
            arm = moveit_commander.MoveGroupCommander("r_arm_waist_group")
            arm.set_max_velocity_scaling_factor(0.1)
            gripper = actionlib.SimpleActionClient("/sciurus17/controller1/right_hand_controller/gripper_cmd", GripperCommandAction)
            gripper.wait_for_server()
            gripper_goal = GripperCommandGoal()
            gripper_goal.command.max_effort = 2.0

            rospy.sleep(1.0)



    # アーム初期ポーズを表示
            arm_initial_pose = arm.get_current_pose().pose
 
    # 何かを掴んでいた時のためにハンドを開く
            gripper_goal.command.position = 0.9
            gripper.send_goal(gripper_goal)
            gripper.wait_for_result(rospy.Duration(1.0))

    # SRDFに定義されている"home"の姿勢にする
            arm.set_named_target("r_arm_waist_init_pose")
            arm.go()
            gripper_goal.command.position = 0.0
            gripper.send_goal(gripper_goal)
            gripper.wait_for_result(rospy.Duration(1.0))
    
    
    
    
    
            main()
            
            
            
            
            
            
    except rospy.ROSInterruptException:
        pass
