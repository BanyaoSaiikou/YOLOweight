#! /usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import moveit_commander
import geometry_msgs.msg
import rosnode
import actionlib
from tf.transformations import quaternion_from_euler
from control_msgs.msg import (GripperCommandAction, GripperCommandGoal)


def main():
    rospy.init_node("sciurus17_pick_and_place_controller")
    robot = moveit_commander.RobotCommander()
    arm = moveit_commander.MoveGroupCommander("l_arm_waist_group")
    arm.set_max_velocity_scaling_factor(0.1)
    gripper = actionlib.SimpleActionClient("/sciurus17/controller2/left_hand_controller/gripper_cmd", GripperCommandAction)
    gripper.wait_for_server()
    gripper_goal = GripperCommandGoal()
    gripper_goal.command.max_effort = 2.0

    rospy.sleep(1.0)


    # SRDFに定義されている"home"の姿勢にする
    arm.set_named_target("l_arm_waist_init_pose")
    arm.go()
    gripper_goal.command.position = 0.0
    gripper.send_goal(gripper_goal)
    gripper.wait_for_result(rospy.Duration(1.0))


    # ハンドを開く
    gripper_goal.command.position = -0.4   #open
    gripper.send_goal(gripper_goal)
    gripper.wait_for_result(rospy.Duration(1.0))





#x=+0.23  y=+0.02  z=+0.647054



    target_pose = geometry_msgs.msg.Pose()
    target_pose.position.x =0.332191+0.23
    target_pose.position.y =0.00130406+0.02
    target_pose.position.z =-0.325437+0.647054
    q = quaternion_from_euler(-3.14/2.0, 0.0,0.0)  # 上方から掴みに行く場合
    target_pose.orientation.x = q[0]
    target_pose.orientation.y = q[1]
    target_pose.orientation.z = q[2]
    target_pose.orientation.w = q[3]
    arm.set_pose_target(target_pose)  # 目標ポーズ設定
    arm.go()  # 実行




# SRDFに定義されている"home"の姿勢にする
    # SRDFに定義されている"home"の姿勢にする
    arm.set_named_target("l_arm_waist_init_pose")
    arm.go()
    gripper_goal.command.position = -0.9
    gripper.send_goal(gripper_goal)
    gripper.wait_for_result(rospy.Duration(1.0))

    # SRDFに定義されている"home"の姿勢にする
    #arm.set_named_target("l_arm_waist_init_pose")
    #arm.go()
    #gripper_goal.command.position = -0.9
    #gripper.send_goal(gripper_goal)
    #gripper.wait_for_result(rospy.Duration(1.0))



#　CLOSE　


if __name__ == '__main__':

    try:
        if not rospy.is_shutdown():
            main()
    except rospy.ROSInterruptException:
        pass







"""
blue_cubeX=[marker.pose.position.x+0.13579217, marker.pose.position.y+0.02989325, marker.pose.position.z+0.57816425]
x=0.236326, y=0.0709958, z=-0.491707


    target_pose = geometry_msgs.msg.Pose()
    target_pose.position.x = 0.51
    target_pose.position.y = 0.09
    target_pose.position.z = 0.12
    q = quaternion_from_euler(-3.14/2.0, 0.0,0.0)  # 上方から掴みに行く場合
    target_pose.orientation.x = q[0]
    target_pose.orientation.y = q[1]
    target_pose.orientation.z = q[2]
    target_pose.orientation.w = q[3]
    arm.set_pose_target(target_pose)  # 目標ポーズ設定
    arm.go()  # 実行



   # ハンドを閉じる
    gripper_goal.command.position = -0.35   #close
    gripper.send_goal(gripper_goal)
    gripper.wait_for_result(rospy.Duration(1.0))


    rospy.sleep(1.0)#shan

    target_pose = geometry_msgs.msg.Pose()
    target_pose.position.x = 0.51
    target_pose.position.y = 0.09
    target_pose.position.z = 0.18
    q = quaternion_from_euler(-3.14/2.0, 0.0,0.0)  # 上方から掴みに行く場合
    target_pose.orientation.x = q[0]
    target_pose.orientation.y = q[1]
    target_pose.orientation.z = q[2]
    target_pose.orientation.w = q[3]
    arm.set_pose_target(target_pose)  # 目標ポーズ設定
    arm.go()  # 実行

    rospy.sleep(3.0)#shan

"""
