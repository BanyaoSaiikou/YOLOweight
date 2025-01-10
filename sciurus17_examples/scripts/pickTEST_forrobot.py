#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ast
import rospy
from std_msgs.msg import String
import moveit_commander
import geometry_msgs.msg
import actionlib
from tf.transformations import quaternion_from_euler
from control_msgs.msg import GripperCommandAction, GripperCommandGoal

# Global variables
converted_list = None
received_signal = False
coord_list= False
def callback(data):
    global converted_list, received_signal
    # Convert received string to list
    converted_list = ast.literal_eval(data.data)
    received_signal = True
    #rospy.loginfo("Received signal.")
    # Optionally shut down the node if you want to stop after receiving the message
    # rospy.signal_shutdown("Received one message, shutting down.")
def callback2(data):
    global coord_list, received_signal2
    # Convert received string to list
    coord_list = ast.literal_eval(data.data)
    received_signal2 = True
    #rospy.loginfo("Received signal.")
    # Optionally shut down the node if you want to stop after receiving the message
    # rospy.signal_shutdown("Received one message, shutting down.")
    
def callback3(data):
    global coord_list3, received_signal3
    
    for marker in data.markers:
        rospy.loginfo(marker.pose.position.x  ,  marker.pose.position.y,  marker.pose.position.z)#memo 也许可以把cube，nobe分多个publish
  
    
def initialize_robot():
    robot = moveit_commander.RobotCommander()
    arm = moveit_commander.MoveGroupCommander("l_arm_waist_group")
    arm.set_max_velocity_scaling_factor(0.1)
    return robot, arm

def initialize_gripper():
    gripper = actionlib.SimpleActionClient("/sciurus17/controller2/left_hand_controller/gripper_cmd", GripperCommandAction)
    gripper.wait_for_server()
    gripper_goal = GripperCommandGoal()
    gripper_goal.command.max_effort = 2.0
    return gripper, gripper_goal

def robot_initial_pose(arm, pose_name):
    arm.set_named_target(pose_name)
    arm.go()

def release_object(gripper, gripper_goal):
    gripper_goal.command.position = -0.9
    gripper.send_goal(gripper_goal)
    gripper.wait_for_result(rospy.Duration(1.0))

def grasp_object(gripper, gripper_goal):
    gripper_goal.command.position = -0.1
    gripper.send_goal(gripper_goal)
    gripper.wait_for_result(rospy.Duration(1.0))

def move_hand(arm, position):
    target_pose = geometry_msgs.msg.Pose()
    target_pose.position.x = position[0]
    target_pose.position.y = position[1]
    target_pose.position.z = position[2] + 0.05
    
    q = quaternion_from_euler(0.0, 0.0, -3.14 / 2.0)  # Orientation
    target_pose.orientation.x = q[0]
    target_pose.orientation.y = q[1]
    target_pose.orientation.z = q[2]
    target_pose.orientation.w = q[3]
    
    arm.set_pose_target(target_pose)
    arm.go()
    
    target_pose.position.z = position[2]
    arm.set_pose_target(target_pose)
    arm.go()
    
    #朝向下
def move_hand2(arm, position):
    target_pose = geometry_msgs.msg.Pose()
    target_pose.position.x = position[0]
    target_pose.position.y = position[1]
    target_pose.position.z = position[2] + 0.05
    
    q = quaternion_from_euler(-3.14 / 2.0, 0.0,0.0 )  # Orientation
    target_pose.orientation.x = q[0]
    target_pose.orientation.y = q[1]
    target_pose.orientation.z = q[2]
    target_pose.orientation.w = q[3]
    
    arm.set_pose_target(target_pose)
    arm.go()
    
    target_pose.position.z = position[2]
    arm.set_pose_target(target_pose)
    arm.go()

def open_by_slide(arm, position):
    target_pose = geometry_msgs.msg.Pose()
    target_pose.position.x = position[0] - 0.13
    target_pose.position.y = position[1]
    target_pose.position.z = position[2]
    
    q = quaternion_from_euler(0.0, 0.0, -3.14 / 2.0)  # Orientation
    target_pose.orientation.x = q[0]
    target_pose.orientation.y = q[1]
    target_pose.orientation.z = q[2]
    target_pose.orientation.w = q[3]
    
    arm.set_pose_target(target_pose)
    arm.go()
    
    
def close_by_slide(arm, position):
    target_pose = geometry_msgs.msg.Pose()
    target_pose.position.x = position[0] + 0.13
    target_pose.position.y = position[1]
    target_pose.position.z = position[2]
    
    q = quaternion_from_euler(0.0, 0.0, -3.14 / 2.0)  # Orientation
    target_pose.orientation.x = q[0]
    target_pose.orientation.y = q[1]
    target_pose.orientation.z = q[2]
    target_pose.orientation.w = q[3]
    
    arm.set_pose_target(target_pose)
    arm.go()
 
def detach_from_plane(arm, position):
    target_pose = geometry_msgs.msg.Pose()
    target_pose.position.x = position[0] 
    target_pose.position.y = position[1]
    target_pose.position.z = position[2] +0.08
    
    q = quaternion_from_euler(0.0, 0.0, -3.14 / 2.0)  # Orientation
    target_pose.orientation.x = q[0]
    target_pose.orientation.y = q[1]
    target_pose.orientation.z = q[2]
    target_pose.orientation.w = q[3]
    
    arm.set_pose_target(target_pose)
    arm.go() 
    
def attach_to_plane(arm, position):
    target_pose = geometry_msgs.msg.Pose()
    target_pose.position.x = position[0] 
    target_pose.position.y = position[1]
    target_pose.position.z = position[2] -0.05
    
    q = quaternion_from_euler(-3.14 / 2.0, 0.0, 0.0)  # Orientation
    target_pose.orientation.x = q[0]
    target_pose.orientation.y = q[1]
    target_pose.orientation.z = q[2]
    target_pose.orientation.w = q[3]
    
    arm.set_pose_target(target_pose)
    arm.go()
    
    
def map_action(method,object, arm, gripper, gripper_goal):

    # Initialize robot and gripper
        robot, arm = initialize_robot()
        gripper, gripper_goal = initialize_gripper()
        if method == 'move_hand()':
            move_hand(arm, object)
        elif method == 'move_hand2()':
            move_hand2(arm, object)
        elif method == 'grasp_object()':
            grasp_object(gripper, gripper_goal)
        elif method == 'release_object()':
            release_object(gripper, gripper_goal)
        elif method == 'robot_initial_pose()':
            robot_initial_pose(arm, object)
        elif method == 'detach_from_plane()':
            detach_from_plane(arm, object)
        elif method == 'attach_to_plane()':
            attach_to_plane(arm, object)
        elif method == 'open_by_slide()':
            open_by_slide(arm, object)
        elif method == 'close_by_slide()':
            close_by_slide(arm, object)
        elif method == 'robot_initial_pose()':
            robot_initial_pose(arm, "l_arm_waist_init_pose")
        else:
            rospy.logwarn(f"未知的动作: {func_name}")

def map_object(object):
        blue_cube = coord_list[0]  # TEST
        bottom_knob =  coord_list[1]# TEST
        bottom_drawer =coord_list[2] # TEST
        #print("!!!!!!!!!!!!!!!!!",coord_list)
    # Initialize robot and gripper

        if object == 'blue_cube':
            object =blue_cube
        elif object == 'bottom_knob':
            object =bottom_knob
        elif object == 'bottom_drawer':
            object =bottom_drawer
        elif object == '':
            object =[]
        else:
            rospy.logwarn(f"未知的Object: {func_name}")
        return object
            
def process_action_batch(i,batch, arm, gripper, gripper_goal):
    xxx=coord_list[0]  # for attanch
    xxx2=coord_list[0]  # for slide
    xxx3=coord_list[0]  # for slide    
    print("step:",i)

    print(len(batch))
    print(batch)
    count=1
    while  count!=len(batch):
        print([count+1])
        if count + 1 < len(batch) and batch[count+1][1] == 'attach_to_plane()' and batch[count][1] == 'move_hand()':
            print(batch[(count)][1],"!!!!!")
            print(batch[(count+1)][1],"!!!!!!")
            map_action('move_hand2()',xxx3, arm, gripper, gripper_goal)
            rospy.sleep(1.0)
    #print(map_object(batch[1][2]))
        elif batch[count][1] == 'detach_from_plane()':
            print(batch[count][1],xxx)
            map_action(batch[count][1],xxx, arm, gripper, gripper_goal)
            rospy.sleep(1.0)
        elif batch[count][1] == 'attach_to_plane()':
            print(batch[count][1],xxx)
            map_action(batch[count][1],xxx, arm, gripper, gripper_goal)
            rospy.sleep(1.0)
            
       #==========
        elif batch[count][1] == 'open_by_slide()':
            print(batch[count][1],xxx)
            map_action(batch[count][1],xxx2, arm, gripper, gripper_goal)
            rospy.sleep(1.0)
        elif batch[count][1] == 'close_by_slide()':
            print(batch[count][1],xxx)
            map_action(batch[count][1],xxx2, arm, gripper, gripper_goal)
            rospy.sleep(1.0)
       #==========
        elif batch[count][1] == 'robot_initial_pose()':
            print(batch[count][1],[])
            map_action(batch[count][1],"l_arm_waist_init_pose", arm, gripper, gripper_goal)
        else:
            print(batch[count][1],map_object(batch[count][2]))
            map_action(batch[count][1],map_object(batch[count][2])   , arm, gripper, gripper_goal)
            
            rospy.sleep(1.0)
        count+=1



    
def main():
    global received_signal


    
    rospy.init_node("sciurus17_pick_and_place_controller", anonymous=True)

    # Subscribe to robot_actions_topic
    rospy.Subscriber('robot_actions_topic', String, callback)
    rospy.Subscriber('chatter', String, callback2)#2
    rospy.Subscriber('/sciurus17/example/my_object', MarkerArray, callback3)#3
    # Initialize robot and gripper
    robot, arm = initialize_robot()
    gripper, gripper_goal = initialize_gripper()

    rospy.sleep(1.0)

    # 1. Initial position
    robot_initial_pose(arm, "l_arm_waist_init_pose")
    gripper_goal.command.position = 0.0
    gripper.send_goal(gripper_goal)
    gripper.wait_for_result(rospy.Duration(1.0))

    # 2. Release object
    release_object(gripper, gripper_goal)
    
        # ハンドを開く
    gripper_goal.command.position = -0.9
    gripper.send_goal(gripper_goal)
    gripper.wait_for_result(rospy.Duration(1.0))

    # Keep node running and check for received signal
    while not rospy.is_shutdown():
        if received_signal:
            # Received signal; process and execute the actions
            if converted_list is not None:
                rospy.loginfo("Processing received data...")
                #print("Final data in main:")
                #print(converted_list)

            else:
                rospy.loginfo("No data received.")
            
            
            

            i = 0
            while i < len(converted_list):
                    batch = converted_list[i]
                    process_action_batch(i,batch, arm, gripper, gripper_goal)
        
                    user_input = input("Continue to the next batch? (1 for yes, 0 for no): ")
        
                    if user_input == '1':
                        i += 1
                    elif user_input == '0':
                        i = i
                        
                    else:
                        rospy.loginfo("Invalid input. Stopping execution.")
                        break
            
            
            rospy.sleep(3.0)
            
            
            
            
            
            # Exit the node after processing
            rospy.signal_shutdown("Completed actions after receiving signal.")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

