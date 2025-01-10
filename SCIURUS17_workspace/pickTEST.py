#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import ast
import rospy
from std_msgs.msg import String
import moveit_commander
import geometry_msgs.msg
import actionlib
from tf.transformations import quaternion_from_euler
from control_msgs.msg import GripperCommandAction, GripperCommandGoal
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Int32
# Global variables
converted_list = None
received_signal = False
coord_list= False
subscriberX = None

blue_cubeX= None
green_cubeX= None
bottom_spaceX= None
top_left_knobX= None
top_right_knobX= None
bottom_knobX= None


pub = rospy.Publisher('start_action', Int32, queue_size=10)
pubcheck = rospy.Publisher('start_action', Int32, queue_size=10)
def callback(data):
    global converted_list, received_signal
    # Convert received string to list
    converted_list = ast.literal_eval(data.data)
    received_signal = True
    #print("!!!!!!!!!!!!!",converted_list)
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
    global blue_cubeX
    global green_cubeX
    global bottom_spaceX
    global top_left_knobX
    global top_right_knobX
    global bottom_knobX
    while data.markers is None:
        print("Waiting for x,y,z data")
        time.sleep(1)
    #rospy.loginfo("!!!data.markers",data.markers)  
    # Print the received MarkerArray message
    #rospy.loginfo("Received MarkerArray message:")
    i=0
    for marker in data.markers:
    
        #rospy.loginfo(i)
        #rospy.loginfo("Marker position: x={}, y={}, z={}".format(marker.pose.position.x+0.14979217, marker.pose.position.y+0.02789325, marker.pose.position.z+0.57816425))#for 1280x1080
        #print("------------")
        
        if i==0:
            #blue_cubeX=[marker.pose.position.x+0.14979217, marker.pose.position.y+0.02889325, marker.pose.position.z+0.57816425]
            blue_cubeX=[marker.pose.position.x+0.13579217, marker.pose.position.y+0.02989325, marker.pose.position.z+0.57816425]
            
        if i==1:
            bottom_spaceX=[marker.pose.position.x+0.14779217, marker.pose.position.y+0.02889325, marker.pose.position.z+0.57816425]
        if i==2:
            top_left_knobX=[marker.pose.position.x+0.14779217, marker.pose.position.y+0.02889325, marker.pose.position.z+0.57816425]
        if i==3:
            top_right_knobX=[marker.pose.position.x+0.14779217, marker.pose.position.y+0.02889325, marker.pose.position.z+0.57816425]
        if i==4:
            bottom_knobX=[marker.pose.position.x+0.15379217, marker.pose.position.y+0.02889325, marker.pose.position.z+0.57816425]
        if i==5:
            green_cubeX=[marker.pose.position.x+0.14779217, marker.pose.position.y+0.02889325, marker.pose.position.z+0.57816425]
        i+=1




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
    gripper_goal.command.position = -0.3#cube:-0.3 ,knob:-0.1 
    gripper.send_goal(gripper_goal)
    gripper.wait_for_result(rospy.Duration(1.0))

def grasp_object2(gripper, gripper_goal):
    gripper_goal.command.position = -0.1#cube:-0.3 ,knob:-0.1 
    gripper.send_goal(gripper_goal)
    gripper.wait_for_result(rospy.Duration(1.0))

def move_hand(arm, position):
    #print("!!!!",position)
    target_pose = geometry_msgs.msg.Pose()

    #print("!!!!!!!shan",position,type(position[2]))
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
    target_pose.position.x = position[0]+0.05
    target_pose.position.y = position[1]+0.04
    target_pose.position.z = position[2] + 0.2
    
    q = quaternion_from_euler(-3.14 / 2.0, 0.0,0.0 )  # Orientation
    target_pose.orientation.x = q[0]
    target_pose.orientation.y = q[1]
    target_pose.orientation.z = q[2]
    target_pose.orientation.w = q[3]
    
    arm.set_pose_target(target_pose)
    arm.go()
    
    target_pose.position.z = position[2]+ 0.2
    arm.set_pose_target(target_pose)
    arm.go()

def open_by_slide(arm, position):
    target_pose = geometry_msgs.msg.Pose()
    target_pose.position.x = position[0] - 0.11
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
    target_pose.position.x = position[0] + 0.11#yuan lai shi 0.13
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
    target_pose.position.x = position[0] +0.05
    target_pose.position.y = position[1]+0.04
    target_pose.position.z = position[2] +0.15
    
    q = quaternion_from_euler(-3.14 / 2.0, 0.0, 0.0)  # Orientation
    target_pose.orientation.x = q[0]
    target_pose.orientation.y = q[1]
    target_pose.orientation.z = q[2]
    target_pose.orientation.w = q[3]
    
    arm.set_pose_target(target_pose)
    arm.go()
    
    
def map_action(method,object, arm, gripper, gripper_goal):
        #print("!!!!obj",object)
    # Initialize robot and gripper

        if method == 'move_hand()':
            move_hand(arm, object)
        elif method == 'move_hand2()':
            move_hand2(arm, object)
        elif method == 'grasp_object()':
            grasp_object(gripper, gripper_goal)

        elif method == 'grasp_object2()':
            grasp_object2(gripper, gripper_goal)

        elif method == 'release_object()':
            release_object(gripper, gripper_goal)

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
            rospy.loginfo("未知的动作")

def map_object(object):

        blue_cube = blue_cubeX  # TEST
        green_cube = green_cubeX  # TEST
        bottom_space =  bottom_spaceX# TEST
        top_left_knob =top_left_knobX # TEST
        top_right_knob =top_right_knobX # TEST
        bottom_knob =bottom_knobX # TEST
        #print("!!!!!!!!!!!!!!!!!",coord_list)
    # Initialize robot and gripper

        if object == 'blue_cube':
            object =blue_cube
        elif object == 'green_cube':
            object =green_cube
        elif object == 'bottom_space':
            object =bottom_space
        elif object == 'top-left_knob':
            object =top_left_knob
        elif object == 'top-right_knob':
            object =top_right_knob
        elif object == 'bottom_knob':
            object =bottom_knob
        elif object == '':
            object =[]
        else:
            rospy.loginfo("未知的Object")
        #print("!!!!!!!!!!!!testtestshan",object)
        return object

xxx3open=[]    
xxx3close=[]         
def process_action_batch(i,batch, arm, gripper, gripper_goal):
        
    action_publisher = rospy.Publisher('action2_topic', String, queue_size=10)
    xxx=blue_cubeX  # for attanch
    xxx2=bottom_spaceX  # for slide
    xxx3=top_left_knobX  # for slide   
    xxx4=top_right_knobX  # for slide  
    xxx5=bottom_knobX  # for slide 
    xxx6=green_cubeX  # for attanch
    print("step:",i)
    print("blue_cubeX:",xxx)
    print("bottom_spaceX:",xxx2)
    print("top-left_knobX:",xxx3)
    print("top-right_knobX:",xxx4)
    print("bottom_knobX:",xxx5)
    print("green_cubeX:",xxx6)
    print(len(batch))
    print(batch)
    count=1
    while  count!=len(batch):



        action_publisher.publish(str(batch[count][1]))#for UI

        if (batch[count][2])=='top-left_knob':
            global xxx3open 
            xxx3open=map_object(batch[count][2])
            #print("111!!!!!!!kankankankankan",xxx3open)
        if (batch[count][2])=='top-right_knob':
            global xxx3open 
            xxx3open=map_object(batch[count][2])
            #print("222!!!!!!!kankankankankan",xxx3open)
        if (batch[count][2])=='bottom_knob':
            global xxx3open 
            xxx3open=map_object(batch[count][2])
            #print("111!!!!!!!kankankankankan",xxx3open)



        if (batch[count][2])=='top-left_knob':
            global xxx3close 
            xxx3close=map_object(batch[count][2])
            #print("333!!!!!!!kankankankankan",xxx3close)
        if (batch[count][2])=='top-right_knob':
            global xxx3close 
            xxx3close=map_object(batch[count][2])
            #print("444!!!!!!!kankankankankan",xxx3close)
        if (batch[count][2])=='bottom_knob':
            global xxx3close 
            xxx3close=map_object(batch[count][2])
            #print("333!!!!!!!kankankankankan",xxx3close)
#------------------------------------------------
        print([count+1])
        if count + 1 < len(batch) and batch[count+1][1] == 'attach_to_plane()' and batch[count][1] == 'move_hand()':
            print(batch[(count)][1],"!!!!!")
            print(batch[(count)][2],"!!!!!!")
            map_action('move_hand2()',xxx2, arm, gripper, gripper_goal)
            rospy.sleep(1.0)
    #print(map_object(batch[1][2]))

        elif batch[count][1] == 'grasp_object()' and batch[count-1][2] == 'top-left_knob':
            print(batch[(count)][1],"!!!!!")
            print(batch[(count)][2],"!!!!!!")
            print(xxx3,"!!!!!!")
            map_action('grasp_object2()',xxx3, arm, gripper, gripper_goal)
            rospy.sleep(1.0)

        elif batch[count][1] == 'grasp_object()' and batch[count-1][2] == 'top-right_knob':
            print(batch[(count)][1],"!!!!!")
            print(batch[(count)][2],"!!!!!!")
            map_action('grasp_object2()',xxx4, arm, gripper, gripper_goal)
            rospy.sleep(1.0)


        elif batch[count][1] == 'grasp_object()' and batch[count-1][2] == 'bottom_knob':
            print(batch[(count)][1],"!!!!!")
            print(batch[(count)][2],"!!!!!!")
            map_action('grasp_object2()',xxx5, arm, gripper, gripper_goal)
            rospy.sleep(1.0)




        elif batch[count][1] == 'detach_from_plane()':
            print(batch[count][1],xxx)
            map_action(batch[count][1],xxx, arm, gripper, gripper_goal)
            rospy.sleep(1.0)
        elif batch[count][1] == 'attach_to_plane()':
            print(batch[count][1],xxx2)
            map_action(batch[count][1],xxx2, arm, gripper, gripper_goal)
            rospy.sleep(1.0)

       #==========
        elif batch[count][1] == 'open_by_slide()':
            print(batch[count][1],xxx3open)
            map_action(batch[count][1],xxx3open, arm, gripper, gripper_goal)
            rospy.sleep(1.0)
        elif batch[count][1] == 'close_by_slide()':
            print(batch[count][1],xxx)
            map_action(batch[count][1],xxx3close, arm, gripper, gripper_goal)
            rospy.sleep(1.0)
       #==========
        elif batch[count][1] == 'robot_initial_pose()':
            print(batch[count][1],[])
            map_action(batch[count][1],xxx, arm, gripper, gripper_goal)
        else:
            print(batch[count][1],map_object(batch[count][2]))
            map_action(batch[count][1],map_object(batch[count][2])   , arm, gripper, gripper_goal)
            
            rospy.sleep(1.0)

        #print("test!",batch[count][1])

        count+=1



    
def main():
    global received_signal
    global user_check_signal

    rospy.init_node("sciurus17_pick_and_place_controller", anonymous=True)

    # Subscribe to robot_actions_topic
    rospy.Subscriber('robot_actions_topic', String, callback)
    rospy.Subscriber('chatter', String, callback2)
    rospy.Subscriber("/sciurus17/example/my_object", MarkerArray, callback3)


    action_publisher = rospy.Publisher('action2_topic', String, queue_size=10)#!!!!!!!!!!!!!!!!
    action_publisher3 = rospy.Publisher('action3_topic3', String, queue_size=10)#!!!!!!!!!!!!!!!!

    check_pub = rospy.Publisher('user_check_topic', Int32, queue_size=10)
    rospy.sleep(0.1) 
    user_check_signal = None  # 用于存储用户确认信号

    def user_input_callback(data):
        global user_check_signal
        user_check_signal = data.data  # 更新用户确认信号
        print("user_check_signal", user_check_signal)

    rospy.Subscriber('user_check_topic', Int32, user_input_callback)

    # Initialize robot and gripper
    robot, arm = initialize_robot()
    gripper, gripper_goal = initialize_gripper()
    rospy.sleep(1.0)

    # 1. Initial position
    robot_initial_pose(arm, "l_arm_waist_init_pose")
    gripper_goal.command.position = -0.1
    gripper.send_goal(gripper_goal)
    gripper.wait_for_result(rospy.Duration(1.0))

    # 2. Release object
    release_object(gripper, gripper_goal)

    # Open gripper
    gripper_goal.command.position = -0.9
    gripper.send_goal(gripper_goal)
    gripper.wait_for_result(rospy.Duration(1.0))
    print("received_signal", received_signal)

    # Keep node running and check for received signal
    while not rospy.is_shutdown():
        if received_signal:
            # Received signal; process and execute the actions
            if converted_list is not None:
                rospy.loginfo("Processing received data...")
            else:
                rospy.loginfo("No data received.")

            i = 0
            while i < len(converted_list):
                global pub
                global pubcheck
                rospy.loginfo("等待订阅者连接...")
                while pub.get_num_connections() == 0:
                    rospy.sleep(0.1)

                # 发布数字 100
                rospy.loginfo("保存动作前图像")
                pub.publish(100)
                rospy.sleep(5.0)

                print("converted_list", converted_list, len(converted_list))
                batch = converted_list[i]

                converted_list.append([[len(converted_list), 'Finish']])#################FINAL

                #print("high-level action:",converted_list[i][0][1])
                action_publisher.publish(str(converted_list[i][0][1]))#for UI
                action_publisher3.publish(str(converted_list[i][0][1]))#for UI
                #print("low-level action:",converted_list[i])
                #print("test action:",converted_list[0][0][1])
                #print("test action:",converted_list[1][0][1])
                #print("test action:",converted_list[2][0][1])
                #print("test action:",converted_list[3][0][1])

                process_action_batch(i, batch, arm, gripper, gripper_goal)

                rospy.loginfo("保存动作后图像")
                rospy.sleep(10.0)
                pub.publish(200)

                # Publish check T/F
                rospy.loginfo("等待订阅者check连接...")
                while pubcheck.get_num_connections() == 0:
                    rospy.sleep(0.1)
                rospy.loginfo("发布check请求")

                pubcheck.publish(1)  # 使用 pubcheck 发布确认请求
                rospy.sleep(3.0)

                print("user_check_signal", user_check_signal, type(user_check_signal))

                # 等待用户输入信号
                rospy.loginfo("等待check确认...")
                rate = rospy.Rate(10)  # 10 Hz
                while user_check_signal is None:
                    rate.sleep()  # 等待信号更新
                    #rospy.loginfo("user_check_signal: {}, type: {}".format(user_check_signal, type(user_check_signal)))

                print("check T or F")
                if user_check_signal == 10:
                    print("the number of action ", i)
                    i += 1  # 继续到下一批
                    print("action success. continue next action.")
                    user_check_signal = None  # 重置用户确认信号
                elif user_check_signal == 20:
                    rospy.sleep(25.0)
                    print("the number of action ", i)

                    # Initialize robot and gripper
                    robot, arm = initialize_robot()
                    gripper, gripper_goal = initialize_gripper()
                    rospy.sleep(1.0)
#--------------------------------------------------------
# robot init pose

                    # 2. Release object
                    release_object(gripper, gripper_goal)

                    # Open gripper
                    gripper_goal.command.position = -0.9
                    gripper.send_goal(gripper_goal)
                    gripper.wait_for_result(rospy.Duration(1.0))
                    print("received_signal", received_signal)
#--------------------------------------------------------
                    i = 0  # 不改变i
                    print("action failed. wait new action sequence.")

                    user_check_signal = None  # 重置用户确认信号
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

