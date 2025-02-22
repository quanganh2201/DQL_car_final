#!/usr/bin/env python3
import rclpy
import numpy as np
import math
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gazebo_msgs.srv import DeleteEntity
from gazebo_msgs.srv import SpawnEntity
from gazebo_msgs.msg import ModelState
from rclpy.time import Time
from rclpy.node import Node
from datetime import datetime
from std_srvs.srv import Trigger
import random
from tf_transformations import euler_from_quaternion

class Control(Node):
    
    def __init__(self):
        super().__init__('run_node')
        self.delclient = self.create_client(DeleteEntity, '/delete_entity')
        self.delresult = False

        self.spawnclient = self.create_client(SpawnEntity, '/spawn_entity')
        self.req = SpawnEntity.Request()
        self.setPosPub = self.create_publisher(ModelState, '/gazebo/set_model_state', 10)
        self.velPub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_subscription = self.create_subscription(Odometry, '/wheel/odometry',
                                                          self.odom_callback,
                                                          30)
        
        self.callShutdown = False
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.action = 0  # Khởi tạo hành động

    def delete_entity(self, name):
        request = DeleteEntity.Request()
        request.name = name

        # Call the service
        future = self.delclient.call_async(request)
        future.add_done_callback(self.delete_entity_callback)

    def delete_entity_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.delresult = True
            else:
                self.delresult = False
        except Exception as e:
            self.get_logger().error(f'Service call failed: {str(e)}')

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, self.yaw = euler_from_quaternion(orientation_list)

    def timer_callback(self):
        vel_cmd = Twist()
        
        # Sử dụng biến hành động đã được cập nhật
        if self.action == 5:
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.0
        elif self.action == 6:  # quay trái
            vel_cmd.linear.x = 0.1
            vel_cmd.angular.z = -0.1
        elif self.action == 4:  # quay phai nhe
            vel_cmd.linear.x = 0.1
            vel_cmd.angular.z = 0.1
        elif self.action == 8:  # di thang
            vel_cmd.linear.x = 0.1
            vel_cmd.angular.z = 0.0



        self.velPub.publish(vel_cmd)

    def set_action(self, action):
        self.action = action

def main(args=None):
    rclpy.init(args=args)

    control_node = Control()
    
    try:
        while rclpy.ok():
            action = input("Nhập số : ")
            if action.isdigit():
                control_node.set_action(int(action))
            else:
                print("Vui lòng nhập số từ .")
            
            rclpy.spin_once(control_node)  # Cập nhật node
    except KeyboardInterrupt:
        control_node.get_logger().info("Node đã dừng lại.")

    control_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()