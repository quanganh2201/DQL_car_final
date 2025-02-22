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


from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2

import model5
import torch

import matplotlib.pyplot as plt
import threading
import sys

MIN_DISTANCE = 0.8
ERROR_DISTANCE = 0.5
XML_FILE_PATH = '/home/botcanh/dev_ws/src/two_wheeled_robot/urdf/two_wheeled_robot_copy.urdf'
# XML_FILE_PATH = '/home/botcanh/turtlebot3_ws/src/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_burger/model.sdf'
X_INIT = 0.0
Y_INIT = 0.0
THETA_INIT = 0.0
GOAL_X = 5
GOAL_Y = 5
YAW = math.atan2(GOAL_X- X_INIT, GOAL_Y - Y_INIT)
HORIZONTAL_DIS = 9999
GOAL_THRESHOLD1 = 1.0



class Env(Node):
    def __init__(self):
        super().__init__('env_node')
        self.delclient = self.create_client(DeleteEntity, '/delete_entity')
        self.delresult = False

        self.spawnclient = self.create_client(SpawnEntity, '/spawn_entity')
        self.req = SpawnEntity.Request()
        self.setPosPub = self.create_publisher(ModelState, '/gazebo/set_model_state', 10)
        self.velPub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.callShutdown = False

        self.odom_subcription = self.odom_subscription = self.create_subscription(Odometry, '/wheel/odometry',
                                                                                  self.odom_callback,
                                                                                  30)
        self.depth_subcription = self.create_subscription(Image, '/camera_link/depth/image_raw',
                                                          self.process_data_depth, 30)
        self.subscription = self.create_subscription(
            Image,
            '/camera_link/image_raw',
            self.image_callback,
            30)
        self.model = YOLO('~/yolobot/src/yolobot_recognition/scripts/yolov8n.pt')
        self.bridge = CvBridge()
        self.img_pub = self.create_publisher(Image, "/inference_result", 1)
        self.view_depth = None
        self.depth_img = None
        self.view_depth_range = 10 * np.ones([5], dtype=float) #  depth in each range

        self.done = False  # done episode or not
        self.EPISODES = 500
        self.steps = 200
        self.current_step = 0
        self.current_ep = 0
        self.ep_done = False
        self.pre_action = 0

        #odometry
        self.x_distance = GOAL_X
        self.y_distance = GOAL_Y 
        self.x_pre_distance = GOAL_X
        self.y_pre_distance = GOAL_Y
        self.yaw = YAW
        self.distance = round(math.hypot(GOAL_X, GOAL_Y)) 
        self.pre_distance = HORIZONTAL_DIS
        self.ABSOLUTE_DISTANCE = math.hypot(GOAL_X, GOAL_Y)

        self.rewards = 0
        self.step_count = 0
        self.pre_best = -100
        self.best_rewards = -100
        self.current_state = None
        self.new_state = None
        self.threshold_done1 = False

        # TRAIN PARAMETERS
        self.train_model = model5.CarDQL()
        self.num_states = 7
        self.num_actions = 5

        self.pre_epsilon = 1
        self.epsilon = 1  # 1 = 100% random actions
        self.memory = model5.ReplayMemory(self.train_model.replay_memory_size)
        self.policy_dqn = model5.DeepQNetwork(input_dims=self.num_states, fc1_dims=32, fc2_dims=32,
                                              n_actions=self.num_actions)
        self.target_dqn = model5.DeepQNetwork(input_dims=self.num_states, fc1_dims=32, fc2_dims=32,
                                              n_actions=self.num_actions)
        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

        # Policy network optimizer. "Adam" optimizer can be swapped to something else.
        self.train_model.optimizer = torch.optim.Adam(self.policy_dqn.parameters(), lr=self.train_model.learning_rate_a)

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        self.rewards_per_episode = []

        # List to keep track of epsilon decay
        self.epsilon_history = []

        timer_period = 0.15
        self.timer = self.create_timer(timer_period, self.timer_callback)

        #PLOTTING PARAMETERS AND FUNCTIONS
        self.fig, self.axes = plt.subplots(1, 1)
        self.lock = threading.Lock()

    # FOR SPAWNING MODEL IN GAZEBO
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

    def spawn_entity_callback(self, future):
        try:
            response = future.result()
            if response is not None:
                self.get_logger().info(
                    'Spawn entity response: success={}, status_message={}'.format(
                        response.success, response.status_message
                    )
                )
            else:
                self.get_logger().error('Service call failed')
        except Exception as e:
            self.get_logger().error('Service call failed: %r' % e)

    def call_spawn_entity_service(self, name, xml_file_path, x, y, z):
        try:
            with open(xml_file_path, 'r') as file:
                xml_content = file.read()

            self.req.name = name
            self.req.xml = xml_content
            self.req.initial_pose.position.x = x
            self.req.initial_pose.position.y = y
            self.req.initial_pose.position.z = z
            self.req.initial_pose.orientation.z = 0.38
            self.req.initial_pose.orientation.w = 0.92

            future = self.spawnclient.call_async(self.req)
            future.add_done_callback(self.spawn_entity_callback)
        except Exception as e:
            self.get_logger().error('Failed to call spawn_entity service: %r' % e)

    # HANDLE CAMERA STREAM
    def process_data_depth(self, data):
        self.view_depth = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        self.view_depth = np.array(self.view_depth, dtype=np.float32)
        # alpha: contrast 0 -127, beta: brightness 0 -100
        self.depth_img = cv2.convertScaleAbs(self.view_depth, alpha=10, beta=30)
        # cv2.imshow('view0', self.view_depth)

    def image_callback(self, msg):
        self.view_depth_range = 10 * np.ones([5], dtype=float)
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        results = self.model(cv_image)
        # print(cv_image.size)
        height, width, _ = cv_image.shape
        self.img_center_x = cv_image.shape[0] // 2
        img_center_y = cv_image.shape[1] // 2
        view_range = width / 5
        depth = 10
        if len(results) > 0:
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    b = box.xyxy[0].to(
                        'cpu').detach().numpy().copy()  # get box coordinates in (top, left, bottom, right) format
                    c = box.cls
                    top = int(b[0])
                    left = int(b[1])
                    bottom = int(b[2])
                    right = int(b[3])

                    center_y = (left + right) // 2
                    center_x = (top + bottom) // 2

                    # Draw the center point
                    cv2.circle(cv_image, (center_x, center_y), radius=5, color=(255, 255, 0), thickness=-1)
                    if self.view_depth is None:
                        pass
                    else:
                        # find the closest point within the box
                        for y in range(left, right):
                            for x in range(top, bottom):
                                temp = self.view_depth[y, x]
                                if temp < depth:
                                    depth = temp
                        cv2.putText(cv_image, f"{depth :.2f}m", (center_x + 5, center_y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (255, 255, 255), 2)
                        if self.view_depth_range[int((center_x - 1)/view_range)] > depth:
                            self.view_depth_range[int((center_x - 1)/view_range)] = depth

        annotated_frame = results[0].plot(labels=True)
        img_msg = self.bridge.cv2_to_imgmsg(annotated_frame)
        self.img_pub.publish(img_msg)

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.distance = math.hypot(x - GOAL_X, y - GOAL_Y)
        self.x_distance = abs(x - GOAL_X)
        self.y_distance = abs(y - GOAL_Y)
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _,_,self.yaw = euler_from_quaternion(orientation_list)
        print("diff, ",math.fabs(self.yaw - YAW))

    def getState(self):
        x_distance = self.x_distance
        y_distance = self.y_distance
        distance = self.view_depth_range
        goal_distance = self.distance
        yaw = self.yaw
        state = [distance[0], distance[1], distance[2], distance[3], distance[4], goal_distance, yaw]
        return state

    def setReward(self, state):
        done = False
        distance = state[-2]
        yaw = state[-1]
        rx_dis = 0
        ry_dis = 0
        r_dis = 0
        reward = 0
        '''
            if pre_action == None and pre_distance == None:
                pre = 0
            else:
                pre = 0
            '''
        '''
        if action == 0:  # straight
            r_action = +0.2
        else:
            r_action = -0.1

        if (pre_action == 1 and action == 2) or (pre_action == 2 and action == 1):
            r_change = -0.3
        else:
            r_change = 0
        '''
        '''
        if min_ran < MIN_DISTANCE and min_ran > ERROR_DISTANCE:
            r_ob = -0.3
        else:
            r_ob = +0.05
        '''

        #if x_pre > x_dis:
        #    rx_dis = 2**(x_dis/GOAL_X) 
        #else:
        #    rx_dis = -5 
        

        #if x_pre <= x_dis:
        #   rx_dis = -(2 ** ((GOAL_X - x_dis) / GOAL_X))

        
        #if y_pre > y_dis:
        #    ry_dis = 2**(y_dis/GOAL_Y) 
        #else:
        #    ry_dis = -5
        
        #if y_pre <= y_dis:
        #    ry_dis = -(2 ** ((GOAL_Y - y_dis) / GOAL_Y))
        if math.fabs(yaw - YAW) <= 0.5:
        	reward =  2**(distance / self.ABSOLUTE_DISTANCE)
        for temp in range(1,4):
            if state[temp] <= ERROR_DISTANCE:
                reward -= 100
                done = True
        if distance <= GOAL_THRESHOLD1:
            reward = 100
            done = True
        return reward, done

    def step(self, action, reward):
        if action == 0:
            ang_vel = 0.0
        elif action == 1:  # turn right
            ang_vel = 0.35
        elif action == 2:
            ang_vel = -0.35
        elif action == 3:
            ang_vel = 0.1
        elif action == 4:
            ang_vel = -0.1

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.25 * reward
        if vel_cmd.linear.x < 0.0:
            vel_cmd.linear.x = 0.05
        vel_cmd.angular.z = ang_vel
        self.velPub.publish(vel_cmd)

        state = self.getState()
        return state

    def reset(self):
        self.delete_entity('two_wheeled_robot')

        # SHOULD HAVE A TIMER HERE
        print("deleted")
        i = 0

        self.call_spawn_entity_service('two_wheeled_robot', XML_FILE_PATH, X_INIT, Y_INIT, THETA_INIT)
        print("respawned")
        state = self.getState()
        done = False
        self.current_step = 0
        self.ep_done = False
        self.pre_action = None
        self.pre_distance = HORIZONTAL_DIS
        self.rewards = 0
        self.view_depth_range = 10 * np.ones([5], dtype=float)
        return state, done

    #PLOTTING FUNCTIONS
    def plot_progress(self):
        # Create new graph
        plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        # rewards_curve = np.zeros(len(rewards_per_episode))
        # for x in range(len(rewards_per_episode)):
        # rewards_curve[x] = np.min(rewards_per_episode[max(0, x-10):(x+1)])
        #plt.subplot(121)  # plot on a 1 row x 2 col grid, at cell 1
        # plt.plot(sum_rewards)
        plt.plot(self.rewards_per_episode)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        #plt.subplot(122)  # plot on a 1 row x 2 col grid, at cell 2
        #plt.plot(self.epsilon_history)

        # Save plots
        plt.savefig('car_dql.png')

    def timer_callback(self):  # train
        if self.current_state is None:
            self.current_state = self.getState()
        self.current_step = self.current_step + 1
        if self.current_step < self.steps and self.ep_done == False:
            # Select action based on epsilon-greedy
            if random.random() < self.epsilon:
                # select random action
                action = np.random.choice([0, 1, 2, 3, 4])  # actions: 0=left,1=left,2=right
            else:
                # select best action
                with torch.no_grad():
                    action = self.policy_dqn(
                        self.train_model.state_to_dqn_input(self.current_state)).argmax().item()

            # Execute action
            reward, self.ep_done = self.setReward(self.current_state)
            self.new_state = self.step(action, reward)
            # Accumulate reward
            self.rewards += reward
            self.step_count += 1
            self.pre_action = action
            #self.x_pre_distance = self.current_state[-3]
            #self.y_pre_distance = self.current_state[-2]
            #self.pre_distance = self.current_state[-1]
            # Save experience into memory
            self.memory.append((self.current_state, action, self.new_state, reward, self.done))
            self.current_state = self.new_state
            print("current, step",self.current_step)
            print("adding reward", reward)
        else:
            if self.rewards < -100:
                self.rewards = -100
            elif self.rewards > 100:
                self.rewards = 100   
            self.rewards_per_episode.append(self.rewards)
            self.ep_done = True
            self.current_ep += 1
            # Graph training progress
            if (self.current_ep != 0):
                print(f'Episode {self.current_ep} Epsilon {self.epsilon}')
                print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
                with self.lock:
                    self.plot_progress()
                # self.train_model.plot_progress(self.rewards_per_episode, self.epsilon_history)
            print(self.best_rewards, self.rewards)
            # AVOID ERROR SPAWN
            if self.current_step <= 5:
                self.rewards = -100
                self.best_rewards = -100
                self.epsilon = self.pre_epsilon
            if self.rewards > self.best_rewards:
                self.best_rewards = self.rewards
                self.pre_best = self.best_rewards
                print(f'Best rewards so far: {self.best_rewards}')
                # Save policy
                torch.save(self.policy_dqn.state_dict(), f"car_dql_{self.current_ep}.pt")
            # Check if enough experience has been collected
            if len(self.memory) > self.train_model.mini_batch_size:
                mini_batch = self.memory.sample(self.train_model.mini_batch_size)
                self.train_model.optimize(mini_batch, self.policy_dqn, self.target_dqn)
                print(len(self.memory))
                # Decay epsilon
                self.epsilon = max(self.epsilon - 1 / self.EPISODES, 0)
                self.epsilon_history.append(self.epsilon)
                self.pre_epsilon = self.epsilon

                # Copy policy network to target network after a certain number of steps
                if self.step_count > self.train_model.network_sync_rate:
                    self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
                    self.step_count = 0
                    # Close environment
            else:
                print("OVER LOAD !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            self.current_state, self.done = self.reset()


def main(args=None):
    rclpy.init(args=args)

    a = Env()
    rclpy.spin(a)


if __name__ == '__main__':
    main()
