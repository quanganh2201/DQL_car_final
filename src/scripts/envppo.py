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

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2

import trainmodelpytorch

import sys

MIN_DISTANCE = 0.8
ERROR_DISTANCE = 0.6
XML_FILE_PATH = '/home/botcanh/dev_ws/src/two_wheeled_robot/urdf/two_wheeled_robot_copy.urdf'
# XML_FILE_PATH = '/home/botcanh/turtlebot3_ws/src/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_burger/model.sdf'
X_INIT = 0.0
Y_INIT = 0.0
THETA_INIT = 0.0
GOAL_X = 6
GOAL_Y = 6
HORIZONTAL_DIS = 9999
GOAL_THRESHOLD1 = 0.8
GOAL_THRESHOLD2 = 1
GOAL_THRESHOLD3 = 5
GOAL_THRESHOLD4 = 10


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
        self.view_depth_range = 10 * np.ones([5], dtype=float)  # depth in each range

        self.done = False  # done episode or not
        self.EPISODES = 100
        self.steps = 300
        self.current_step = 0
        self.current_ep = 0
        self.ep_done = False
        self.pre_action = 0

        # odometry
        self.x_distance = GOAL_X
        self.y_distance = GOAL_Y
        self.x_pre_distance = GOAL_X
        self.y_pre_distance = GOAL_Y
        self.distance = HORIZONTAL_DIS

        self.rewards = 0
        self.step_count = 0
        self.pre_best = -1000
        self.best_rewards = -1000
        self.current_state = None
        self.new_state = None
        self.threshold_done1 = False

        # TRAIN PARAMETERS
        self.num_states = 7
        self.num_actions = 7

        self.pre_epsilon = 1
        self.epsilon = 1  # 1 = 100% random actions
        # Make the target and policy networks the same (copy weights/biases from one network to the other)

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        self.rewards_per_episode = []

        # List to keep track of epsilon decay
        self.epsilon_history = []

        timer_period = 0.15
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.learn_iters = 0
        self.best_score = 0
        self.score_history = []
        self.agent = trainmodelpytorch.Agent(n_actions=self.num_actions, batch_size= 5, alpha= 0.0003, n_epochs= 4, input_dims= self.num_states)

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

    def getState(self):
        x_distance = self.x_distance
        y_distance = self.y_distance
        distance = self.view_depth_range
        state = [distance[0], distance[1], distance[2], distance[3], distance[4], x_distance, y_distance]
        return state

    def setReward(self, state, x_pre, y_pre):
        done = False
        distance = self.distance
        x_dis = state[-2]
        y_dis = state[-1]
        reward = 0
        rx_dis = 0
        ry_dis = 0
        for temp in range(0,5):
            if state[temp] <= ERROR_DISTANCE:
                reward = -1000
                done = True
        if distance <= GOAL_THRESHOLD1:
            reward += 1000
            done = True
            print("REACHED")
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

        if x_pre > x_dis:
            rx_dis = 2 ** (x_dis / GOAL_X)
        else:
            rx_dis = -5

            # if x_pre <= x_dis:
        #   rx_dis = -(2 ** ((GOAL_X - x_dis) / GOAL_X))

        if y_pre > y_dis:
            ry_dis = 2 ** (y_dis / GOAL_Y)
        else:
            ry_dis = -5

        # if y_pre <= y_dis:
        #    ry_dis = -(2 ** ((GOAL_Y - y_dis) / GOAL_Y))

        reward += rx_dis + ry_dis
        return reward, done

    def step(self, action):
        if action == 0:
            ang_vel = 0.0
        elif action == 1:  # turn right
            ang_vel = 0.35
        elif action == 2:
            ang_vel = -0.35
        elif action == 3:
            ang_vel = 0.7
        elif action == 4:
            ang_vel = -0.7
        elif action == 5:
            ang_vel = 0.17
        elif action == 6:
            ang_vel = -0.17

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.15
        vel_cmd.angular.z = ang_vel
        self.velPub.publish(vel_cmd)

        state = self.getState()
        return np.asarray(state)

    def reset(self):
        self.delete_entity('two_wheeled_robot')

        # SHOULD HAVE A TIMER HERE
        print("deleted")
        i = 0

        while (i < 10000):
            j = 0
            while (j < 10000):
                j = j + 1
            i = i + 1

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
        self.learn_iters = 0
        self.best_score = 0
        self.score_history = []
        return state, done

    def timer_callback(self):  # train
        if self.current_state is None:
            self.current_state = self.getState()
        self.current_step = self.current_step + 1
        if self.current_step < self.steps and self.ep_done == False:
            action, prob, val = self.agent.choose_action(self.current_state)

            # Execute action
            self.new_state = self.step(action)
            reward, self.ep_done = self.setReward(self.current_state,self.x_pre_distance, self.y_pre_distance)
            # Accumulate reward
            self.rewards += reward
            self.pre_action = action
            self.x_pre_distance = self.current_state[-3]
            self.y_pre_distance = self.current_state[-2]
            # Save experience into memory
            self.agent.remember(self.current_state, action, prob, val, reward, self.done)
            if self.current_step % 20 == 0:
                self.agent.learn()
                self.learn_iters +=1
            self.current_state = self.new_state
            self.score_history.append(self.rewards)
            avg_score = np.mean(self.score_history[-100:])

            if avg_score > self.best_score:
                self.best_score = avg_score
                self.agent.save_models()
            print('episode', self.current_ep, 'score %.1f' % self.best_score, 'avg score %.1f' % avg_score,
                  'time_steps', self.current_step, 'learning_steps', self.learn_iters)
        else:
            self.rewards_per_episode.append(self.rewards)
            self.ep_done = True
            self.current_ep += 1
            # Graph training progress
            if (self.current_ep != 0):
                print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
                # self.train_model.plot_progress(self.rewards_per_episode, self.epsilon_history)
            print(self.best_rewards, self.rewards)
            # AVOID ERROR SPAWN
            if self.current_step <= 10:
                self.rewards = self.pre_best
                self.best_rewards = self.pre_best
                self.epsilon = self.pre_epsilon
            if self.rewards > self.best_rewards:
                self.best_rewards = self.rewards
                self.pre_best = self.best_rewards
                print(f'Best rewards so far: {self.best_rewards}')
            self.current_state, self.done = self.reset()


def main(args=None):
    rclpy.init(args=args)

    a = Env()
    rclpy.spin(a)

if __name__ == '__main__':
    main()
