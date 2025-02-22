#!/usr/bin/env python3
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from geometry_msgs.msg import Twist
import numpy as np

class ImageSubscriber(Node):

    def __init__(self):
        super().__init__('image_subscriber')
        self.depth_subcription = self.create_subscription(Image,'/camera_link/depth/image_raw',self.process_data_depth,10)
        self.subscription = self.create_subscription(
            Image,
            '/camera_link/image_raw',
            self.image_callback,
            10)
        self.model = YOLO('~/yolobot/src/yolobot_recognition/scripts/yolov8n.pt')
        self.timer = self.create_timer(1, self.timer_callback)

        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()
        self.out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))
        self.img_pub = self.create_publisher(Image, "/inference_result", 1)
        #self.velPub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.turnLeft = False
        self.turnRight = False
        self.view_depth = None
        self.depth_img = None

    def depthToCV8UC1(self, float_img):
        # Process images
        mono8_img = np.zeros_like(float_img, dtype=np.float16)
        cv2.convertScaleAbs(float_img, mono8_img, alpha=1.5, beta=10)
        return mono8_img
    
    def process_data_depth(self, data):
        self.view_depth = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        self.view_depth = np.array(self.view_depth, dtype=np.float32)
        #alpha: contrast 0 -127, beta: brightness 0 -100
        self.depth_img = cv2.convertScaleAbs(self.view_depth, alpha=10 , beta=30)
        cv2.imshow('view', self.depth_img)
        cv2.imshow('view0', self.view_depth)

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

    def timer_callback(self):
        '''
        velMsg =Twist()
        velMsg.linear.x = 0.3
        velMsg.linear.y = 0.0
        velMsg.linear.z = 0.0
        velMsg.angular.x = 0.0
        velMsg.angular.y = 0.0
        if self.turnLeft == True:
            velMsg.angular.z = -0.3
        if self.turnRight == True:
            velMsg.angular.z = 0.3
        '''
        #self.velPub.publish(velMsg)
        #cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)

    image_subscriber = ImageSubscriber()

    rclpy.spin(image_subscriber)

    image_subscriber.out.release()

    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
