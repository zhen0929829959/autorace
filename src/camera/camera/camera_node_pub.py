#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class UsbCameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        # Publisher
        self.publisher = self.create_publisher(Image, '/camera/image_raw', 10)
        self.bridge = CvBridge()

        # Open USB camera
        # self.cap = cv2.VideoCapture("/dev/video0")
        self.cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)

        if not self.cap.isOpened():
            self.get_logger().error('Cannot open camera')
            return

        #  Force MJPG format
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        # Resolution (must be supported under MJPG)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        # 關閉自動曝光（V4L2）
        # self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

        # # 設定固定曝光（數值需調整）
        # self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)
        # 約 20 FPS
        self.timer = self.create_timer(0.016, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Failed to read frame')
            return

        # Resize to 640x480
        frame = cv2.resize(frame, (640, 480))

        # OpenCV → ROS2 Image
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher.publish(msg)

        # 本地顯示（教學用）
        # cv2.imshow('usb_camera', frame)
        # cv2.waitKey(1)


def main():
    rclpy.init()
    node = UsbCameraNode()
    rclpy.spin(node)

    node.cap.release()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
