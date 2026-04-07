#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import numpy as np


class UsbCameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        # Publisher
        self.publisher = self.create_publisher(Image, '/camera/image_raw', 10)
        self.bridge = CvBridge()

        # Open USB camera (V4L2)
        self.cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)

        if not self.cap.isOpened():
            self.get_logger().error('Cannot open camera')
            return

        # ========= 基本設定 =========
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # MJPG（降低過曝 + 提升穩定）
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        # ========= 抗反光核心設定 =========
        # 關閉自動曝光
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

        # 固定曝光（可微調 -5 ~ -8）
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)

        # 關閉自動白平衡
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)

        # 壓亮度、提高對比
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 90)
        self.cap.set(cv2.CAP_PROP_CONTRAST, 40)

        # 降低 buffer（減少延遲/曝光漂移）
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # ========= FPS（穩定）=========
        fps = 30
        self.timer = self.create_timer(1.0 / fps, self.timer_callback)

        # warm-up（讓曝光穩定）
        for _ in range(10):
            self.cap.read()
            
    def remove_glare(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)

        mask_3 = cv2.merge([mask, mask, mask])
        frame = np.where(mask_3 == 255, frame * 0.7, frame)

        return frame.astype(np.uint8)

    def timer_callback(self):
        ret, frame = self.cap.read()

        if not ret:
            self.get_logger().warn('Failed to read frame')
            return

        # 軟體去反光（補強）
        # frame = self.remove_glare(frame)

        # OpenCV → ROS2 Image
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher.publish(msg)

        # 顯示
        cv2.imshow('usb_camera', frame)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = UsbCameraNode()
    rclpy.spin(node)

    node.cap.release()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()