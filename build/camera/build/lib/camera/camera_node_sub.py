#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from ultralytics import YOLO
from std_msgs.msg import String
import json
import torch
import os


class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.publisher = self.create_publisher(String, 'yolo_detections', 10)

        model_path = './src/camera/best.pt'
        if not os.path.exists(model_path):
            self.get_logger().error(f'Model not found: {model_path}')
            raise FileNotFoundError(model_path)

        # Check CUDA
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda:0' if self.use_cuda else 'cpu'

        self.get_logger().info(f'Loading YOLO model on {self.device}')
        self.model = YOLO(model_path)

        # Move model to selected device
        self.model.to(self.device)

        if self.use_cuda:
            self.get_logger().info(
                f'CUDA available. GPU: {torch.cuda.get_device_name(0)}'
            )
        else:
            self.get_logger().warn('CUDA not available, using CPU.')


    def image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        # Inference on selected device
        results = self.model(image, device=0 if self.use_cuda else 'cpu', verbose=False)

        detections = []
        for result in results[0].boxes:
            bbox = result.xyxy[0].tolist()
            confidence = float(result.conf[0].item())
            class_id = int(result.cls[0].item())

            detections.append({
                "bbox": bbox,
                "confidence": confidence,
                "class_id": class_id
            })

        j_detections = json.dumps(detections)

        annotated_image = results[0].plot()

        out_msg = String()
        out_msg.data = j_detections
        self.publisher.publish(out_msg)

        cv2.imshow("Detection Results", annotated_image)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = YoloNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()