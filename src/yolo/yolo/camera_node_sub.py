#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from ultralytics import YOLO
from std_msgs.msg import String
import json


class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            'image_raw',
            self.image_callback,
            10
        )

        self.publisher = self.create_publisher(String, 'yolo_detections', 10)


    def image_callback(self, msg):
        model = YOLO('src/yolo/best.pt')

        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        results = model(image)

        # Extract bounding boxes and other details
        detections = []
        for result in results[0].boxes:
            bbox = result.xyxy[0].tolist()  # Bounding box coordinates [x_min, y_min, x_max, y_max]
            confidence = result.conf[0].item()  # Confidence score
            class_id = result.cls[0].item()  # Class ID
            detections.append({
                "bbox": bbox,
                "confidence": confidence,
                "class_id": class_id
            })

        # Send detections to another node or save to a file
        # with open("./detections.json", "w") as f:
        #     json.dump(detections, f)
        # print("Detections saved to detections.json")
        j_detections=json.dumps(detections)

        # Render predictions on the image
        annotated_image = results[0].plot()

        msg = String()
        msg.data = j_detections
        self.publisher.publish(msg)
        # self.get_logger().info('Publishing: "%s"' % msg.data)

        cv2.imshow("Detection Results", annotated_image)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = YoloNode()
    rclpy.spin(node)


    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

