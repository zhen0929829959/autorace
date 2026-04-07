#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node

import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO

import json
import torch

MIN_AREA_THRESHOLD = 3500


class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')

        self.bridge = CvBridge()

        # 訂閱相機
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # 發布 detection
        self.publisher = self.create_publisher(
            String,
            '/yolo_detection',
            10
        )

        # ===== 檢查 CUDA =====
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda:0' if self.use_cuda else 'cpu'

        self.get_logger().info(f'Loading YOLO model on {self.device}')
        self.model = YOLO('src/camera/best.pt')

        # ===== 模型搬到 GPU / CPU =====
        self.model.to(self.device)

        if self.use_cuda:
            self.get_logger().info(
                f'CUDA available. GPU: {torch.cuda.get_device_name(0)}'
            )
        else:
            self.get_logger().warn('CUDA not available, using CPU.')

        self.get_logger().info('YOLO node started')

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        # ===== YOLO 推論 =====
        results = self.model(
            frame,
            device=0 if self.use_cuda else 'cpu',
            verbose=False
        )

        max_area = 0
        best_class_id = None
        best_confidence = 0.0
        best_box = None

        # ===== 找最大框 =====
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

                area = (x2 - x1) * (y2 - y1)

                if area > max_area:
                    max_area = area
                    best_class_id = class_id
                    best_confidence = confidence
                    best_box = (x1, y1, x2, y2)

        # ===== 判斷是否有效 =====
        if best_box is not None and max_area > MIN_AREA_THRESHOLD:
            data = {
                "class_id": best_class_id,
                "confidence": best_confidence,
                "area": max_area
            }

            msg_out = String()
            msg_out.data = json.dumps(data)
            self.publisher.publish(msg_out)

            self.get_logger().info(
                f'Detect class={best_class_id}, area={max_area}, conf={best_confidence:.2f}'
            )

            # debug 畫框
            x1, y1, x2, y2 = best_box
            label = f"ID:{best_class_id} ({best_confidence:.2f})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2
            )
        else:
            # 沒偵測到 → 發 None
            msg_out = String()
            msg_out.data = json.dumps({"class_id": None})
            self.publisher.publish(msg_out)

        cv2.imshow("YOLO", frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('YOLO stopped')
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()