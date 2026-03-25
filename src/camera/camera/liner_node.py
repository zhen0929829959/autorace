#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


@dataclass
class PIDConfig:
    kp: float
    ki: float
    kd: float
    base_speed: int
    target_x: int


class PIDState:
    def __init__(self) -> None:
        self.prev_error = 0.0
        self.integral = 0.0

    def reset(self) -> None:
        self.prev_error = 0.0
        self.integral = 0.0

    def update(self, error: float, cfg: PIDConfig) -> float:
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        return cfg.kp * error + cfg.ki * self.integral + cfg.kd * derivative


class LaneDetectorFollowerNode(Node):
    def __init__(self) -> None:
        super().__init__('lane_detector_follower_node')

        self.bridge = CvBridge()
        self.current_mode = 'dual'
        self.last_found = False

        # 直接沿用你原本三份程式的 PID 參數與目標中心
        self.pid_configs: Dict[str, PIDConfig] = {
            'dual': PIDConfig(kp=0.7, ki=0.01, kd=0.25, base_speed=50, target_x=320),
            'white_only': PIDConfig(kp=1.7, ki=0.0, kd=0.0, base_speed=50, target_x=520),
            'yellow_only': PIDConfig(kp=3.5, ki=0.0, kd=0.0, base_speed=50, target_x=130),
        }
        self.pid_states: Dict[str, PIDState] = {
            mode: PIDState() for mode in self.pid_configs
        }

        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10,
        )
        self.mode_sub = self.create_subscription(
            String,
            '/follow_mode',
            self.mode_callback,
            10,
        )

        self.lane_info_pub = self.create_publisher(String, '/lane_info', 10)
        self.motor_cmd_pub = self.create_publisher(String, '/motor_cmd', 10)
        self.debug_image_pub = self.create_publisher(Image, '/lane/debug_image', 10)

        self.get_logger().info('lane_detector_follower_node started')
        self.get_logger().info('default mode: dual')

    def mode_callback(self, msg: String) -> None:
        raw = msg.data.strip()
        new_mode = raw

        # 允許直接傳字串，也允許傳 JSON: {"mode": "white_only"}
        if raw.startswith('{'):
            try:
                data = json.loads(raw)
                new_mode = data.get('mode', self.current_mode)
            except json.JSONDecodeError:
                self.get_logger().warn(f'Invalid /follow_mode JSON: {raw}')
                return

        if new_mode not in self.pid_configs:
            self.get_logger().warn(
                f'Unknown mode: {new_mode}. Use dual / white_only / yellow_only'
            )
            return

        if new_mode != self.current_mode:
            self.current_mode = new_mode
            self.pid_states[new_mode].reset()
            self.get_logger().info(f'Switched follow mode to: {self.current_mode}')

    def image_callback(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'cv_bridge failed: {exc}')
            return

        road_center, found, debug_frame = self.detect_lane(frame, self.current_mode)

#///////////////////////////////
        cv2.imshow('line_debug_camera', debug_frame)
        cv2.waitKey(1)

        lane_info = {
            'mode': self.current_mode,
            'found': found,
            'road_center': int(road_center),
        }

        if not found:
            lane_info['error'] = None
            lane_info['correction'] = None
            self.publish_lane_info(lane_info)
            self.publish_motor_cmd(0, 0, reason='lane_not_found')
            self.publish_debug_image(debug_frame, msg.header.frame_id)
            return

        cfg = self.pid_configs[self.current_mode]
        error = float(road_center - cfg.target_x)
        correction = self.pid_states[self.current_mode].update(error, cfg)

        left_speed = int(np.clip(cfg.base_speed + correction, -100, 100)) 
        right_speed = int(np.clip(cfg.base_speed - correction, -100, 100))

        lane_info.update({
            'error': error,
            'target_x': cfg.target_x,
            'correction': correction,
            'left_speed': left_speed,
            'right_speed': right_speed,
        })

        self.publish_lane_info(lane_info)
        self.publish_motor_cmd(left_speed, right_speed, reason='lane_follow')
        self.publish_debug_image(debug_frame, msg.header.frame_id)

    def publish_lane_info(self, info: Dict) -> None:
        msg = String()
        msg.data = json.dumps(info)
        self.lane_info_pub.publish(msg)

    def publish_motor_cmd(self, left_speed: int, right_speed: int, reason: str = '') -> None:
        cmd = {
            'left_speed': int(left_speed),
            'right_speed': int(right_speed),
            'mode': self.current_mode,
            'reason': reason,
        }
        msg = String()
        msg.data = json.dumps(cmd)
        self.motor_cmd_pub.publish(msg)

    def publish_debug_image(self, frame: np.ndarray, frame_id: str = '') -> None:
        try:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            msg.header.frame_id = frame_id
            self.debug_image_pub.publish(msg)
        except Exception as exc:
            self.get_logger().warn(f'Failed to publish debug image: {exc}')

    def detect_lane(
        self,
        frame: np.ndarray,
        mode: str,
    ) -> Tuple[int, bool, np.ndarray]:
        if frame is None:
            return 0, False, np.zeros((480, 640, 3), dtype=np.uint8)

        debug_frame = frame.copy()

        if mode == 'dual':
            return self.detect_dual_lane(debug_frame)
        if mode == 'white_only':
            return self.detect_white_lane(debug_frame)
        if mode == 'yellow_only':
            return self.detect_yellow_lane(debug_frame)

        return 0, False, debug_frame

    def detect_dual_lane(self, frame: np.ndarray) -> Tuple[int, bool, np.ndarray]:
        height, width, _ = frame.shape
        y_start = int(height * 1.5/ 3)
        y_end = min(y_start + 150, height)
        x_start = int(width / 2 - 310)
        x_end = int(width / 2 + 310)

        roi = frame[y_start:y_end, x_start:x_end]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([0, 44, 100])
        upper_yellow = np.array([40, 255, 255])
        lower_white = np.array([0, 0, 225])
        upper_white = np.array([180, 8, 255])

        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        yellow_x = self.find_centroid_x(mask_yellow)
        white_x = self.find_centroid_x(mask_white)

        found = yellow_x is not None or white_x is not None

        if yellow_x is not None and white_x is not None:
            center_x = (yellow_x + white_x) // 2
        elif yellow_x is not None:
            center_x = yellow_x + 50
        elif white_x is not None:
            center_x = white_x - 50
        else:
            center_x = 100

        road_center = x_start + center_x
        self.draw_debug(frame, x_start, y_start, x_end, y_end, road_center, (0, 0, 255), 'dual')
        return road_center, found, frame

    def detect_white_lane(self, frame: np.ndarray) -> Tuple[int, bool, np.ndarray]:
        height, width, _ = frame.shape
        y_start = int(height * 2 / 3)
        y_end = min(y_start + 150, height)
        x_start = int(width / 2 - 120)
        x_end = int(width / 2 + 300)

        roi = frame[y_start:y_end, x_start:x_end]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0, 0, 225])
        upper_white = np.array([180, 8, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        white_x = self.find_centroid_x(mask_white)
        found = white_x is not None

        if white_x is not None:
            center_x = white_x - 200
        else:
            center_x = 250

        road_center = x_start + center_x
        self.draw_debug(frame, x_start, y_start, x_end, y_end, road_center, (255, 0, 255), 'white_only')
        return road_center, found, frame

    def detect_yellow_lane(self, frame: np.ndarray) -> Tuple[int, bool, np.ndarray]:
        height, width, _ = frame.shape
        y_start = int(height * 2 / 3)
        y_end = min(y_start + 150, height)
        x_start = int(width / 2 - 320)
        x_end = int(width / 2 + 150)

        roi = frame[y_start:y_end, x_start:x_end]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([0, 74, 0])
        upper_yellow = np.array([27, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        yellow_x = self.find_centroid_x(mask_yellow)
        found = yellow_x is not None

        if yellow_x is not None:
            center_x = yellow_x + 200
        else:
            center_x = 250

        road_center = x_start + center_x
        self.draw_debug(frame, x_start, y_start, x_end, y_end, road_center, (0, 255, 255), 'yellow_only')
        return road_center, found, frame

    @staticmethod
    def find_centroid_x(mask: np.ndarray) -> Optional[int]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        moments = cv2.moments(largest)
        if moments['m00'] == 0:
            return None

        return int(moments['m10'] / moments['m00'])

    @staticmethod
    def draw_debug(
        frame: np.ndarray,
        x_start: int,
        y_start: int,
        x_end: int,
        y_end: int,
        road_center: int,
        point_color: Tuple[int, int, int],
        label: str,
    ) -> None:
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        cv2.circle(frame, (int(road_center), y_start + 50), 5, point_color, -1)
        cv2.putText(
            frame,
            label,
            (x_start, max(30, y_start - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LaneDetectorFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('lane_detector_follower_node stopped by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()