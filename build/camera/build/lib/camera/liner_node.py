#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import cv2
import numpy as np
import rclpy

from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String


class LaneFollowerNode(Node):
    def __init__(self):
        super().__init__('lane_follower_node')

        # ===== 基本設定 =====
        self.bridge = CvBridge()

        # follow_mode: 決定循哪條線
        self.mode = 'none'
        # self.mode = 'yellow'

        # drive_mode: 決定現在是不是這顆 node 有控制權
        self.drive_mode = 'line_follow'

        # ===== PID 參數 =====
        self.pid_settings = {
            'dual': {
                'kp': 2.0, 'ki': 0.00, 'kd': 0.0,
                'base_speed': 100, 'target_x': 340
            },
            'white': {
                'kp': 2.0, 'ki': 0.00, 'kd': 0.0,
                'base_speed': 100, 'target_x': 340
            },
            'yellow': {
                'kp': 2.0, 'ki': 0.0, 'kd': 0.0,
                'base_speed': 100, 'target_x': 340
            },
            'none': {
                'kp': 0.0, 'ki': 0.0, 'kd': 0.0,
                'base_speed': 0, 'target_x': 0
            }
        }

        # ===== PID 狀態 =====
        self.pid_state = {
            'dual': {'prev_error': 0.0, 'integral': 0.0},
            'white': {'prev_error': 0.0, 'integral': 0.0},
            'yellow': {'prev_error': 0.0, 'integral': 0.0},
            'none': {'prev_error': 0.0, 'integral': 0.0}
        }

        # ===== 訂閱 topic =====
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.mode_sub = self.create_subscription(
            String,
            '/follow_mode',
            self.mode_callback,
            10
        )

        self.drive_mode_sub = self.create_subscription(
            String,
            '/drive_mode',
            self.drive_mode_callback,
            10
        )

        # ===== 發布 topic =====
        self.lane_info_pub = self.create_publisher(String, '/lane_info', 10)
        self.motor_cmd_pub = self.create_publisher(String, '/motor_cmd', 10)
        self.debug_image_pub = self.create_publisher(Image, '/lane/debug_image', 10)

        self.get_logger().info('Lane follower node started')
        self.get_logger().info('Default follow_mode: dual')
        self.get_logger().info('Default drive_mode: line_follow')

    # =========================================================
    # follow_mode 切換
    # =========================================================
    def mode_callback(self, msg):
        text = msg.data.strip()
        new_mode = text

        # 如果收到的是 JSON，例如 {"mode":"white"}
        if text.startswith('{'):
            try:
                data = json.loads(text)
                new_mode = data.get('mode', self.mode)
            except Exception:
                self.get_logger().warn('follow_mode JSON 格式錯誤')
                return

        if new_mode not in self.pid_settings:
            self.get_logger().warn('未知 follow_mode，只能用 dual / white / yellow')
            return

        if new_mode != self.mode:
            self.mode = new_mode
            self.reset_pid(self.mode)
            self.get_logger().info(f'Switch follow_mode to: {self.mode}')

    # =========================================================
    # drive_mode 切換
    # =========================================================
    def drive_mode_callback(self, msg):
        new_drive_mode = msg.data.strip()

        if new_drive_mode != self.drive_mode:
            self.drive_mode = new_drive_mode
            self.get_logger().info(f'Switch drive_mode to: {self.drive_mode}')

            # 切出 line_follow 時，順手把 PID 重置
            if self.drive_mode != 'line_follow':
                self.reset_pid(self.mode)

    def reset_pid(self, mode):
        self.pid_state[mode]['prev_error'] = 0.0
        self.pid_state[mode]['integral'] = 0.0

    # =========================================================
    # 收到影像後的主流程
    # =========================================================
    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'影像轉換失敗: {e}')
            return

        # 無論是否有控制權，都可以發 debug image 方便看畫面
        road_center, found, debug_frame = self.detect_lane(frame, self.mode)

        cv2.imshow('line_debug_camera', debug_frame)
        cv2.waitKey(1)

        lane_info = {
            'follow_mode': self.mode,
            'drive_mode': self.drive_mode,
            'found': found,
            'road_center': int(road_center)
        }

        # 如果現在不是 line_follow 模式，這顆 node 不准發 motor_cmd
        if self.drive_mode != 'line_follow':
            lane_info['error'] = None
            lane_info['correction'] = None
            lane_info['note'] = 'drive_mode is not line_follow, skip motor_cmd'

            self.publish_lane_info(lane_info)
            self.publish_debug_image(debug_frame, msg.header.frame_id)
            return

        # # 沒找到線
        # if not found:
        #     lane_info['error'] = None
        #     lane_info['correction'] = None

        #     self.publish_lane_info(lane_info)
        #     self.publish_debug_image(debug_frame, msg.header.frame_id)
        #     if self.mode == 'dual':
        #         self.publish_motor_cmd(50, 50, 'lane_not_found')
        #     elif self.mode == 'white':
        #         self.publish_motor_cmd(30, 80, 'lane_not_found')
        #     elif self.mode == 'yellow':
        #         self.publish_motor_cmd(80, 30, 'lane_not_found')
        #     return

        # 找到線後做 PID
        setting = self.pid_settings[self.mode]
        error = road_center - setting['target_x']
        correction = self.calculate_pid(error, self.mode)

        left_speed = int(np.clip(setting['base_speed'] + correction, -200, 200))
        right_speed = int(np.clip(setting['base_speed'] - correction, -200, 200))

        lane_info['error'] = float(error)
        lane_info['target_x'] = setting['target_x']
        lane_info['correction'] = float(correction)
        lane_info['left_speed'] = left_speed
        lane_info['right_speed'] = right_speed

        self.publish_lane_info(lane_info)
        self.publish_motor_cmd(left_speed, right_speed, 'lane_follow')
        self.publish_debug_image(debug_frame, msg.header.frame_id)

    # =========================================================
    # PID 計算
    # =========================================================
    def calculate_pid(self, error, mode):
        setting = self.pid_settings[mode]
        state = self.pid_state[mode]

        state['integral'] += error
        derivative = error - state['prev_error']
        state['prev_error'] = error

        correction = (
            setting['kp'] * error +
            setting['ki'] * state['integral'] +
            setting['kd'] * derivative
        )
        return correction

    # =========================================================
    # 發布訊息
    # =========================================================
    def publish_lane_info(self, info):
        msg = String()
        msg.data = json.dumps(info)
        self.lane_info_pub.publish(msg)

    def publish_motor_cmd(self, left_speed, right_speed, reason=''):
        cmd = {
            'left_speed': int(left_speed),
            'right_speed': int(right_speed),
            'follow_mode': self.mode,
            'drive_mode': self.drive_mode,
            'source': 'line_follow',
            'reason': reason
        }
        msg = String()
        msg.data = json.dumps(cmd)
        self.motor_cmd_pub.publish(msg)

    def publish_debug_image(self, frame, frame_id=''):
        try:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            msg.header.frame_id = frame_id
            self.debug_image_pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(f'debug image 發布失敗: {e}')

    # =========================================================
    # 根據模式呼叫不同找線函式
    # =========================================================
    def detect_lane(self, frame, mode):
        if frame is None:
            empty = np.zeros((480, 640, 3), dtype=np.uint8)
            return 0, False, empty

        debug_frame = frame.copy()

        if mode == 'dual':
            return self.detect_dual_lane(debug_frame)
        elif mode == 'white':
            return self.detect_white_lane(debug_frame)
        elif mode == 'yellow':
            return self.detect_yellow_lane(debug_frame)
        else:
            return 0, False, debug_frame

    # =========================================================
    # 雙線模式：找黃線和白線
    # =========================================================
    def detect_dual_lane(self, frame):
        h, w, _ = frame.shape

        y_start = int(h * 2 / 3)
        y_end = min(y_start + 150, h)
        x_start = int(w / 2 - 310)
        x_end = int(w / 2 + 310)

        roi = frame[y_start:y_end, x_start:x_end]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([0, 82, 177])
        upper_yellow = np.array([171, 255, 255])

        lower_white = np.array([30, 22, 156])
        upper_white = np.array([179, 69, 255])

        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        yellow_x = self.find_line_center_x(yellow_mask)
        white_x = self.find_line_center_x(white_mask)

        found = (yellow_x is not None) or (white_x is not None)

        if yellow_x is not None and white_x is not None:
            center_x_in_roi = (yellow_x + white_x) // 2
        elif yellow_x is not None:
            center_x_in_roi = yellow_x + 225
        elif white_x is not None:
            center_x_in_roi = white_x - 225
        else:
            center_x_in_roi = 290

        road_center = x_start + center_x_in_roi

        self.draw_debug(frame, x_start, y_start, x_end, y_end, road_center, (0, 0, 255), 'dual')
        return road_center, found, frame

    # =========================================================
    # 白線模式：只找白線
    # =========================================================
    def detect_white_lane(self, frame):
        h, w, _ = frame.shape

        y_start = int(h * 2 / 3)
        y_end = min(y_start + 150, h)
        x_start = int(w / 2 - 120)
        x_end = int(w / 2 + 310)

        roi = frame[y_start:y_end, x_start:x_end]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_white = np.array([30, 22, 156])
        upper_white = np.array([179, 69, 255])

        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        white_x = self.find_line_center_x(white_mask)

        found = white_x is not None

        if white_x is not None:
            center_x_in_roi = white_x - 255
        else:
            center_x_in_roi = 40

        road_center = x_start + center_x_in_roi

        self.draw_debug(frame, x_start, y_start, x_end, y_end, road_center, (255, 0, 255), 'white')
        return road_center, found, frame

    # =========================================================
    # 黃線模式：只找黃線
    # =========================================================
    def detect_yellow_lane(self, frame):
        h, w, _ = frame.shape

        y_start = int(h * 2 / 3)
        y_end = min(y_start + 150, h)
        x_start = int(w / 2 - 320)
        x_end = int(w / 2 + 150)

        roi = frame[y_start:y_end, x_start:x_end]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([0, 82, 177])
        upper_yellow = np.array([171, 255, 255])

        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        yellow_x = self.find_line_center_x(yellow_mask)

        found = yellow_x is not None

        if yellow_x is not None:
            center_x_in_roi = yellow_x + 255
        else:
            center_x_in_roi = 380

        road_center = x_start + center_x_in_roi

        self.draw_debug(frame, x_start, y_start, x_end, y_end, road_center, (0, 255, 255), 'yellow')
        return road_center, found, frame

    # =========================================================
    # 找線中心
    # =========================================================
    def find_line_center_x(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        biggest = max(contours, key=cv2.contourArea)
        M = cv2.moments(biggest)

        if M['m00'] == 0:
            return None

        center_x = int(M['m10'] / M['m00'])
        return center_x

    # =========================================================
    # 畫 debug 畫面
    # =========================================================
    def draw_debug(self, frame, x_start, y_start, x_end, y_end, road_center, color, label):
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        cv2.circle(frame, (int(road_center), y_start + 50), 5, color, -1)
        cv2.putText(
            frame,
            label,
            (x_start, max(30, y_start - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


def main(args=None):
    rclpy.init(args=args)
    node = LaneFollowerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node stopped by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()