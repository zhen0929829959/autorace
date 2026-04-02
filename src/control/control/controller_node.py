#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
from enum import Enum

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MissionState(Enum):
    TRAFFICLIGHT = 'trafficlight'
    FOLLOW_DUAL = 'follow_dual'
    FOLLOW_WHITE = 'follow_white'
    FOLLOW_YELLOW = 'follow_yellow'
    AVOID = 'avoid'
    DETECT_P = 'detect_p'
    PARKING = 'parking'
    TURN_LINE2 = 'turn_line2'
    RED_BARRIER = 'red_barrier'
    MAZE = 'maze'
    FINISH = 'finish'


class ControllerNode(Node):
    def __init__(self):
        super().__init__('controller_node')

        # ========= 狀態 =========
        self.state = MissionState.FOLLOW_DUAL
        self.follow_mode = 'dual'
        self.drive_mode = 'line_follow'

        self.current_sign = None
        self.last_sign_time = 0.0
        self.last_mode_change_time = 0.0

        self.sign_hold_time = 0.3
        self.mode_cooldown = 1.0

        self.pending_class_id = None
        self.pending_since = None

        self.front_distance = 9999.0

        # ========= Publisher =========
        self.follow_mode_pub = self.create_publisher(String, '/follow_mode', 10)
        self.drive_mode_pub = self.create_publisher(String, '/drive_mode', 10)

        # ========= Subscriber =========
        self.yolo_sub = self.create_subscription(
            String,
            '/yolo_detection',
            self.yolo_callback,
            10
        )

        self.lidar_sub = self.create_subscription(
            String,
            '/lidar_info',
            self.lidar_callback,
            10
        )

        self.avoid_done_sub = self.create_subscription(
            String,
            '/avoid_done',
            self.avoid_done_callback,
            10
        )

        # 定時發布目前模式
        self.mode_timer = self.create_timer(0.2, self.publish_modes)

        self.get_logger().info('Main Controller Node started.')
        self.log_state()

    # =====================================================
    # publish
    # =====================================================
    def publish_modes(self):
        msg1 = String()
        msg1.data = self.follow_mode
        self.follow_mode_pub.publish(msg1)

        msg2 = String()
        msg2.data = self.drive_mode
        self.drive_mode_pub.publish(msg2)

    # =====================================================
    # 狀態切換
    # =====================================================
    def set_state(self, new_state, follow_mode=None, drive_mode=None, reason=''):
        now = time.time()

        if now - self.last_mode_change_time < self.mode_cooldown:
            return

        old_state = self.state
        old_follow = self.follow_mode
        old_drive = self.drive_mode

        self.state = new_state

        if follow_mode is not None:
            self.follow_mode = follow_mode

        if drive_mode is not None:
            self.drive_mode = drive_mode

        self.last_mode_change_time = now

        self.get_logger().info(
            f'[STATE CHANGE] {old_state.value} -> {new_state.value} | '
            f'follow: {old_follow}->{self.follow_mode} | '
            f'drive: {old_drive}->{self.drive_mode} | '
            f'reason: {reason}'
        )

        self.publish_modes()

    def log_state(self):
        self.get_logger().info(
            f'Current state={self.state.value}, '
            f'follow_mode={self.follow_mode}, '
            f'drive_mode={self.drive_mode}'
        )

    # =====================================================
    # parsing
    # =====================================================
    def parse_yolo_msg(self, raw_text):
        try:
            data = json.loads(raw_text)
        except Exception as e:
            self.get_logger().warn(f'YOLO JSON parse failed: {e}')
            return None

        if isinstance(data, dict):
            return data

        if isinstance(data, list) and len(data) > 0:
            return data[0]

        return None

    def parse_lidar_msg(self, raw_text):
        try:
            data = json.loads(raw_text)
            return data
        except Exception as e:
            self.get_logger().warn(f'Lidar JSON parse failed: {e}')
            return None

    # =====================================================
    # callback
    # =====================================================
    def yolo_callback(self, msg):
        det = self.parse_yolo_msg(msg.data)
        if det is None:
            return

        class_id = det.get('class_id', None)
        confidence = det.get('confidence', None)

        if class_id is None:
            return

        now = time.time()

        # 防抖動
        if class_id != self.pending_class_id:
            self.pending_class_id = class_id
            self.pending_since = now
            return

        if self.pending_since is None:
            self.pending_since = now
            return

        if (now - self.pending_since) < self.sign_hold_time:
            return

        if self.current_sign == class_id and (now - self.last_sign_time) < 1.0:
            return

        self.current_sign = class_id
        self.last_sign_time = now

        self.get_logger().info(
            f'[YOLO] class_id={class_id}, confidence={confidence}, state={self.state.value}'
        )

        # =================================================
        # 根據目前 state 決定怎麼轉
        # =================================================

        # 一開始雙線，看到左右轉號誌
        if self.state == MissionState.FOLLOW_DUAL:
            if class_id == 6:
                self.set_state(
                    MissionState.FOLLOW_WHITE,
                    follow_mode='white',
                    drive_mode='line_follow',
                    reason='Detected right turn sign (class 6)'
                )

            elif class_id == 2:
                self.set_state(
                    MissionState.FOLLOW_WHITE,   # 你現在先都走白線
                    follow_mode='white',
                    drive_mode='line_follow',
                    reason='Detected left turn sign (class 2)'
                )

        # 白線跟隨時看到 P
        elif self.state == MissionState.FOLLOW_WHITE:
            if class_id == 3:
                self.set_state(
                    MissionState.PARKING,
                    drive_mode='parking',
                    reason='Detected parking sign (class 3)'
                )

        # 停車後切回雙線或後續流程
        elif self.state == MissionState.TURN_LINE2:
            if class_id == 7:
                self.set_state(
                    MissionState.RED_BARRIER,
                    drive_mode='line_follow',
                    follow_mode='dual',
                    reason='Detected stop sign (class 7)'
                )

        elif self.state == MissionState.RED_BARRIER:
            if class_id == 4:
                self.set_state(
                    MissionState.MAZE,
                    drive_mode='maze',
                    reason='Detected tunnel sign (class 4)'
                )

    def lidar_callback(self, msg):
        data = self.parse_lidar_msg(msg.data)
        if data is None:
            return

        self.front_distance = data.get('front_min', 9999.0)

        # 只有在循線狀態才會因障礙切到避障
        if self.state in [MissionState.FOLLOW_WHITE, MissionState.FOLLOW_YELLOW, MissionState.FOLLOW_DUAL]:
            if self.front_distance < 225:
                self.set_state(
                    MissionState.AVOID,
                    drive_mode='avoid',
                    reason=f'Obstacle detected, front={self.front_distance}'
                )

    def avoid_done_callback(self, msg):
        result = msg.data.strip()

        if self.state != MissionState.AVOID:
            return

        if result == 'done':
            self.set_state(
                MissionState.FOLLOW_WHITE,
                drive_mode='line_follow',
                follow_mode='white',
                reason='Avoid finished'
            )


def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Main Controller stopped by user.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()