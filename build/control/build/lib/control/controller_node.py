#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
from enum import Enum

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MissionState(Enum):
    FOLLOW_DUAL = 'follow_dual'
    FOLLOW_WHITE = 'follow_white'
    FOLLOW_YELLOW = 'follow_yellow'
    WAIT_OBSTACLE = 'wait_obstacle'   # 已看到施工號誌，持續循白線等障礙
    AVOID = 'avoid'
    FINISH = 'finish'


class ControllerNode(Node):
    def __init__(self):
        super().__init__('controller_node')

        # ========= 狀態 =========
        self.state = MissionState.FOLLOW_DUAL
        self.follow_mode = 'dual'
        self.drive_mode = 'line_follow'

        # YOLO 防抖
        self.current_sign = None
        self.last_sign_time = 0.0
        self.pending_class_id = None
        self.pending_since = None

        self.sign_hold_time = 0.3
        self.mode_cooldown = 0.5
        self.last_mode_change_time = 0.0

        # 雷達資料
        self.front_distance = 9999.0

        # 避障執行保護，避免重複進入
        self.avoid_running = False

        # ========= Publisher =========
        self.follow_mode_pub = self.create_publisher(String, '/follow_mode', 10)
        self.drive_mode_pub = self.create_publisher(String, '/drive_mode', 10)
        self.motor_cmd_pub = self.create_publisher(String, '/motor_cmd', 10)

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

        # 定時發布模式
        self.mode_timer = self.create_timer(0.2, self.publish_modes)

        self.get_logger().info('Controller Node started.')
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

    def publish_motor_cmd(self, left_speed, right_speed):
        msg = String()
        msg.data = json.dumps({
            'left_speed': int(left_speed),
            'right_speed': int(right_speed)
        })
        self.motor_cmd_pub.publish(msg)

    # =====================================================
    # 狀態切換
    # =====================================================
    def set_state(self, new_state, follow_mode=None, drive_mode=None, reason=''):
        now = time.time()

        if (now - self.last_mode_change_time) < self.mode_cooldown:
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
    # 判斷前方是否有障礙
    # =====================================================
    def is_obstacle_ahead(self):
        return self.front_distance < 225

    # =====================================================
    # 避障流程（固定地圖版）
    # =====================================================
    def run_avoid_sequence(self):
        if self.avoid_running:
            return

        self.avoid_running = True

        self.get_logger().info('Start avoid sequence')

        # 切成 controller 接管馬達
        self.drive_mode = 'controller'
        self.publish_modes()

        # 第一段：右轉 -> 直走 -> 左轉
        self.publish_motor_cmd(80, -80)
        time.sleep(2.1)

        self.publish_motor_cmd(140, 140)
        time.sleep(2.4)

        self.publish_motor_cmd(-80, 80)
        time.sleep(2.1)

        # 第二段：一路直走，直到前方再次看到障礙
        while rclpy.ok():
            if self.is_obstacle_ahead():
                self.get_logger().info('>> 前方有障礙，準備進入下一段')
                break
            else:
                self.get_logger().info('>> 前方清空，直行')
                self.publish_motor_cmd(140, 140)

            time.sleep(0.1)

        # 第三段：左轉 -> 直走 -> 右轉 -> 直走
        self.publish_motor_cmd(-80, 80)
        time.sleep(2.1)

        self.publish_motor_cmd(140, 140)
        time.sleep(2.4)

        self.publish_motor_cmd(80, -80)
        time.sleep(2.1)

        self.publish_motor_cmd(140, 140)
        time.sleep(2.0)

        # 避障結束，回到循白線
        self.set_state(
            MissionState.FOLLOW_WHITE,
            follow_mode='white',
            drive_mode='line_follow',
            reason='Avoid sequence finished, back to white line'
        )

        self.avoid_running = False
        self.get_logger().info('Avoid finished')

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

        # 防抖
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
        # 狀態切換邏輯
        # =================================================

        # 一開始雙線，看到左轉或右轉號誌 -> 切白線
        if self.state == MissionState.FOLLOW_DUAL:
            if class_id == 6 :
                self.set_state(
                    MissionState.FOLLOW_WHITE,
                    follow_mode='white',
                    drive_mode='line_follow',
                    reason=f'Detected turn sign class {class_id}'
                )
            if class_id == 2 :
                self.set_state(
                    MissionState.FOLLOW_YELLOW,
                    follow_mode='yellow',
                    drive_mode='line_follow',
                    reason=f'Detected turn sign class {class_id}'
                )

        # 看到施工號誌 -> 進入等待障礙狀態
        elif self.state == MissionState.FOLLOW_WHITE or self.state == MissionState.FOLLOW_YELLOW:
            if class_id == 5:
                self.set_state(
                    MissionState.WAIT_OBSTACLE,
                    follow_mode='white',
                    drive_mode='line_follow',
                    reason='Detected construction sign (class 5), keep following white line'
                )

    def lidar_callback(self, msg):
        data = self.parse_lidar_msg(msg.data)
        if data is None:
            return

        self.front_distance = data.get('front_min', 9999.0)

        # 看到施工號誌後，開始等前方障礙
        if self.state == MissionState.WAIT_OBSTACLE:
            if self.is_obstacle_ahead() and not self.avoid_running:
                self.set_state(
                    MissionState.AVOID,
                    follow_mode='white',
                    drive_mode='controller',
                    reason=f'Obstacle detected, front={self.front_distance}'
                )
                self.run_avoid_sequence()


def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Controller stopped by user.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()