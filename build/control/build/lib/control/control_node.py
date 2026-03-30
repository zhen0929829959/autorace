#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MainControllerNode(Node):
    def __init__(self):
        super().__init__('main_controller_node')

        # ===== 狀態 =====
        self.stage = 2                  # 目前先做第二關
        self.follow_mode = 'dual'       # 預設循雙線
        self.current_sign = None
        self.last_sign_time = 0.0
        self.last_mode_change_time = 0.0

        # 防抖動參數
        self.sign_hold_time = 0.3       # 同一號誌至少穩定存在 0.3 秒
        self.mode_cooldown = 1.0        # 模式切換冷卻時間，避免瘋狂切換

        # 記錄同一個 class_id 是否持續出現
        self.pending_class_id = None
        self.pending_since = None

        # ===== Publisher =====
        self.follow_mode_pub = self.create_publisher(String, '/follow_mode', 10)

        # ===== Subscriber =====
        self.yolo_sub = self.create_subscription(
            String,
            '/yolo_detection',
            self.yolo_callback,
            10
        )

        # 之後雷達可以加這個
        # self.lidar_sub = self.create_subscription(
        #     String,
        #     '/lidar_info',
        #     self.lidar_callback,
        #     10
        # )

        # 定時發 mode，避免新啟動的循線 node 沒收到
        self.mode_timer = self.create_timer(0.2, self.publish_follow_mode)

        self.get_logger().info('Main Controller Node started.')
        self.get_logger().info('Default follow mode = dual')

    def publish_follow_mode(self):
        msg = String()
        msg.data = self.follow_mode
        self.follow_mode_pub.publish(msg)

    def set_follow_mode(self, new_mode, reason=''):
        now = time.time()

        if new_mode == self.follow_mode:
            return

        if now - self.last_mode_change_time < self.mode_cooldown:
            return

        old_mode = self.follow_mode
        self.follow_mode = new_mode
        self.last_mode_change_time = now

        self.get_logger().info(
            f'[MODE CHANGE] {old_mode} -> {new_mode} | reason: {reason}'
        )

        # 立刻補發一次
        msg = String()
        msg.data = self.follow_mode
        self.follow_mode_pub.publish(msg)

    def parse_yolo_msg(self, raw_text):
        """
        支援兩種格式：
        1. dict: {"class_id": 6, ...}
        2. list: [{"class_id": 6, ...}, ...]
        """
        try:
            data = json.loads(raw_text)
        except Exception as e:
            self.get_logger().warn(f'YOLO JSON parse failed: {e}')
            return None

        if isinstance(data, dict):
            return data

        if isinstance(data, list) and len(data) > 0:
            # 取第一個偵測結果
            return data[0]

        return None

    def yolo_callback(self, msg):
        det = self.parse_yolo_msg(msg.data)
        if det is None:
            return

        class_id = det.get('class_id', None)
        confidence = det.get('confidence', None)

        if class_id is None:
            return

        now = time.time()

        # ===== 防抖動 =====
        if class_id != self.pending_class_id:
            self.pending_class_id = class_id
            self.pending_since = now
            return

        if self.pending_since is None:
            self.pending_since = now
            return

        # 還沒穩定出現夠久，先不切
        if (now - self.pending_since) < self.sign_hold_time:
            return

        # 避免同一號誌重複一直觸發
        if self.current_sign == class_id and (now - self.last_sign_time) < 1.0:
            return

        self.current_sign = class_id
        self.last_sign_time = now

        self.get_logger().info(
            f'[YOLO] class_id={class_id}, confidence={confidence}'
        )

        # ===== 第二關邏輯 =====
        # 先預設循雙線
        # class_id=6 -> 改循白線
        # class_id=2 -> 你目前也指定改循白線
        if self.stage == 2:
            if class_id == 6:
                self.set_follow_mode('white', 'Detected right turn sign (class 6)')

            elif class_id == 2:
                self.set_follow_mode('white', 'Detected left turn sign (class 2)')

            # 之後如果你想改成左轉走黃線，只要改這行：
            # elif class_id == 2:
            #     self.set_follow_mode('yellow', 'Detected left turn sign (class 2)')

    # 之後可以加
    # def lidar_callback(self, msg):
    #     pass


def main(args=None):
    rclpy.init(args=args)
    node = MainControllerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Main Controller stopped by user.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()