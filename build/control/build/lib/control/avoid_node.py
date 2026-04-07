#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from rplidar import RPLidar

np.float = float  # 相容舊套件


class AvoidNode(Node):
    def __init__(self):
        super().__init__('avoid_node')

        # =========================
        # LIDAR 參數
        # =========================
        self.PORT_NAME = '/dev/ttyUSB0'
        self.DIST_THRESHOLD = 225
        self.FRONT_ANGLE_RANGE = (350, 30)
        self.LEFT_ANGLE_RANGE = (270, 340)
        self.RIGHT_ANGLE_RANGE = (20, 90)

        # =========================
        # 狀態
        # =========================
        self.running = True
        self.scan_data = []
        self.drive_mode = 'line_follow'
        self.avoid_running = False
        self.lidar = None

        # =========================
        # publisher
        # =========================
        self.motor_pub = self.create_publisher(String, '/motor_cmd', 10)
        self.lidar_pub = self.create_publisher(String, '/lidar_info', 10)
        self.avoid_done_pub = self.create_publisher(String, '/avoid_done', 10)

        # =========================
        # subscriber
        # =========================
        self.drive_mode_sub = self.create_subscription(
            String,
            '/drive_mode',
            self.drive_mode_callback,
            10
        )

        # =========================
        # 初始化 lidar
        # =========================
        try:
            self.lidar = RPLidar(self.PORT_NAME)
            self.lidar.start_motor()
            self.get_logger().info('RPLidar 啟動成功')
        except Exception as e:
            self.get_logger().error(f'RPLidar 初始化失敗: {e}')
            raise

        # =========================
        # 啟動 lidar thread
        # =========================
        self.lidar_thread = threading.Thread(target=self.lidar_loop, daemon=True)
        self.lidar_thread.start()

        # 定時發布 lidar_info
        self.lidar_info_timer = self.create_timer(0.1, self.publish_lidar_info)

        self.get_logger().info('AvoidNode started.')

    # =========================================================
    # callback
    # =========================================================
    def drive_mode_callback(self, msg):
        new_mode = msg.data.strip()

        if new_mode != self.drive_mode:
            self.get_logger().info(f'drive_mode: {self.drive_mode} -> {new_mode}')
            self.drive_mode = new_mode

        # 只有切進 avoid 時才啟動一次
        if self.drive_mode == 'avoid' and not self.avoid_running:
            self.avoid_running = True
            threading.Thread(target=self.control_loop, daemon=True).start()

    # =========================================================
    # motor cmd publish
    # =========================================================
    def publish_motor_cmd(self, left, right, reason=''):
        if self.drive_mode != 'avoid':
            self.get_logger().warn(
                f'drive_mode={self.drive_mode}，avoid_node 不發 motor_cmd'
            )
            return

        msg = String()
        msg.data = json.dumps({
            'left_speed': int(left),
            'right_speed': int(right),
            'source': 'avoid',
            'reason': reason
        })
        self.motor_pub.publish(msg)

        self.get_logger().info(
            f'/motor_cmd -> left={left}, right={right}, reason={reason}'
        )

    def stop_motor(self):
        msg = String()
        msg.data = json.dumps({
            'left_speed': 0,
            'right_speed': 0,
            'source': 'avoid',
            'reason': 'stop'
        })
        self.motor_pub.publish(msg)

    def publish_avoid_done(self):
        msg = String()
        msg.data = 'done'
        self.avoid_done_pub.publish(msg)
        self.get_logger().info('publish /avoid_done -> done')

    # =========================================================
    # lidar info publish
    # =========================================================
    def publish_lidar_info(self):
        front = self.get_front_min_distance()
        left = self.get_min_distance_left()
        right = self.get_min_distance_right()

        msg = String()
        msg.data = json.dumps({
            'front_min': float(front),
            'left_min': float(left),
            'right_min': float(right)
        })
        self.lidar_pub.publish(msg)

    # =========================================================
    # 角度 / 距離判斷
    # =========================================================
    def in_front(self, angle):
        if self.FRONT_ANGLE_RANGE[0] > self.FRONT_ANGLE_RANGE[1]:
            return angle >= self.FRONT_ANGLE_RANGE[0] or angle <= self.FRONT_ANGLE_RANGE[1]
        return self.FRONT_ANGLE_RANGE[0] <= angle <= self.FRONT_ANGLE_RANGE[1]

    def get_front_min_distance(self):
        front_dists = [dist for angle, dist in self.scan_data if self.in_front(angle)]
        return min(front_dists) if front_dists else 9999

    def get_min_distance_left(self):
        left_dists = [
            dist for angle, dist in self.scan_data
            if self.LEFT_ANGLE_RANGE[0] <= angle <= self.LEFT_ANGLE_RANGE[1]
        ]
        return min(left_dists) if left_dists else 9999

    def get_min_distance_right(self):
        right_dists = [
            dist for angle, dist in self.scan_data
            if self.RIGHT_ANGLE_RANGE[0] <= angle <= self.RIGHT_ANGLE_RANGE[1]
        ]
        return min(right_dists) if right_dists else 9999

    # =========================================================
    # lidar loop
    # =========================================================
    def lidar_loop(self):
        while self.running:
            try:
                scan_iter = self.lidar.iter_scans()

                for scan in scan_iter:
                    if not self.running:
                        break

                    temp_data = []
                    for measurement in scan:
                        if not isinstance(measurement, (list, tuple)) or len(measurement) != 3:
                            continue

                        quality, angle, dist = measurement

                        if 0 <= int(angle) < 360:
                            temp_data.append((angle, dist))

                    self.scan_data = temp_data
                    time.sleep(0.05)

            except Exception as e:
                self.get_logger().error(f'LIDAR 錯誤: {e}')
                time.sleep(1.0)

    # =========================================================
    # 避障流程
    # 這份是照你原本 ra2 control_loop 改的
    # =========================================================
    def control_loop(self):
        try:
            self.get_logger().info('開始執行避障流程')

            # 第一段：先右繞
            self.publish_motor_cmd(-80, 80, 'turn_right_1')
            time.sleep(2.1)

            self.publish_motor_cmd(-140, -140, 'forward_1')
            time.sleep(2.4)

            self.publish_motor_cmd(80, -80, 'turn_left_1')
            time.sleep(2.1)

            # 第二段：前進直到前方再次偵測到障礙
            while rclpy.ok() and self.running and self.drive_mode == 'avoid':
                front_dist = self.get_front_min_distance()

                if front_dist < self.DIST_THRESHOLD:
                    self.get_logger().info(
                        f'前方有障礙，準備第二段左繞，front={front_dist:.1f}'
                    )
                    break
                else:
                    self.get_logger().info(f'前方清空，直行 | front={front_dist:.1f}')
                    self.publish_motor_cmd(-140, -140, 'forward_search')

                time.sleep(0.1)

            # 第三段：左繞回來
            self.publish_motor_cmd(80, -80, 'turn_left_2')
            time.sleep(2.1)

            self.publish_motor_cmd(-140, -140, 'forward_2')
            time.sleep(2.4)

            self.publish_motor_cmd(-80, 80, 'turn_right_2')
            time.sleep(2.1)

            self.publish_motor_cmd(-140, -140, 'forward_3')
            time.sleep(2.0)

            self.stop_motor()
            time.sleep(0.2)

            self.publish_avoid_done()
            self.get_logger().info('避障完成')

        except Exception as e:
            self.get_logger().error(f'control_loop error: {e}')
            self.stop_motor()

        finally:
            self.avoid_running = False

    # =========================================================
    # 清理
    # =========================================================
    def cleanup(self):
        self.running = False

        try:
            self.stop_motor()
        except Exception:
            pass

        try:
            if self.lidar is not None:
                self.lidar.stop()
                self.lidar.stop_motor()
                self.lidar.disconnect()
                self.get_logger().info('RPLidar 已釋放')
        except Exception as e:
            self.get_logger().warn(f'RPLidar 清理失敗: {e}')

    def destroy_node(self):
        self.cleanup()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = AvoidNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()