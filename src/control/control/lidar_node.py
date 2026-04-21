#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rplidar import RPLidar


class LidarNode(Node):
    def __init__(self):
        super().__init__('lidar_node')

        # ===== Lidar 參數 =====
        self.port_name = '/dev/ttyUSB0'

        # 角度範圍（度）
        self.front_range = (350, 40)   # 前方跨 0 度
        self.left_range = (270, 340)
        self.right_range = (20, 100)

        # 儲存每個角度的距離，單位 mm
        self.scan_data = [(angle, 9999.0) for angle in range(360)]

        # 最新最小距離
        self.front_min = 9999.0
        self.left_min = 9999.0
        self.right_min = 9999.0

        # 執行保護
        self.scan_lock = threading.Lock()
        self.running = True

        # Publisher
        self.lidar_info_pub = self.create_publisher(String, '/lidar_info', 10)

        # 啟動雷達
        self.lidar = RPLidar(self.port_name)
        self.lidar.start_motor()
        time.sleep(1.0)

        # 背景掃描執行緒
        self.scan_thread = threading.Thread(target=self.lidar_loop, daemon=True)
        self.scan_thread.start()

        # 定時發布最小距離
        self.timer = self.create_timer(0.1, self.publish_lidar_info)

        self.get_logger().info('Lidar Node started: open lidar and publish /lidar_info')

    # =====================================================
    # 工具函式
    # =====================================================
    def in_range(self, angle, angle_range):
        start, end = angle_range
        if start > end:
            return angle >= start or angle <= end
        return start <= angle <= end

    def get_min_distance(self, angle_range):
        with self.scan_lock:
            dists = [
                dist for angle, dist in self.scan_data
                if self.in_range(angle, angle_range) and dist > 0
            ]
        return min(dists) if dists else 9999.0

    # =====================================================
    # Lidar 掃描執行緒
    # =====================================================
    def lidar_loop(self):
        try:
            for scan in self.lidar.iter_scans():
                if not self.running:
                    break

                temp_data = [(angle, 9999.0) for angle in range(360)]

                for measurement in scan:
                    if not isinstance(measurement, (list, tuple)) or len(measurement) != 3:
                        continue

                    quality, angle, dist = measurement
                    a = int(angle)

                    if 0 <= a < 360:
                        temp_data[a] = (a, dist)

                with self.scan_lock:
                    self.scan_data = temp_data
                    self.front_min = self.get_min_distance_no_lock(self.front_range)
                    self.left_min = self.get_min_distance_no_lock(self.left_range)
                    self.right_min = self.get_min_distance_no_lock(self.right_range)

        except Exception as e:
            self.get_logger().error(f'Lidar loop error: {e}')

    def get_min_distance_no_lock(self, angle_range):
        dists = [
            dist for angle, dist in self.scan_data
            if self.in_range(angle, angle_range) and dist > 0
        ]
        return min(dists) if dists else 9999.0

    # =====================================================
    # 發布資料
    # =====================================================
    def publish_lidar_info(self):
        msg = String()
        msg.data = json.dumps({
            'front_min': round(self.front_min, 1),
            'left_min': round(self.left_min, 1),
            'right_min': round(self.right_min, 1)
        })
        self.lidar_info_pub.publish(msg)

    # =====================================================
    # 結束清理
    # =====================================================
    def destroy_node(self):
        self.get_logger().info('Stopping lidar node...')
        self.running = False

        try:
            self.lidar.stop()
            self.lidar.stop_motor()
            self.lidar.disconnect()
        except Exception as e:
            self.get_logger().warn(f'Lidar cleanup warning: {e}')

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LidarNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Lidar node stopped by user.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()