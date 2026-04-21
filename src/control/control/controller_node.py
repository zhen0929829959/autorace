#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
import threading
from enum import Enum

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MissionState(Enum):
    FOLLOW_DUAL = 'follow_dual'
    FOLLOW_WHITE = 'follow_white'
    FOLLOW_YELLOW = 'follow_yellow'
    TRAFFICLIGHT = 'trafficlight'
    WAIT_TURN = 'wait_turn'
    WAIT_WORK='wait_work'
    WAIT_OBSTACLE = 'wait_obstacle'   # 已看到施工號誌，持續循白線等障礙
    AVOID = 'avoid'
    WAIT_PARK = 'wait_park'           # 已看到 P，切循黃線等停車
    PARKING = 'parking'               # 正在停車
    WAIT_DUAL = 'wait_dual'
    WAIT_ROW = 'wait_row'
    TUNNEL = 'tunnel'
    FINISH = 'finish'


class ControllerNode(Node):
    def __init__(self):
        super().__init__('controller_node')

        # ========= 狀態 =========
        self.state = MissionState.TRAFFICLIGHT
        self.follow_mode = 'none'
        self.drive_mode = 'controller'
        # self.state = MissionState.TUNNEL
        # self.follow_mode = 'yellow'
        # self.drive_mode = 'controller'
        # self.state = MissionState.WAIT_ROW
        # self.follow_mode = 'dual'
        # self.drive_mode = 'line_follow'

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
        self.left_distance = 9999.0
        self.right_distance = 9999.0

        # ========= 迷宮 / 貼右牆參數 =========
        self.in_wall_follow_mode = True   # True=貼牆模式, False=避障找牆模式

        self.DIST_THRESHOLD = 290         # 前方障礙門檻260
        self.RIGHT_THRESHOLD = 300        # 判斷右邊牆不見的門檻
        self.IDEAL_WALL_DISTANCE = 200    # 理想貼牆距離
        self.DEAD_ZONE = 20               # 誤差容許範圍
        self.KP = 1                   # 比例控制增益
        self.exit_THRESHOLD = 400

        self.maze_big = 120
        self.maze_small = 10
        self.maze_base_speed = 150

        self.found=False
        self.road_center=0
        self.no_line_count = 0
        self.detect_green=False

        # 避障/停車保護
        self.avoid_running = False
        self.parking_running = False
        self.parking_distance_threshold = 850
        self.parking_stop_dist = 210
        self.line_count=0

        # ========= Publisher =========
        self.follow_mode_pub = self.create_publisher(String, '/follow_mode', 10)
        self.drive_mode_pub = self.create_publisher(String, '/drive_mode', 10)
        self.motor_cmd_pub = self.create_publisher(String, '/controller/motor_cmd', 10)

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
        self.lane_sub = self.create_subscription(
            String,
            '/lane_info',
            self.lane_callback,
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

    def parse_lane_msg(self, raw_text):
        try:
            data = json.loads(raw_text)
            return data
        except Exception as e:
            self.get_logger().warn(f'Lane JSON parse failed: {e}')
            return None

    # =====================================================
    # 判斷前方是否有障礙
    # =====================================================
    def is_obstacle_ahead(self):
        self.get_logger().info(f'front_distance = {self.front_distance}')
        if self.front_distance < 225:
            self.get_logger().info('front yes')
            return True
        return False

    # =====================================================
    # 停車動作
    # =====================================================
    def right_parking(self):
        # 左側有障礙物，向右停車
        self.publish_motor_cmd(150, 150)
        time.sleep(1.6)
        self.publish_motor_cmd(150, -150)  
        time.sleep(0.9)
        self.publish_motor_cmd(150, 150)
        time.sleep(2.4)
        self.publish_motor_cmd(0, 0)
        time.sleep(2.0)
        self.publish_motor_cmd(-150, -150)
        time.sleep(2.4)
        self.publish_motor_cmd(150, -150)   
        time.sleep(1.1)
        self.publish_motor_cmd(150, 150)
        time.sleep(1)

    def left_parking(self):
        # 右側有障礙物，向左停車
        self.publish_motor_cmd(150, 150)
        time.sleep(1.7)
        self.publish_motor_cmd(-150, 150)   
        time.sleep(1.1)
        self.publish_motor_cmd(150, 150)
        time.sleep(2.7)
        self.publish_motor_cmd(0, 0)
        time.sleep(2.0)
        self.publish_motor_cmd(-150, -150)
        time.sleep(2.6)
        self.publish_motor_cmd(-150, 150)  
        time.sleep(1.1)
        self.publish_motor_cmd(150, 150)
        time.sleep(1)

    # =====================================================
    # 避障流程
    # =====================================================
    def run_avoid_sequence(self):
        if self.avoid_running:
            return

        self.avoid_running = True
        self.get_logger().info('Start avoid')

        # 切成 controller 接管馬達
        self.drive_mode = 'controller'
        self.publish_modes()

        # 第一段
        self.publish_motor_cmd(-80, 80)  
        time.sleep(2.1)
        self.publish_motor_cmd(140, 140)
        time.sleep(2.8)
        self.publish_motor_cmd(80, -80)  
        time.sleep(2.1)

        # 第二段：直走直到前方再次看到障礙
        while rclpy.ok():
            if self.is_obstacle_ahead():
                self.get_logger().info('>> 前方有障礙，準備進入下一段')
                break
            else:
                self.get_logger().info('>> 前方清空，直行')
                self.publish_motor_cmd(190, 190)

            time.sleep(0.1)

        # 第三段：左轉 -> 直走 -> 右轉 -> 直走
        self.publish_motor_cmd(80, -80)   
        time.sleep(1.9)
        self.publish_motor_cmd(140, 140)
        time.sleep(2.8)
        self.publish_motor_cmd(-80, 80) 
        time.sleep(1.8)
        self.publish_motor_cmd(140, 140)
        time.sleep(1.0)

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
    # 停車流程
    # =====================================================
    def run_parking_sequence(self):
        if self.parking_running:
            return

        self.parking_running = True
        self.get_logger().info('Start parking')

        self.drive_mode = 'controller'
        self.publish_modes()
        self.publish_motor_cmd(0, 0)
        time.sleep(0.2)
        self.get_logger().info(
            f'Parking decision | left={self.left_distance:.1f}, right={self.right_distance:.1f}'
        )

        if self.right_distance < self.parking_distance_threshold:
            self.get_logger().info('右側有障礙物，向左停車')
            self.left_parking()
        elif self.left_distance < self.parking_distance_threshold:
            self.get_logger().info('左側有障礙物，向右停車')
            self.right_parking()
        else:
            self.get_logger().warn('沒找到可判斷的停車參考，先停車')
            self.publish_motor_cmd(0, 0)

        self.set_state(
            MissionState.WAIT_DUAL,
            follow_mode='yellow',
            drive_mode='line_follow',
            reason='Parking finished,wait dual'
        )

        self.parking_running = False
        self.get_logger().info('Parking finished')

    def maze_open_control(self):
        front = self.front_distance
        right = self.right_distance

        self.get_logger().info(
            f'[MAZE] front={front:.1f}, right={right:.1f}, wall_follow={self.in_wall_follow_mode}'
        )

        if self.in_wall_follow_mode:
            if front < self.DIST_THRESHOLD:
                # 前方有障礙，切到避障找牆模式
                self.in_wall_follow_mode = False
                self.publish_motor_cmd(10, 160)
                self.get_logger().info('MAZE: left to avoid')

            elif right > self.RIGHT_THRESHOLD:
                # 右邊沒牆，右轉找牆
                self.publish_motor_cmd(160, 10)
                self.get_logger().info('MAZE: right to find wall')

            elif abs(right - self.IDEAL_WALL_DISTANCE) > self.DEAD_ZONE:
                # 距離牆太近或太遠，做比例修正
                error = right - self.IDEAL_WALL_DISTANCE
                adjustment = int(self.KP * error)
                adjustment = max(min(adjustment, 50), -50)

                self.publish_motor_cmd(
                    self.maze_base_speed + adjustment,
                    self.maze_base_speed - adjustment
                )
                self.get_logger().info(f'MAZE: adjusting {adjustment}')

            else:
                # 在理想範圍內，直走
                self.publish_motor_cmd(self.maze_base_speed, self.maze_base_speed)
                self.get_logger().info('MAZE: go straight')
        else:
            # 避障 / 找牆模式
            if front < self.DIST_THRESHOLD:
                self.publish_motor_cmd(5, 190)
                self.get_logger().info('MAZE: avoiding...')

            elif right < self.RIGHT_THRESHOLD:
                self.in_wall_follow_mode = True     
                self.publish_motor_cmd(self.maze_base_speed, self.maze_base_speed)
                self.get_logger().info('MAZE: back to wall follow')

            else:
                self.publish_motor_cmd(160, 10)
                self.get_logger().info('MAZE: searching for wall...')

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

        # 一開始雙線，看到左右轉號誌 -> 切白/黃線
        if self.state == MissionState.WAIT_TURN:
            if class_id == 6:
                self.set_state(
                    MissionState.FOLLOW_WHITE,
                    follow_mode='white',
                    drive_mode='line_follow',
                    reason=f'Detected turn sign class {class_id}'
                )
            elif class_id == 2:
                self.set_state(
                    MissionState.FOLLOW_YELLOW,
                    follow_mode='yellow',
                    drive_mode='controller',
                    reason=f'Detected turn sign class {class_id}'
                )
                self.publish_motor_cmd(50, 150)
                time.sleep(1.9)
                self.set_state(
                    MissionState.FOLLOW_YELLOW,
                    follow_mode='yellow',
                    drive_mode='line_follow',
                    reason=f'Detected turn sign class {class_id}'
                )
        # 白線/黃線中看到施工號誌或 P
        elif self.state in [MissionState.FOLLOW_WHITE, MissionState.FOLLOW_YELLOW, MissionState.FOLLOW_DUAL]:
            # if class_id == 1:
            #     self.set_state(
            #         MissionState.WAIT_WORK,
            #         follow_mode='dual',
            #         drive_mode='line_follow',
            #         reason='Detected construction sign (class 5), keep following white line'
            #     )
            if class_id == 5:
                self.set_state(
                    MissionState.WAIT_OBSTACLE,
                    follow_mode='white',
                    drive_mode='line_follow',
                    reason='Detected construction sign (class 5), keep following white line'
                )
            elif class_id == 3:
                self.set_state(
                    MissionState.WAIT_PARK,
                    follow_mode='yellow',
                    drive_mode='line_follow',
                    reason='Detected P sign (class 3), switch to yellow line'
                )
        elif self.state == MissionState.WAIT_DUAL:
            if class_id == 2: 
                self.set_state(
                    MissionState.WAIT_ROW,
                    follow_mode='dual',
                    drive_mode='line_follow',
                    reason=f'WAIT_ROW {class_id}'
                )
        elif self.state == MissionState.WAIT_ROW:
            if class_id == 7:
                self.set_state(
                    MissionState.WAIT_ROW,
                    follow_mode='dual',
                    drive_mode='controller',
                    reason=f'stop to WAIT_ROW {class_id}'
                )
                self.publish_motor_cmd(0, 0)
            elif class_id == 4:
                self.in_wall_follow_mode = True
                self.set_state(
                    MissionState.TUNNEL,
                    follow_mode='yellow',
                    drive_mode='controller',
                    reason=f'in TUNNEL {class_id}'
                )
                # self.publish_motor_cmd(0, 0) #####
                # time.sleep(0.5)
                self.publish_motor_cmd(200, 200)
                time.sleep(1.3)
                # self.publish_motor_cmd(0, 0) #####
                # time.sleep(0.5)
            else:
                self.set_state(
                    MissionState.WAIT_ROW,
                    follow_mode='dual',
                    drive_mode='line_follow',
                    reason=f'WAIT_ROW {class_id}'
                )

    def lidar_callback(self, msg):
        data = self.parse_lidar_msg(msg.data)
        if data is None:
            return

        self.front_distance = data.get('front_min', 9999.0)
        self.left_distance = data.get('left_min', 9999.0)
        self.right_distance = data.get('right_min', 9999.0)

        # 看到施工號誌後，開始等前方障礙
        if self.state == MissionState.WAIT_OBSTACLE:
            if self.is_obstacle_ahead() and not self.avoid_running:
                self.set_state(
                    MissionState.AVOID,
                    follow_mode='white',
                    drive_mode='controller',
                    reason=f'Obstacle detected, front={self.front_distance}'
                )
                threading.Thread(target=self.run_avoid_sequence, daemon=True).start()

        # #看有無線!!!!
        # #看到 P 後，循黃線，等左右出現停車判斷條件
        # elif self.state == MissionState.WAIT_PARK:
        #     if not self.parking_running:
        #         if self.no_line_count > 2:
        #             self.get_logger().info('黃線連續消失，直接停車結束')
        #             self.set_state(
        #                 MissionState.PARKING,
        #                 follow_mode='yellow',
        #                 drive_mode='controller',
        #                 reason='Yellow line lost continuously, stop parking'
        #             )
        #             self.publish_motor_cmd(0, 0)
        #             time.sleep(0.2)
        #             if self.right_distance < self.parking_distance_threshold:
        #                 self.set_state(
        #                     MissionState.PARKING,
        #                     follow_mode='yellow',
        #                     drive_mode='controller',
        #                     reason=f'Right side occupied, start left parking | right={self.right_distance}'
        #                 )
        #                 self.run_parking_sequence()
        #             elif self.left_distance < self.parking_distance_threshold:
        #                 self.set_state(
        #                     MissionState.PARKING,
        #                     follow_mode='yellow',
        #                     drive_mode='controller',
        #                     reason=f'Left side occupied, start right parking | left={self.left_distance}'
        #                 )
        #                 self.run_parking_sequence()

        # 看有無障      看到 P 後，循黃線，等左右出現停車判斷條件
        elif self.state == MissionState.WAIT_PARK:
            if not self.parking_running:
                if self.right_distance < self.parking_stop_dist or self.left_distance < self.parking_stop_dist:
                    self.get_logger().info('有障礙物，停車')
                    self.set_state(
                        MissionState.PARKING,
                        follow_mode='yellow',
                        drive_mode='controller',
                        reason='Yellow line lost continuously, stop parking'
                    )
                    self.publish_motor_cmd(0, 0)
                    time.sleep(0.2)
                    if self.right_distance < self.parking_distance_threshold:
                        self.set_state(
                            MissionState.PARKING,
                            follow_mode='yellow',
                            drive_mode='controller',
                            reason=f'Right side occupied, start left parking | right={self.right_distance}'
                        )
                        self.run_parking_sequence()
                    elif self.left_distance < self.parking_distance_threshold:
                        self.set_state(
                            MissionState.PARKING,
                            follow_mode='yellow',
                            drive_mode='controller',
                            reason=f'Left side occupied, start right parking | left={self.left_distance}'
                        )
                        self.run_parking_sequence()

        elif self.state == MissionState.TUNNEL:
            self.maze_open_control()
            # if self.right_distance > self.exit_THRESHOLD and self.front_distance > self.exit_THRESHOLD :
            #     self.get_logger().warn(f'出去啦')
            #     self.set_state(
            #         MissionState.FINISH,
            #         follow_mode='dual',
            #         drive_mode='line_follow',
            #         reason=f'TUNNEL FINISH to follow line'
            #     )

    def lane_callback(self, msg):
        ddd = self.parse_lane_msg(msg.data)
        if ddd is None:
            return

        self.found = ddd.get('found', False)
        self.road_center = ddd.get('road_center', 0)
        self.detect_green=ddd.get('detect_green', False)
        # if self.state == MissionState.WAIT_PARK:
        #     if self.road_center > 350:
        #         self.no_line_count += 1
        #         self.get_logger().warn(f'黃線消失 {self.no_line_count} 次')
        #     else:
        #         self.no_line_count = 0
        # else:
        #     self.no_line_count = 0
        if self.state == MissionState.TRAFFICLIGHT:
            if self.detect_green :
                self.set_state(
                    MissionState.WAIT_TURN,
                    follow_mode='dual',
                    drive_mode='line_follow',
                    reason=f'detect TRAFFICLIGHT, WAIT_TURN'
                )
            else:
                self.set_state(
                    MissionState.TRAFFICLIGHT,
                    follow_mode='none',
                    drive_mode='controller',
                    reason='wait TRAFFICLIGHT'
                )
                self.publish_motor_cmd(0, 0)
                time.sleep(0.2)

        # if self.state == MissionState.TUNNEL:
        #     if self.found :
        #         self.line_count += 1
        #         self.get_logger().warn(f'找到黃線 {self.no_line_count} 次')
        #         if self.line_count >5:
        #             self.set_state(
        #                 MissionState.FINISH,
        #                 follow_mode='dual',
        #                 drive_mode='line_follow',
        #                 reason=f'TUNNEL FINISH to follow line'
        #             )
        #     else:
        #         self.line_count = 0
        if self.state == MissionState.TUNNEL:
            if self.found :
                self.get_logger().warn(f'找到黃線 {self.no_line_count} 次')
                ###########333###
                # self.set_state(
                #     MissionState.FINISH,
                #     follow_mode='dual',
                #     drive_mode='line_follow',
                #     reason=f'TUNNEL FINISH to follow line'
                # )

        

        # if self.state == MissionState.WAIT_PARK:
        #     if not self.found:
        #         self.no_line_count += 1
        #         self.get_logger().warn(f'黃線消失 {self.no_line_count} 次')
        #     else:
        #         self.no_line_count = 0
        # else:
        #     self.no_line_count = 0



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