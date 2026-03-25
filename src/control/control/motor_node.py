#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from typing import Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

try:
    from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS
except ImportError:
    PortHandler = None
    PacketHandler = None
    COMM_SUCCESS = None


class MotorNode(Node):
    """
    訂閱 /motor_cmd，將左右輪速度送到 Dynamixel。

    目前預設使用 std_msgs/String + JSON：
    {
        "left_speed": 120,
        "right_speed": 180
    }

    也支援：
    {
        "left": 120,
        "right": 180
    }
    """

    ADDR_GOAL_VELOCITY = 104
    ADDR_TORQUE_ENABLE = 64
    PROTOCOL_VERSION = 2.0
    TORQUE_ENABLE = 1
    TORQUE_DISABLE = 0

    def __init__(self) -> None:
        super().__init__('motor_node')

        # -------- ROS2 Parameters --------
        self.declare_parameter('device_name', '/dev/ttyACM0')
        self.declare_parameter('baudrate', 1000000)
        self.declare_parameter('dxl_left_id', 1)
        self.declare_parameter('dxl_right_id', 2)
        self.declare_parameter('max_speed', 500)
        self.declare_parameter('invert_left', True)
        self.declare_parameter('invert_right', True)
        self.declare_parameter('stop_on_invalid_cmd', True)

        self.device_name = self.get_parameter('device_name').value
        self.baudrate = int(self.get_parameter('baudrate').value)
        self.dxl_left_id = int(self.get_parameter('dxl_left_id').value)
        self.dxl_right_id = int(self.get_parameter('dxl_right_id').value)
        self.max_speed = int(self.get_parameter('max_speed').value)
        self.invert_left = bool(self.get_parameter('invert_left').value)
        self.invert_right = bool(self.get_parameter('invert_right').value)
        self.stop_on_invalid_cmd = bool(self.get_parameter('stop_on_invalid_cmd').value)

        self.subscription = self.create_subscription(
            String,
            '/motor_cmd',
            self.motor_cmd_callback,
            10,
        )

        self.port_handler = None
        self.packet_handler = None
        self.hardware_ready = False

        self.init_dynamixel()

        self.get_logger().info('motor_node started, subscribing to /motor_cmd')

    # ------------------------------------------------------------------
    # Dynamixel init
    # ------------------------------------------------------------------
    def init_dynamixel(self) -> None:
        if PortHandler is None or PacketHandler is None:
            self.get_logger().error(
                'dynamixel_sdk not found. Install it first, or this node cannot control motors.'
            )
            return

        self.port_handler = PortHandler(self.device_name)
        self.packet_handler = PacketHandler(self.PROTOCOL_VERSION)

        if not self.port_handler.openPort():
            self.get_logger().error(f'Failed to open port: {self.device_name}')
            return

        if not self.port_handler.setBaudRate(self.baudrate):
            self.get_logger().error(f'Failed to set baudrate: {self.baudrate}')
            self.port_handler.closePort()
            return

        ok_left = self.enable_torque(self.dxl_left_id)
        ok_right = self.enable_torque(self.dxl_right_id)

        self.hardware_ready = ok_left and ok_right

        if self.hardware_ready:
            self.get_logger().info(
                f'Dynamixel ready. device={self.device_name}, baudrate={self.baudrate}, '
                f'left_id={self.dxl_left_id}, right_id={self.dxl_right_id}'
            )
        else:
            self.get_logger().error('Dynamixel initialization incomplete. Motors will not move correctly.')

    def enable_torque(self, dxl_id: int) -> bool:
        dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(
            self.port_handler,
            dxl_id,
            self.ADDR_TORQUE_ENABLE,
            self.TORQUE_ENABLE,
        )

        if dxl_comm_result != COMM_SUCCESS:
            self.get_logger().error(
                f'Enable torque failed for ID {dxl_id}: '
                f'{self.packet_handler.getTxRxResult(dxl_comm_result)}'
            )
            return False

        if dxl_error != 0:
            self.get_logger().error(
                f'Dynamixel error while enabling torque for ID {dxl_id}: '
                f'{self.packet_handler.getRxPacketError(dxl_error)}'
            )
            return False

        self.get_logger().info(f'Torque enabled for motor ID {dxl_id}')
        return True

    # ------------------------------------------------------------------
    # Motor command handling
    # ------------------------------------------------------------------
    def motor_cmd_callback(self, msg: String) -> None:
        left_speed, right_speed = self.parse_motor_cmd(msg.data)

        if left_speed is None or right_speed is None:
            self.get_logger().warn(f'Invalid /motor_cmd payload: {msg.data}')
            if self.stop_on_invalid_cmd:
                self.set_motor_speed(0, 0)
            return

        self.set_motor_speed(left_speed, right_speed)

    def parse_motor_cmd(self, raw: str) -> Tuple[int, int]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None, None

        left_speed = data.get('left_speed', data.get('left'))
        right_speed = data.get('right_speed', data.get('right'))

        if left_speed is None or right_speed is None:
            return None, None

        try:
            return int(left_speed), int(right_speed)
        except (TypeError, ValueError):
            return None, None

    def set_motor_speed(self, left_speed: int, right_speed: int) -> None:
        if not self.hardware_ready:
            self.get_logger().warn('Hardware not ready, motor command ignored.')
            return

        left_speed = max(min(int(left_speed), self.max_speed), -self.max_speed)
        right_speed = max(min(int(right_speed), self.max_speed), -self.max_speed)

        if self.invert_left:
            left_cmd = -left_speed
        else:
            left_cmd = left_speed

        if self.invert_right:
            right_cmd = -right_speed
        else:
            right_cmd = right_speed

        left_result, left_error = self.packet_handler.write4ByteTxRx(
            self.port_handler,
            self.dxl_left_id,
            self.ADDR_GOAL_VELOCITY,
            int(left_cmd),
        )
        right_result, right_error = self.packet_handler.write4ByteTxRx(
            self.port_handler,
            self.dxl_right_id,
            self.ADDR_GOAL_VELOCITY,
            int(right_cmd),
        )

        if left_result != COMM_SUCCESS:
            self.get_logger().error(
                f'Left motor write failed: {self.packet_handler.getTxRxResult(left_result)}'
            )
        elif left_error != 0:
            self.get_logger().error(
                f'Left motor packet error: {self.packet_handler.getRxPacketError(left_error)}'
            )

        if right_result != COMM_SUCCESS:
            self.get_logger().error(
                f'Right motor write failed: {self.packet_handler.getTxRxResult(right_result)}'
            )
        elif right_error != 0:
            self.get_logger().error(
                f'Right motor packet error: {self.packet_handler.getRxPacketError(right_error)}'
            )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def stop_and_cleanup(self) -> None:
        if not self.port_handler or not self.packet_handler:
            return

        try:
            if self.hardware_ready:
                self.set_motor_speed(0, 0)

                for dxl_id in [self.dxl_left_id, self.dxl_right_id]:
                    self.packet_handler.write1ByteTxRx(
                        self.port_handler,
                        dxl_id,
                        self.ADDR_TORQUE_ENABLE,
                        self.TORQUE_DISABLE,
                    )

            self.port_handler.closePort()
            self.get_logger().info('Motor stopped and port closed.')
        except Exception as exc:
            self.get_logger().error(f'Cleanup error: {exc}')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MotorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_and_cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()