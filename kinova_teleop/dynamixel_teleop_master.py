from dynamixel_sdk import *
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt32MultiArray

class DynamixelVal(Node):
    def __init__(self):
        super().__init__('dynamixel_val')
        self.publisher_ = self.create_publisher(UInt32MultiArray, 'topic', 10)
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.read_motor_position)

        self.DEVICENAME = '/dev/ttyUSB0'
        self.BAUDRATE = 57600

        self.motor_ids = [1, 2, 3, 4, 5, 6]

        self.ADDR_TORQUE_ENABLE = 64
        self.ADDR_GOAL_POSITION = 116
        self.ADDR_PRESENT_POSITION = 132

        self.PROTOCOL_VERSION = 2

        self.TORQUE_ENABLE = 1
        self.TORQUE_DISABLE = 0

        self.portHandler = PortHandler(self.DEVICENAME)
        self.packetHandler = PacketHandler(self.PROTOCOL_VERSION)

        if self.portHandler.openPort():
            print("포트 열기 성공")
        else:
            print("포트 열기 실패")
            exit()

        if self.portHandler.setBaudRate(57600):
            print("보드레이트 성공")
        else:
            print("보드레이트 실패")
            exit()

        self.BulkRead = GroupBulkRead(self.portHandler, self.packetHandler)
        self.BulkWrite = GroupBulkWrite(self.portHandler, self.packetHandler)

        for id in self.motor_ids:
            self.packetHandler.write1ByteTxRx(self.portHandler, id, self.ADDR_TORQUE_ENABLE, 0)

        self.data_length_4byte = 4
        for i in range(6):
            self.BulkRead.addParam(self.motor_ids[i], self.ADDR_PRESENT_POSITION, self.data_length_4byte)
    
    
    def read_motor_position(self):
        msg = UInt32MultiArray()
        msg.data = [0, 0, 0, 0, 0, 0]

        
        self.BulkRead.txRxPacket()
        for i in range(6):
            msg.data[i] = self.BulkRead.getData(self.motor_ids[i], self.ADDR_PRESENT_POSITION, self.data_length_4byte)
        self.publisher_.publish(msg)
        # self.get_logger().info(str(msg))

def main():
    rclpy.init()
    node = DynamixelVal()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
