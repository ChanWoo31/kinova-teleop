from dynamixel_sdk import *
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

def deg2rad(x):
    y = x * np.pi / 180
    return y

class DynamixelVal(Node):
    def __init__(self):
        super().__init__('dynamixel_teleop_master')
        self.publisher_ = self.create_publisher(Float32MultiArray, 'dynamixel_pose', 10)
        timer_period = 0.01
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

        self.d_ur5e = [0.1625, 0, 0, 0.1333, 0.0997, 0.0996]
        self.a_ur5e = [0, -0.425, -0.3922, 0, 0, 0]
        self.alpha_ur5e = [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]

        self.dynamixel_angle = np.zeros(6)

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
        msg = Float32MultiArray()

        q = [0, 0, 0, 0, 0, 0]

        
        self.BulkRead.txRxPacket()
        for i in range(6):
            q[i] = self.BulkRead.getData(self.motor_ids[i], self.ADDR_PRESENT_POSITION, self.data_length_4byte)

        dir = [1, 1, -1, 1, 1, 1]
        offset = [0, -90, 0, -90, 0, 180]

        for i in range(6):
            self.dynamixel_angle[i] = deg2rad((q[i] - 2048) * dir[i] * 360 / 4096 + offset[i])
        
        T06 = np.eye(4)
        T06 = self.forward_kinematic(self.dynamixel_angle, self.d_ur5e, self.a_ur5e, self.alpha_ur5e)
        T06[:3, 3] = T06[:3, 3] * 0.7

        msg.data = T06.flatten().tolist()
        self.publisher_.publish(msg)
        # self.get_logger().info(str(msg))
        print(T06)

    def forward_kinematic(self, theta, d, a, alpha):
        T = np.eye(4)
        for i in range(6):
            T_i = self.Trans_mat(theta[i], d[i], a[i], alpha[i])
            T = T @ T_i
        return T
    
    def Trans_mat(self, theta, d, a, alpha):
        T = np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])
        return T

def main():
    rclpy.init()
    node = DynamixelVal()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
