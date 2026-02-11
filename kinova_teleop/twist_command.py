import sys
import os
import time
import numpy as np

# ROS2 관련
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

# Kortex API 관련
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2

# [중요] utilities 모듈 불러오기
# utilities.py가 상위 폴더("..")에 있다고 가정 (예제 구조)
# 만약 같은 폴더에 있다면 sys.path 라인은 지우고 바로 import utilities 하세요.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import utilities

class KinovaRun(Node):
    def __init__(self):
        super().__init__('Kinova_Run')
        
        # 1. ROS2 Subscriber
        self.kinova_run = self.create_subscription(
            Float32MultiArray, 
            'dynamixel_joint_raw', 
            self.subscribe_topic, 
            10
        )
        
        # 2. 목표 위치 저장 변수
        self.target_pose = None

        # ---------------------------------------------------------
        # [핵심] utilities를 사용하여 연결하기
        # ---------------------------------------------------------
        
        # (A) ArgumentParser 대신 가짜 args 객체 만들기
        # ROS2 실행 시 인자 충돌을 막고, IP를 직접 지정하기 위함입니다.
        class ConnectionArgs:
            def __init__(self):
                self.ip = "192.168.1.10" 
                self.username = "admin"
                self.password = "admin"
                self.session_inactivity_timeout_ms = 60000
                self.connection_inactivity_timeout_ms = 2000

        args = ConnectionArgs()

        # (B) utilities.DeviceConnection 생성 (with문 안 씀!)
        try:
            self.device_connection = utilities.DeviceConnection.createTcpConnection(args)
            
            # (C) 수동으로 연결 열기 (__enter__)
            # 원래 'with' 문이 해주던 것을 수동으로 합니다.
            self.router = self.device_connection.__enter__()
            
            # (D) 클라이언트 생성
            self.base = BaseClient(self.router)
            self.base_cyclic = BaseCyclicClient(self.router)
            
            print(" utilities를 통해 로봇 연결 성공!")

        except Exception as e:
            print(f" 연결 실패: {e}")
            raise e
        
        # 3. 제어 루프 타이머 (0.01초)
        self.timer = self.create_timer(0.01, self.control_loop)

    def subscribe_topic(self, msg):
        # 데이터 수신 및 전처리
        pose = np.array(msg.data).reshape(4, 4)
        
        # [안전 장치]
        pose[:3, 3] = pose[:3, 3] * 0.5   # 스케일링
        pose[0, 3] += 0.2               # 오프셋
        pose[2, 3] = np.clip(pose[2, 3], 0.2, 0.6) # 높이 제한

        self.target_pose = pose

    def control_loop(self):
        if self.target_pose is None:
            return

        try:
            # 1. 현재 위치 (Feedback)
            feedback = self.base_cyclic.RefreshFeedback()
            curr_x = feedback.base.tool_pose_x
            curr_y = feedback.base.tool_pose_y
            curr_z = feedback.base.tool_pose_z

            # 2. 목표 위치
            target_x = self.target_pose[0, 3]
            target_y = self.target_pose[1, 3]
            target_z = self.target_pose[2, 3]

            # 3. 속도 계산 (P-Control)
            kp = 2.0
            vel_x = (target_x - curr_x) * kp
            vel_y = (target_y - curr_y) * kp
            vel_z = (target_z - curr_z) * kp

            # 4. 속도 명령 생성
            command = Base_pb2.TwistCommand()
            command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
            command.twist.linear_x = vel_x
            command.twist.linear_y = vel_y
            command.twist.linear_z = vel_z
            command.twist.angular_x = 0.0
            command.twist.angular_y = 0.0
            command.twist.angular_z = 0.0

            # 5. 전송
            self.base.SendTwistCommand(command)

        except Exception as e:
            print(f"Control Error: {e}")

    def __del__(self):
        # 노드 종료 시 연결 해제 (__exit__)
        print("연결 해제 중...")
        if hasattr(self, 'base'):
            # 로봇 정지 명령
            twist = Base_pb2.TwistCommand()
            self.base.SendTwistCommand(twist)
            
        if hasattr(self, 'device_connection'):
            # 수동으로 연결 닫기
            self.device_connection.__exit__(None, None, None)

def main():
    rclpy.init()
    try:
        node = KinovaRun()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()