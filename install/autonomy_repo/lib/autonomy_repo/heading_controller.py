#!/usr/bin/env python3
import numpy
import rclpy
from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState

class HeadingController(BaseHeadingController):
    def __init__(self) -> None:
        super().__init__("heading_controller")
        self.kp = 2.0
    def compute_control_with_goal(self, state: TurtleBotState, goal: TurtleBotState) -> TurtleBotControl:
        heading_error = wrap_angle(goal.theta - state.theta)
        control_out = TurtleBotControl()
        control_out.omega = self.kp*heading_error
        return control_out
    
if __name__ == "__main__":
    rclpy.init(args=None)
    heading_controller_node = HeadingController()
    rclpy.spin(heading_controller_node)
    rclpy.shutdown()