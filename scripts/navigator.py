#!/usr/bin/env python3
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
from asl_tb3_lib.tf_utils import quaternion_to_yaw
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from scipy.interpolate import splev, splrep
import numpy as np
import typing as T
import rclpy
import matplotlib.pyplot as plt
from asl_tb3_lib.grids import StochOccupancyGrid2D
from astar import AStar, DetOccupancyGrid2D
# from utils import plot_line_segments

class Navigator(BaseNavigator):
    def __init__(self) -> None:
        # give it a default node name
        super().__init__("navigator")
        self.kp = 2.0
        self.V_prev = 0.
        self.om_prev = 0.
        self.t_prev = 0.
        self.V_PREV_THRES = 0.0001
        self.kpx = 2
        self.kpy = 2
        self.kdx = 2
        self.kdy = 2
    def compute_heading_control(self, state: TurtleBotState, goal: TurtleBotState) -> TurtleBotControl:
        heading_error = wrap_angle(goal.theta - state.theta)
        control_out = TurtleBotControl()
        control_out.omega = self.kp*heading_error
        return control_out
    def compute_trajectory_tracking_control(self, state: TurtleBotState, plan: TrajectoryPlan, t: float) -> TurtleBotControl:
        x_d = splev(t, plan.path_x_spline, der=0)
        y_d = splev(t, plan.path_y_spline, der=0)
        xd_d = splev(t, plan.path_x_spline, der=1)
        yd_d = splev(t, plan.path_y_spline, der=1)
        xdd_d = splev(t, plan.path_x_spline, der=2)
        ydd_d = splev(t, plan.path_y_spline, der=2)
        x = state.x
        y = state.y
        th = state.theta
        dt = t - self.t_prev

        xd = self.V_prev*np.cos(th)
        yd = self.V_prev*np.sin(th)
        u1 = xdd_d + self.kpx*(x_d-x) + self.kdx*(xd_d-xd)
        u2 = ydd_d + self.kpy*(y_d-y) + self.kdy*(yd_d-yd)
        if self.V_prev < self.V_PREV_THRES:
            self.V_prev = self.V_PREV_THRES
        M = np.array([
            [np.cos(th), -self.V_prev*np.sin(th)],
            [np.sin(th),  self.V_prev*np.cos(th)]
        ])
        sol = np.linalg.solve(M,np.array([u1,u2]))
        V = self.V_prev + sol[0]*dt
        om = sol[1]

        control_out = TurtleBotControl()
        control_out.v = V
        control_out.omega = om
        # save the commands that were applied and the time
        self.t_prev = t
        self.V_prev = V
        self.om_prev = om
        return control_out
    def compute_trajectory_plan(self,state: TurtleBotState,goal: TurtleBotState,occupancy: StochOccupancyGrid2D,resolution: float,horizon: float) -> T.Optional[TrajectoryPlan]:
        """ Compute a trajectory plan using A* and cubic spline fitting
        Args:
            state (TurtleBotState): state
            goal (TurtleBotState): goal
            occupancy (StochOccupancyGrid2D): occupancy
            resolution (float): resolution
            horizon (float): horizon

        Returns:
            T.Optional[TrajectoryPlan]:
        """
        astar = AStar(     np.array([-1e9,-1e9]),
                           np.array([1e9,1e9]),
                           np.array([state.x,state.y]),
                           np.array([goal.x,goal.y]),
                           occupancy,
                           resolution) # How is horizon used? Low bound and high bound?
        if not astar.solve() or len(astar.path) < 4:
            return None
        self.t_prev = 0 # Is this all for resetting class variables?
        return astar.compute_smooth_plan()
 
if __name__ == "__main__":
    rclpy.init()            # initialize ROS client library
    node = Navigator()    # create the node instance
    rclpy.spin(node)        # call ROS2 default scheduler
    rclpy.shutdown()        # clean up after node exits