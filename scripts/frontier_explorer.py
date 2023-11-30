#!/usr/bin/env python3
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
from asl_tb3_lib.tf_utils import quaternion_to_yaw
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from scipy.interpolate import splev, splrep
import numpy as np
import typing as T
import rclpy
from rclpy import Node
import matplotlib.pyplot as plt
from asl_tb3_lib.grids import StochOccupancyGrid2D
from std_msgs.msg import Bool
from nav_msgs.msg import OccupancyGrid, Path
from scipy.signal import convolve2d

class FrontierExplorer(Node):
    def __init__(self) -> None:
        super().__init__("frontier_explorer")
        self.occupancy: T.Optional[StochOccupancyGrid2D] = None
        self.state: T.Optional[TurtleBotState] = None
        self.nav_success_sub = self.create_subscription(Bool,"/nav_success", self.select_goal,10)
        self.state_sub = self.create_subscription(TurtleBotState, "/state", self.state_callback, 10)
        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self.map_callback, 10)
        self.cmd_nav_pub = self.create_publisher(TurtleBotState, "/cmd_nav", 10)
    def select_goal(self,nav_success):
        if nav_success.data: # try msg.data instead of nav_success
            frontier_states = self.explore()
            curr_pos = np.array([self.state.x,self.state.y])
            dist = np.linalg.norm(curr_pos - frontier_states,axis=1)
            explore_pos = frontier_states[np.argmin(dist)]

            explore_state = TurtleBotState()
            explore_state.x = explore_pos[0]
            explore_state.y = explore_pos[1]
            explore_state.theta = self.state.theta
            self.cmd_nav_pub.publish(explore_state)

    def explore(self):
        """ returns potential states to explore
        Args:
            occupancy (StochasticOccupancyGrid2D): Represents the known, unknown, occupied, and unoccupied states. See class in first section of notebook.

        Returns:
            frontier_states (np.ndarray): state-vectors in (x, y) coordinates of potential states to explore. Shape is (N, 2), where N is the number of possible states to explore.

        HINTS:
        - Function `convolve2d` may be helpful in producing the number of unknown, and number of occupied states in a window of a specified cell
        - Note the distinction between physical states and grid cells. Most operations can be done on grid cells, and converted to physical states at the end of the function with `occupancy.grid2state()`
        """

        window_size = 13    # defines the window side-length for neighborhood of cells to consider for heuristics
        ########################### Code starts here ###########################
        # Create a window for convolution to count unknown, occupied, and unoccupied cells
        window = np.ones((window_size, window_size), dtype=int)
        
        # Perform 2D convolution to count unknown, occupied, and unoccupied cells in the neighborhood
        unknown_count = convolve2d(self.occupancy.probs == -1, window, mode='same', boundary='fill', fillvalue=0)
        occupied_count = convolve2d(self.occupancy.probs >= 0.5, window, mode='same', boundary='fill', fillvalue=0)
        unoccupied_count = convolve2d(self.occupancy.probs < 0.5, window, mode='same', boundary='fill', fillvalue=0)

        unknown_threshold = 0.2 * np.sum(window)
        occupied_threshold = 0
        unoccupied_threshold = 0.3 * np.sum(window)

        frontier_mask = (unknown_count >= unknown_threshold) & (occupied_count == occupied_threshold) & (unoccupied_count-unknown_count >= unoccupied_threshold)
        
        # Get indices of the potential frontier cells
        frontier_indices = np.argwhere(frontier_mask)

        frontier_states = self.occupancy.grid2state(frontier_indices)
        frontier_states = np.array([frontier_states.T[1],frontier_states.T[0]]).T
        # dist = np.linalg.norm(self.state - frontier_states,axis=1)
        
        # print(np.min(dist))
        ########################### Code ends here ###########################
        return frontier_states#, np.argmin(dist)
    def state_callback(self,msg: TurtleBotState) -> None:
        self.state = msg
    def map_callback(self, msg: OccupancyGrid) -> None:
        """ Callback triggered when the map is updated

        Args:
            msg (OccupancyGrid): updated map message
        """
        self.occupancy = StochOccupancyGrid2D(
            resolution=msg.info.resolution,
            size_xy=np.array([msg.info.width, msg.info.height]),
            origin_xy=np.array([msg.info.origin.position.x, msg.info.origin.position.y]),
            window_size=9,
            probs=msg.data,
        )

if __name__ == "__main__":
    rclpy.init()            # initialize ROS client library
    node = FrontierExplorer()    # create the node instance
    rclpy.spin(node)        # call ROS2 default scheduler
    rclpy.shutdown()        # clean up after node exits
