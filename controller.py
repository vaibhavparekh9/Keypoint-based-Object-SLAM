import numpy as np
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
import math

class RobustPoseController:
    def __init__(self, wheel_radius, wheel_base, scaler=1.0):
        self.controller = DifferentialController(
            name="robust_diff_controller",
            wheel_radius=wheel_radius * scaler,
            wheel_base=wheel_base * scaler,
            max_linear_speed=0.5,  # m/s
            max_angular_speed=1.0, # rad/s
        )
        # Controller gains
        self.linear_gain = 1.5
        self.angular_gain = 2.5
        self.distance_threshold = 0.1 # meters
        self.angle_threshold = 0.05    # radians (~3 degrees)

    def compute_control(self, robot_position, robot_orientation, goal_position):
        """
        Inputs:
            scaler: scaling factor (applied to distances)
            robot_position: (x, y, z)
            robot_orientation: (qx, qy, qz, qw) quaternion
            goal_position: (x_goal, y_goal, z_goal)
            goal_orientation: (qx_goal, qy_goal, qz_goal, qw_goal)
        """

        # Convert quaternions to yaw angles
        robot_yaw = self._quaternion_to_yaw(robot_orientation)
        # print("robot_yaw: " + str(robot_yaw))

        # Scale positions if needed
        robot_x = robot_position[0] 
        robot_y = robot_position[1] 
        goal_x = goal_position[0] 
        goal_y = goal_position[1] 

        dx = goal_x - robot_x
        dy = goal_y - robot_y
        distance = (dx**2 + dy**2) ** 0.5

        goal_theta = math.atan2(dy, dx)
        heading_error = self._normalize_angle(goal_theta - robot_yaw)

        # Proportional control
        linear_velocity = self.linear_gain * distance
        angular_velocity = self.angular_gain * heading_error

        # Clamp speeds
        linear_velocity = np.clip(linear_velocity, -self.controller.max_linear_speed, self.controller.max_linear_speed)
        angular_velocity = np.clip(angular_velocity, -self.controller.max_angular_speed, self.controller.max_angular_speed)

        # Stop if close enough
        if distance < self.distance_threshold and abs(heading_error) < self.angle_threshold:
            linear_velocity = 0.0
            angular_velocity = 0.0

        # Send to controller
        # print("linear_velocity: " + str(linear_velocity))
        # print("angular_velocity: " + str(angular_velocity))
        # print(heading_error)
        if abs(heading_error) > self.angle_threshold:
            # If the robot is facing the wrong way, set linear velocity to 0
            linear_velocity = 0.0
        else:
            angular_velocity = 0.0
    


        action = self.controller.forward([linear_velocity, angular_velocity])
        # print("action: " + str(action))
        return action

    def _quaternion_to_yaw(self, quat):
        """
        Converts quaternion (w, x, y, z) to yaw (rotation around Z).
        """
        w, x, y, z = quat
        # Yaw extraction formula from quaternion
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle