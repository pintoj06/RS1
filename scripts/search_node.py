#!/usr/bin/env python3 


"Fly a lawnmower search patter over a rectangle, using NAV2"

import math

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node

def make_sweep(min_x, max_x, min_y, max_y, spacing):
    """Return a list fo (x,y) points that snake back and forther over the search area"""

    points= []
    y = min_y
    flip= False
    while y<=max_y + 0.001:
        if flip:
            points.append((max_x,y))
            points.append((min_x,y))
        else:
            points.append((min_x, y))
            points.append((max_x,y))
        flip = not flip
        y += spacing
    return points


class SearchNode(Node):
    def __init__(self):
        super().__init__('search_node')

        # Things that can be changed without editing the code
        self.declare_parameter('robot_name', 'parrot1')
        self.declare_parameter('min_x', -8.0)
        self.declare_parameter('max_x', 8.0)
        self.declare_parameter('min_y', -8.0)
        self.declare_parameter('max_y', 8.0)
        self.declare_parameter('spacing', 4.0)

        robot_name= self.get_parameter('robot_name').value
        self.map_frame= f'{robot_name}_map' # e.g. "parrot1_map", NOT "map"

        self.points= make_sweep(
            self.get_parameter('min_x').value,
            self.get_parameter('max_x').value,
            self.get_parameter('min_y').value,
            self.get_parameter('max_y').value,
            self.get_parameter('spacing').value           
        )

        self.index = 0
        self.busy= False

        self.nav_client= ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.timer = self.create_timer(1.0, self.tick) # runs once a seocnd

    def tick(self):
        if self.busy:
            return
        if self.index>= len(self.points):
            return
        if not self.nav_client.server_is_ready():
            self.get_logger().info('Waiting for Nav2 to be ready...')
            return
        self.send_goal(self.points[self.index])

    def send_goal (self, point):
        """Send one waypoint to Nav2."""
        x,y =point

        #Face the next waypoint, so the camera looks where we're going
        yaw= 0.0

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = self.map_frame 
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        # An angle as a quaternion. For a flat 2D turn, only z and w matter.
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self.busy = True
        self.get_logger().info (
            f'Waypoint {self.index +1}/{len(self.points)}: flying to ({x:.1f}, {y:.1f})'
        )

        # Stage 1: send it. goal_response() gets called wehen Nav2 answers.
        send_future= self.nav_client.send_goal_async(goal)
        send_future.add_done_callback(self.goal_response)

   

    def goal_response(self,future):
        """Stage 2: Nav2 has accepted or rejected the goal. """
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().warn('Nav2 rejected that goal. Skipping it.')
            self.next_waypoint()
            return
        
        #Stage 3: ask to be told when its finished flying.
        result_future= goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_finished)

    def goal_finished(self,future):
        """Stage 3: Nav2 has arrived, failed or given up"""       
        if future.result().status ==GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Arrived. ')
        else:
            self.get_logger().warn('Could not reach that one. Skipping it. ')
        self.next_waypoint()

    def next_waypoint(self):
        self.index +=1
        self.busy = False
        if self.index >=len(self.points):
            self.get_logger().info('Sweep complete. ')
        

def main():
    rclpy.init() 
    node = SearchNode()
    rclpy.spin(node)

if __name__ == '__main__':
        main()





                         