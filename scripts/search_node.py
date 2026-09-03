#!/usr/bin/env python3 


"Fly a lawnmower search patter over a rectangle, using NAV2"

import math

import rclpy
from action_msgs.msg import GoalStatus
# Point is a plain (x, y, z) location. Markers are built out of lists of these.
from geometry_msgs.msg import Point, PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
# Duration is used to put a time limit on TF lookups so they can't block.
from rclpy.duration import Duration
from rclpy.node import Node
# ColorRGBA lets us colour each waypoint individually (done vs still to do).
from std_msgs.msg import ColorRGBA
# tf2_ros answers "where is the drone right now?" by reading the transform tree.
import tf2_ros
# Marker = one drawing in RViz. MarkerArray = several sent together.
from visualization_msgs.msg import Marker, MarkerArray

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
        self.declare_parameter('min_x', -4.0)
        self.declare_parameter('max_x', 4.0)
        self.declare_parameter('min_y', -4.0)
        self.declare_parameter('max_y', 4.0)
        self.declare_parameter('spacing', 0.8)

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

        # ------------------------------------------------------------------
        # VISUALISATION - everything below is only for RViz. It does not
        # affect how the drone flies; it just draws what is going on so the
        # search can be seen (and screenshotted) rather than only logged.
        # ------------------------------------------------------------------

        # The drone's own body frame, e.g. "parrot1_base_link". Asking TF for
        # map -> base_link tells us where the drone is on the map.
        self.base_frame = f'{robot_name}_base_link'

        # The buffer stores recent transforms; the listener fills it from the
        # /tf topic in the background. Both are needed for lookups to work.
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Publisher for the red breadcrumb trail of where the drone has been.
        self.trail_pub = self.create_publisher(Marker, 'search_trail', 10)

        # One Marker that we keep appending points to, rather than creating a
        # new one each time. Same ns + id every publish means RViz replaces
        # the old drawing instead of stacking copies on top of each other.
        self.trail = Marker()
        self.trail.header.frame_id = self.map_frame  # coordinates are map-frame
        self.trail.ns = 'flight_trail'
        self.trail.id = 0
        self.trail.type = Marker.POINTS   # dots; use LINE_STRIP for a line
        self.trail.action = Marker.ADD
        self.trail.scale.x = 0.2          # POINTS needs BOTH x and y set:
        self.trail.scale.y = 0.2          # they are the dot's width/height
        self.trail.color.r = 1.0          # red
        self.trail.color.g = 0.0
        self.trail.color.b = 0.0
        self.trail.color.a = 1.0          # opacity - defaults to 0 = INVISIBLE
        self.trail.pose.orientation.w = 1.0  # identity rotation (points are absolute)

        # Publisher for the search area, planned route and waypoint markers.
        # A MarkerArray because it carries several separate drawings at once.
        self.area_pub = self.create_publisher(MarkerArray, 'search_area', 10)

        # Two extra timers. Redrawing on a timer (instead of publishing once)
        # means the displays appear correctly no matter when you add them in
        # RViz, and lets the waypoint colours update as progress is made.
        self.trail_timer = self.create_timer(0.5, self.record_trail)
        self.area_timer = self.create_timer(1.0, self.publish_search_area)

    def tick(self):
        """Called every second; sends the next waypoint to Nav2 once it's ready and we're not already flying to one."""
        if self.busy: #goal is already in flight
            return
        if self.index>= len(self.points):
            return
        if not self.nav_client.server_is_ready(): #NAv2 sevrer not up yet
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

    # ----------------------------------------------------------------------
    # VISUALISATION METHODS
    # ----------------------------------------------------------------------

    def record_trail(self):
        """Add the drone's current position to the red trail.

        Runs twice a second. Each call asks TF where the drone is and appends
        that position to a growing list of dots, so afterwards you can see
        exactly where it actually flew (as opposed to where it was told to go).
        """
        try:
            # "Where is base_frame, as seen from map_frame, right now?"
            # rclpy.time.Time() with no argument means "the latest available".
            t = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.1),
            )
        except Exception:
            # TF lookups fail constantly at startup before transforms start
            # flowing. That is normal - just skip this tick and try again.
            return

        # transform.translation is the position part of the transform.
        p = Point()
        p.x = t.transform.translation.x
        p.y = t.transform.translation.y
        p.z = 0.0                       # draw flat on the ground
        self.trail.points.append(p)

        # Stop the list growing forever on a long mission (drops oldest first).
        if len(self.trail.points) > 2000:
            self.trail.points.pop(0)

        self.trail.header.stamp = self.get_clock().now().to_msg()
        self.trail_pub.publish(self.trail)

    def publish_search_area(self):
        """Draw the search rectangle, the planned route, and the waypoints.

        Rebuilt and republished every second so the waypoint colours update
        as the sweep progresses.
        """
        stamp = self.get_clock().now().to_msg()
        markers = MarkerArray()

        min_x = self.get_parameter('min_x').value
        max_x = self.get_parameter('max_x').value
        min_y = self.get_parameter('min_y').value
        max_y = self.get_parameter('max_y').value

        def new_marker(marker_id, name, marker_type, size):
            """Small helper - fills in the fields every marker needs."""
            m = Marker()
            m.header.frame_id = self.map_frame
            m.header.stamp = stamp
            m.ns = name                  # ns + id together identify a marker,
            m.id = marker_id             # so reusing them replaces the drawing
            m.type = marker_type
            m.action = Marker.ADD
            m.scale.x = size             # for lines this is the line width;
            m.scale.y = size             # for spheres it is the diameter
            m.scale.z = size
            m.pose.orientation.w = 1.0
            return m

        def point(x, y, z=0.0):
            """Build a Point from plain numbers (markers need float values)."""
            p = Point()
            p.x, p.y, p.z = float(x), float(y), float(z)
            return p

        # 1. GREEN OUTLINE of the search area. A LINE_STRIP joins its points
        #    in order, so we repeat the first corner at the end to close it.
        box = new_marker(0, 'search_area', Marker.LINE_STRIP, 0.15)
        box.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
        for x, y in [(min_x, min_y), (max_x, min_y), (max_x, max_y),
                     (min_x, max_y), (min_x, min_y)]:
            box.points.append(point(x, y))
        markers.markers.append(box)

        # 2. CYAN LINE showing the planned lawnmower route, in visiting order.
        #    Compare this against the red trail to see how closely the drone
        #    actually followed the intended sweep lines.
        route = new_marker(1, 'planned_route', Marker.LINE_STRIP, 0.08)
        route.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0)
        for x, y in self.points:
            route.points.append(point(x, y))
        markers.markers.append(route)

        # 3. WAYPOINT SPHERES. A SPHERE_LIST draws one sphere per point, and
        #    the parallel 'colors' list lets each one have its own colour -
        #    so completed waypoints turn green while pending ones stay yellow.
        dots = new_marker(2, 'waypoints', Marker.SPHERE_LIST, 0.4)
        dots.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)   # fallback colour
        for i, (x, y) in enumerate(self.points):
            dots.points.append(point(x, y, 0.2))   # lifted slightly off ground
            if i < self.index:
                dots.colors.append(ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0))  # done
            else:
                dots.colors.append(ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0))  # to do
        markers.markers.append(dots)

        self.area_pub.publish(markers)


def main():
    rclpy.init() 
    node = SearchNode()
    rclpy.spin(node)

if __name__ == '__main__':
        main()





                         