"""The courier's planned route on the ground station map."""

import time
from types import SimpleNamespace

import pytest
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path


@pytest.fixture
def station(ground_station):
    GS = ground_station.GroundStationNode
    node = SimpleNamespace(
        courier_path=[], courier_path_time=0.0, courier_xy=None,
        courier_plan_timeout_s=3.0,
        _odom_to_map_shift=lambda: (1.0, -2.0),
    )
    node.route = lambda: GS.courier_route(node)
    node.on_plan = lambda msg: GS._on_courier_plan(node, msg)
    node.on_odom = lambda msg: GS._on_courier_odometry(node, msg)
    return node


def path(*points):
    msg = Path()
    for x, y in points:
        pose = PoseStamped()
        pose.pose.position.x, pose.pose.position.y = float(x), float(y)
        msg.poses.append(pose)
    return msg


def odom(x, y):
    msg = Odometry()
    msg.pose.pose.position.x, msg.pose.pose.position.y = float(x), float(y)
    return msg


class TestCourierRoute:
    def test_nothing_until_a_plan_arrives(self, station):
        assert station.route() == []

    def test_plan_is_shifted_into_the_map_frame(self, station):
        station.on_plan(path((0, 0), (10, 0)))
        assert station.courier_path == [(1.0, -2.0), (11.0, -2.0)]

    def test_courier_position_is_shifted_too(self, station):
        station.on_odom(odom(3, 4))
        assert station.courier_xy == (4.0, 2.0)

    def test_route_starts_at_the_courier_and_drops_what_is_behind_it(self, station):
        station.on_plan(path((0, 0), (5, 0), (10, 0), (10, 5)))
        station.on_odom(odom(5.4, 0.3))      # just past the second point
        route = station.route()
        assert route[0] == station.courier_xy
        assert route[1:] == [(11.0, -2.0), (11.0, 3.0)]

    def test_ends_at_the_plan_goal(self, station):
        station.on_plan(path((0, 0), (10, 0), (10, 5)))
        station.on_odom(odom(0, 0))
        assert station.route()[-1] == (11.0, 3.0)

    def test_stale_plan_is_not_drawn(self, station):
        # Nav2 stops replanning once the courier arrives or is stopped.
        station.on_plan(path((0, 0), (10, 0)))
        station.courier_path_time = time.monotonic() - 3.5
        assert station.route() == []

    def test_empty_plan_draws_nothing(self, station):
        station.on_plan(path())
        assert station.route() == []

    def test_without_a_position_the_whole_plan_is_drawn(self, station):
        station.on_plan(path((0, 0), (10, 0)))
        assert station.route() == [(1.0, -2.0), (11.0, -2.0)]
