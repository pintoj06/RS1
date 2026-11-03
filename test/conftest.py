"""Shared helpers for the unit tests.

The nodes live in scripts/ and UI/ as plain executables rather than an
installed Python package, so the tests load them straight from their files.
Nothing here starts a node or needs the simulation: tests call the planning
and geometry functions directly, giving methods a small fake `self` holding
only what they read.

Run with ROS sourced (the modules import rclpy and the message types):

    source /opt/ros/humble/setup.bash
    python3 -m pytest test/
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from nav_msgs.msg import OccupancyGrid

PKG_ROOT = Path(__file__).resolve().parent.parent


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, PKG_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='session')
def search_node():
    return load_module('search_node', 'scripts/search_node.py')


@pytest.fixture(scope='session')
def courier_node():
    return load_module('courier_node', 'scripts/courier_node.py')


@pytest.fixture(scope='session')
def ground_station():
    return load_module('ground_station', 'UI/ground_station.py')


class FakeParams:
    """Stands in for Node.get_parameter: get_parameter(name).value."""

    def __init__(self, **values):
        self.values = values

    def __call__(self, name):
        return SimpleNamespace(value=self.values[name])


class FakeLogger:
    def __getattr__(self, _level):
        return lambda *args, **kwargs: None


def make_costmap(width, height, resolution=1.0, origin=(0.0, 0.0), fill=0):
    """An OccupancyGrid with every cell set to `fill`. Row-major, row 0 at
    origin y, the same layout Nav2 publishes."""
    grid = OccupancyGrid()
    grid.info.width = width
    grid.info.height = height
    grid.info.resolution = resolution
    grid.info.origin.position.x = float(origin[0])
    grid.info.origin.position.y = float(origin[1])
    grid.data = [fill] * (width * height)
    return grid


def set_cell(grid, col, row, value):
    data = list(grid.data)
    data[row * grid.info.width + col] = value
    grid.data = data
