"""The lidar's range and the costmap's marking range live in different files,
and the bug was the relationship between them.

With inf_is_valid true, Nav2's obstacle layer replaces an infinite reading
with range_max - epsilon and processes it as a normal return. If the marking
range reaches that far, every ray into open ground leaves a false obstacle
at the end of it. The costmap does not roll, so they accumulate: measured as
a band of wall ~8 m thick on an arc of radius ~41 m round the spawn point,
which made the whole east of the map unsearchable.

So obstacle_max_range must stay strictly below the lidar's range_max. These
tests fail if either number moves to break that again.
"""
import pathlib
import re

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
XACRO = ROOT / 'urdf_parrot' / 'parrot.gazebo.xacro'
SCOUT = ROOT / 'config' / 'nav2_params_parrot1.yaml'


def lidar_range_max():
    """The <max> inside the ray sensor's <range> block."""
    text = XACRO.read_text()
    ray = re.search(r'<ray>.*?</ray>', text, re.S)
    assert ray, 'no <ray> sensor found in the drone model'
    m = re.search(r'<range>.*?<max>\s*([\d.]+)\s*</max>', ray.group(0), re.S)
    assert m, 'the ray sensor has no range max'
    return float(m.group(1))


def obstacle_layers(path):
    """Every obstacle_layer scan source in a Nav2 params file."""
    doc = yaml.safe_load(path.read_text())
    found = []

    def walk(node, trail):
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            if key == 'obstacle_layer' and isinstance(value, dict):
                for src in str(value.get('observation_sources', '')).split():
                    if isinstance(value.get(src), dict):
                        found.append(('/'.join(trail + [key, src]), value[src]))
            walk(value, trail + [str(key)])

    walk(doc, [])
    return found


def test_the_model_really_does_declare_a_40_m_lidar():
    assert lidar_range_max() == pytest.approx(40.0), \
        'the lidar range changed - check obstacle_max_range still sits below it'


def test_there_are_obstacle_layers_to_check():
    layers = obstacle_layers(SCOUT)
    assert len(layers) >= 2, f'expected the global and local costmaps, got {layers}'


@pytest.mark.parametrize('where,source', obstacle_layers(SCOUT))
def test_marking_range_is_below_the_lidar_range(where, source):
    """The actual regression: an inf reading becomes a return at
    range_max - epsilon, so marking must stop short of range_max."""
    if not source.get('inf_is_valid'):
        pytest.skip('inf readings are discarded here, so no phantom to mark')
    limit = lidar_range_max()
    marking = float(source['obstacle_max_range'])
    assert marking < limit, (
        f'{where}: obstacle_max_range {marking} is not below the lidar\'s '
        f'range_max {limit}, so every ray into open ground will mark a '
        f'phantom obstacle at the end of it')


@pytest.mark.parametrize('where,source', obstacle_layers(SCOUT))
def test_clearing_reaches_at_least_as_far_as_marking(where, source):
    """Otherwise ground gets marked that nothing can ever clear."""
    marking = float(source['obstacle_max_range'])
    clearing = float(source['raytrace_max_range'])
    assert clearing >= marking, \
        f'{where}: raytrace {clearing} < obstacle {marking}'


@pytest.mark.parametrize('where,source', obstacle_layers(SCOUT))
def test_clearing_still_reaches_the_full_lidar_range(where, source):
    """The point of inf_is_valid is to clear open ground all the way out.
    Pulling marking back must not pull clearing back with it."""
    if not source.get('inf_is_valid'):
        pytest.skip('inf readings are discarded here')
    assert float(source['raytrace_max_range']) >= lidar_range_max(), \
        f'{where}: clearing no longer reaches the full sensor range'
