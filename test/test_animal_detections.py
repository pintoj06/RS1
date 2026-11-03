"""Kangaroos and wombats: shown on screen, never a contact, never a hold.

The detector sorts the model's classes into humans (published on the topic
contacts are raised from) and ignored animals (their own topic). The ground
station uses the animal topic only to skip or end the thermal check's hold.
"""

import time
from types import SimpleNamespace

import pytest

from conftest import FakeLogger, load_module

NAMES = {0: 'Human', 1: 'Human1', 2: 'Kangaroo', 3: 'Wombat'}
IGNORE = ['kangaroo', 'wombat']
BOX = [10.0, 10.0, 50.0, 80.0]


@pytest.fixture(scope='session')
def detector():
    return load_module('yolo_person_detector', 'scripts/yolo_person_detector.py')


class TestClassifyClassIds:
    def test_new_model(self, detector):
        target, ignored = detector.classify_class_ids(NAMES, 'human', IGNORE)
        assert target == {0, 1}
        assert ignored == {2, 3}

    def test_ignore_beats_target(self, detector):
        # A class matching both lists must never become a target.
        target, ignored = detector.classify_class_ids(
            {0: 'Human', 1: 'HumanWombat'}, 'human', IGNORE)
        assert target == {0}
        assert ignored == {1}

    def test_old_model_has_nothing_to_ignore(self, detector):
        target, ignored = detector.classify_class_ids(
            {0: 'Human', 1: 'Human1'}, 'human', IGNORE)
        assert target == {0, 1}
        assert ignored == set()


class TestSplitDetections:
    def split(self, detector, boxes, target={0, 1}, ignored={2, 3}):
        return detector.split_detections(boxes, 0.5, target, ignored)

    def test_animals_never_in_matches(self, detector):
        matches, ignored = self.split(detector, [(2, 0.9, BOX), (3, 0.8, BOX)])
        assert matches == []
        assert [c for c, _, _ in ignored] == [2, 3]

    def test_humans_and_animals_together(self, detector):
        matches, ignored = self.split(detector, [(0, 0.7, BOX), (2, 0.9, BOX)])
        assert [c for c, _, _ in matches] == [0]
        assert [c for c, _, _ in ignored] == [2]

    def test_threshold_applies_to_both(self, detector):
        matches, ignored = self.split(detector, [(0, 0.4, BOX), (2, 0.3, BOX)])
        assert matches == [] and ignored == []

    def test_fallback_still_excludes_animals(self, detector):
        # No class matched target_class_name: accept everything else, but an
        # animal must still not become a match.
        matches, ignored = self.split(
            detector, [(0, 0.9, BOX), (2, 0.9, BOX)], target=set())
        assert [c for c, _, _ in matches] == [0]
        assert [c for c, _, _ in ignored] == [2]


class TestFullyInFrame:
    """A human only counts once the whole box is clear of the frame edge
    (5% of 720x480 = 36 px sideways, 24 px vertically)."""

    def check(self, detector, box):
        return detector.fully_in_frame(box, 720, 480, 0.05)

    def test_centred_person(self, detector):
        assert self.check(detector, [300, 200, 360, 300])

    def test_cut_off_at_each_edge(self, detector):
        assert not self.check(detector, [0, 200, 60, 300])       # left
        assert not self.check(detector, [660, 200, 720, 300])    # right
        assert not self.check(detector, [300, 0, 360, 100])      # top
        assert not self.check(detector, [300, 400, 360, 480])    # bottom

    def test_margin_boundary(self, detector):
        assert self.check(detector, [36, 24, 684, 456])
        assert not self.check(detector, [35, 24, 684, 456])

    def test_zero_margin_only_rejects_off_frame(self, detector):
        assert detector.fully_in_frame([0, 0, 720, 480], 720, 480, 0.0)

    def test_thermal_view_is_inside_the_margin(self):
        # Thermal (60 deg) and RGB (80 deg) share a mount, so the thermal
        # view's edge sits this far in from the RGB edge. It must be more
        # than the margin, or a person who triggered the thermal check could
        # be stuck as "entering" for the whole hold.
        import math
        inset = (1 - math.tan(math.radians(30)) / math.tan(0.7)) / 2
        assert inset > 0.05


def node_with(ground_station, rgb_hit=None, animal_hit=None):
    node = SimpleNamespace(rgb_hit=rgb_hit, animal_hit=animal_hit, rgb_fresh_s=1.5,
                           rgb_hit_xy=None, rgb_hit_track=None)
    cls = ground_station.GroundStationNode
    node.fresh_rgb_hit = lambda: cls.fresh_rgb_hit(node)
    node.fresh_animal_only = lambda: cls.fresh_animal_only(node)
    return node


class TestFreshAnimalOnly:
    def test_recent_animal(self, ground_station):
        node = node_with(ground_station, animal_hit=('Kangaroo', time.monotonic()))
        assert node.fresh_animal_only() == 'Kangaroo'

    def test_stale_animal(self, ground_station):
        node = node_with(ground_station, animal_hit=('Wombat', time.monotonic() - 5))
        assert node.fresh_animal_only() is None

    def test_human_wins(self, ground_station):
        now = time.monotonic()
        node = node_with(ground_station, rgb_hit=(0.8, now),
                         animal_hit=('Kangaroo', now))
        assert node.fresh_animal_only() is None


class FakeVisPage:
    frozen = None


def window_with(ground_station, node, heat=True, checking=False):
    node.thermal_hit = (310.0, 1200) if heat else None
    node.person_min_pixels = 600
    node.rgb_check_s = 6.0
    node.drone_xy = (5.0, 5.0)
    node.display_frame = node.thermal_frame = None
    node.get_logger = FakeLogger
    holds, raised = [], []
    node.set_hold = holds.append
    win = SimpleNamespace(
        node=node, contacts=[], rejected_spots=[], CONTACT_RADIUS=5.0,
        _no_check_until=0.0, hold_for_confirmed=False,
        _check_until=(time.monotonic() + 6.0) if checking else None,
        _check_xy=(5.0, 5.0), _check_heat_conf=0.5, _check_ir_frame=None,
        vis_page=FakeVisPage(), ir_page=FakeVisPage(),
    )
    cls = ground_station.MainWindow
    win._near = lambda x, y, spots: cls._near(win, x, y, spots)
    win._update_hold = lambda: cls._update_hold(win)
    win._raise_contact = lambda *a, **k: raised.append((a, k))
    return win, holds, raised


class TestCheckDetections:
    def run(self, ground_station, win):
        ground_station.MainWindow._check_detections(win)

    def test_animal_heat_does_not_hold(self, ground_station):
        node = node_with(ground_station, animal_hit=('Kangaroo', time.monotonic()))
        win, holds, raised = window_with(ground_station, node)
        self.run(ground_station, win)
        assert win._check_until is None
        assert True not in holds
        assert raised == []
        assert win.rejected_spots == [(5.0, 5.0)]

    def test_animal_ends_running_check(self, ground_station):
        node = node_with(ground_station, animal_hit=('Wombat', time.monotonic()))
        win, holds, raised = window_with(ground_station, node, checking=True)
        self.run(ground_station, win)
        assert win._check_until is None
        assert holds == [False]
        assert raised == []

    def test_unknown_heat_still_holds(self, ground_station):
        node = node_with(ground_station)
        win, holds, raised = window_with(ground_station, node)
        self.run(ground_station, win)
        assert win._check_until is not None
        assert holds == [True]

    def test_human_beside_animal_raises_contact(self, ground_station):
        now = time.monotonic()
        node = node_with(ground_station, rgb_hit=(0.8, now),
                         animal_hit=('Kangaroo', now))
        win, holds, raised = window_with(ground_station, node)
        self.run(ground_station, win)
        assert len(raised) == 1
