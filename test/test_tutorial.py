"""The ground station's HELP tour: box placement and navigation."""

import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
from PyQt5.QtCore import QEvent, QRect, QSize, Qt
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import QApplication, QPushButton, QWidget

BOUNDS = QRect(0, 0, 1440, 900)
BOX = QSize(360, 240)


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


class TestPlaceCallout:
    def test_no_target_is_centred(self, ground_station):
        p = ground_station.place_callout(None, BOX, BOUNDS)
        # QRect.center() rounds down, so allow a pixel.
        assert abs(p.x() - 540) <= 1 and abs(p.y() - 330) <= 1

    def test_left_column_button_gets_box_on_its_right(self, ground_station):
        target = QRect(10, 380, 86, 58)
        p = ground_station.place_callout(target, BOX, BOUNDS)
        assert p.x() > target.right()

    def test_right_column_button_gets_box_on_its_left(self, ground_station):
        target = QRect(1340, 540, 98, 70)
        p = ground_station.place_callout(target, BOX, BOUNDS)
        assert p.x() + BOX.width() < target.left()

    def test_full_width_strip_gets_box_below(self, ground_station):
        target = QRect(4, 4, 1432, 56)
        p = ground_station.place_callout(target, BOX, BOUNDS)
        assert p.y() > target.bottom()

    @pytest.mark.parametrize('target', [
        QRect(1180, 826, 162, 70),    # E-STOP, bottom right
        QRect(478, 826, 98, 70),      # bottom row
        QRect(10, 10, 50, 50),        # top left corner
    ])
    def test_never_off_screen_or_over_target(self, ground_station, target):
        p = ground_station.place_callout(target, BOX, BOUNDS)
        box = QRect(p, BOX)
        assert BOUNDS.contains(box)
        assert not box.intersects(target)


@pytest.fixture
def tour(app, ground_station):
    parent = QWidget()
    parent.resize(1200, 800)
    button = QPushButton('X', parent)
    button.setGeometry(20, 300, 86, 58)
    parent.show()
    pages, closed = [], []
    steps = [
        ground_station.TutorialStep('One', 'first'),
        ground_station.TutorialStep('Two', 'second', lambda: [button], page=2),
        ground_station.TutorialStep('Three', 'third'),
    ]
    overlay = ground_station.TutorialOverlay(
        parent, steps, on_page=pages.append, on_close=lambda: closed.append(1))
    overlay.start()
    yield overlay, button, pages, closed
    parent.close()


def key(overlay, k):
    overlay.keyPressEvent(QKeyEvent(QEvent.KeyPress, k, Qt.NoModifier))


class TestTutorialOverlay:
    def test_starts_on_first_step_centred(self, tour):
        overlay, _, _, _ = tour
        assert overlay.isVisible() and overlay.index == 0
        assert overlay._hole is None
        assert not overlay.back_btn.isEnabled()

    def test_step_highlights_its_target_and_switches_page(self, tour):
        overlay, button, pages, _ = tour
        overlay.next()
        assert overlay._hole.contains(button.geometry())
        assert pages == [2]
        assert not overlay.box.geometry().intersects(button.geometry())

    def test_keys_navigate_and_escape_closes(self, tour):
        overlay, _, _, closed = tour
        key(overlay, Qt.Key_Right)
        key(overlay, Qt.Key_Right)
        assert overlay.index == 2 and overlay.next_btn.text().startswith('DONE')
        key(overlay, Qt.Key_Left)
        assert overlay.index == 1
        key(overlay, Qt.Key_Escape)
        assert not overlay.isVisible() and closed == [1]

    def test_next_on_last_step_finishes(self, tour):
        overlay, _, _, closed = tour
        for _ in range(3):
            overlay.next()
        assert not overlay.isVisible() and closed == [1]
