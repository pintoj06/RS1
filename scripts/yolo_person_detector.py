#!/usr/bin/env python3
"""Run a YOLO model on the drone's camera feed to detect people live.

Self-contained on purpose (one file, no other repo files touched): loads
your trained weights (e.g. a YOLO26n .pt file) via the `ultralytics`
package, subscribes to the camera topic, runs inference on each frame, and
publishes both an annotated debug image and structured detections.

Setup (one-time, not a repo change - installed into your user site-packages
so the rest of the ROS2/cv_bridge stack, which needs numpy 1.x, is
untouched):

    python3 -m pip install --user torch torchvision --index-url https://download.pytorch.org/whl/cpu
    python3 -m pip install --user ultralytics "opencv-python<5" "numpy<2"
    python3 -m pip uninstall -y opencv-python
    python3 -m pip install --user "opencv-python-headless<5" "numpy<2"

(That exact numpy/opencv-python pinning matters: ultralytics pulls in
numpy>=2 and a numpy2-only opencv-python by default, which silently breaks
cv_bridge - it's compiled against the system's numpy 1.21. Pin both back
down or every other script in this package that uses cv_bridge, including
the dataset-capture scripts, breaks too, in the same Python environment.

The swap to the headless build matters too: opencv-python bundles its own
Qt plugins, and they replace the system ones PyQt5 needs, so the ground
station dies with 'Could not load the Qt platform plugin "xcb"'. Nothing
here opens OpenCV windows, so headless loses nothing.

torchvision has to come from the CPU index in the first command too. If
ultralytics pulls it from PyPI instead, it is a CUDA build that does not
match CPU torch, and the first frame fails with "operator torchvision::nms
does not exist". Check with `python3 -m pip list --user | grep torch`: both
should end in +cpu.)

No GPU on this machine (checked earlier this session) - this runs on CPU.
A "nano" model is the right choice for that, but still expect single-digit
FPS, not real-time. The node naturally throttles itself to its own
inference speed: ROS's default single-threaded executor means the next
image callback can't start until the current (slow) one returns, and the
depth=1 image subscription just keeps the latest frame meanwhile, so frames
that arrive mid-inference are dropped rather than queued.

Target class note: `target_class_name` (default 'human') is matched as a
case-insensitive *substring* against the model's own class names
(model.names[int(cls)]), not a hardcoded class index or an exact-match
string - weights/best.pt (the default weights_path) has TWO classes,
{0: 'Human', 1: 'Human1'}, and the substring match catches both under the
one default instead of silently matching neither (an exact match against
'person' would match neither of those and fall back to accepting every
class, which is the wrong kind of wrong - it can't tell you it guessed).
If `target_class_name` still doesn't match any class the model actually
has, this logs a warning once and falls back to accepting every detection
that isn't an ignored class.

Ignored classes: the current weights have four classes, {0: 'Human',
1: 'Human1', 2: 'Kangaroo', 3: 'Wombat'}. Animals are warm, so the thermal
camera flags them the same as a person; the model seeing them is how the
system knows a heat signature is not one. Classes matching
`ignore_class_names` (default 'kangaroo,wombat') are boxed in amber on the
annotated image and published on `ignored_detections_topic`
(camera/animal_detections), never on `detections_topic`, so they can
never raise a contact. The ground station uses them only to skip or cut
short the search hold for an animal's heat signature.

Example (weights_path defaults to weights/best.pt in this repo, so it
doesn't need to be passed explicitly unless you're using different
weights):

    ros2 run 41068_ignition_bringup yolo_person_detector.py --ros-args \\
        -r __ns:=/parrot1 -p robot_name:=parrot1
"""

import os
from typing import Optional

# Must happen before numpy/opencv/torch are imported (the very next lines) -
# these only read their thread-pool size from the environment at import
# time, a runtime call like torch.set_num_threads() later doesn't reach
# OpenCV's or BLAS's own separate pools. setdefault so a value the user
# already exported is respected rather than overridden. Why this matters:
# verified live (2026-10-02) that without this, inference alone held 450%+
# CPU (4.5+ of 8 cores, no GPU on this VM) and pushed system load average
# over 12 - high enough that Gazebo's camera bridge and the ground station
# GUI stopped getting enough CPU time to keep their own topics flowing,
# even though the simulation itself was fine. 2 leaves headroom for those.
for _env_var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_env_var, '2')

import cv2
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose


class YoloPersonDetector(Node):

    def __init__(self):
        super().__init__('yolo_person_detector')

        self.declare_parameter('robot_name', 'parrot1')
        self.declare_parameter('image_topic', 'camera/image')
        # __file__-relative would resolve correctly for `python3
        # scripts/yolo_person_detector.py` run from source, but `ros2 run`
        # executes the file colcon *installed* to lib/41068_ignition_bringup/,
        # where a __file__-relative path lands in the wrong place entirely
        # (verified live: resolved to .../install/41068_ignition_bringup/lib/
        # weights/best.pt, which doesn't exist - lib/, not share/). Using the
        # ament package index instead, same as multi_entity_dataset_capture.py
        # already does for model.sdf, works under both invocations.
        self.declare_parameter(
            'weights_path',
            os.path.join(
                get_package_share_directory('41068_ignition_bringup'),
                'weights', 'best.pt',
            ),
        )
        # Substring match (case-insensitive), not exact: this specific
        # model (weights/best.pt) has TWO classes, {0: 'Human', 1:
        # 'Human1'} - not 'person'. An exact match against 'person' would
        # match neither and silently fall back to accepting every class;
        # 'human' as a substring correctly catches both.
        self.declare_parameter('target_class_name', 'human')
        # Classes the model knows but that must never raise a contact. They
        # are still boxed on the annotated image (in a different colour) so
        # the operator can see what the camera is looking at, and published
        # on their own topic, never on detections_topic, which is the only
        # one the ground station raises contacts from. Comma-separated,
        # case-insensitive substring match, same as target_class_name.
        self.declare_parameter('ignore_class_names', 'kangaroo,wombat')
        self.declare_parameter('ignored_detections_topic', 'camera/animal_detections')
        self.declare_parameter('confidence_threshold', 0.5)
        # A human only counts once the box is fully in the frame: this far in
        # from every edge, as a fraction of the image size. A box touching
        # the edge is a person half in view, which makes for evidence the
        # operator can't judge, so it is drawn as "entering" and not
        # published. The thermal camera has the same 80 deg view as the RGB
        # one, so heat at the very edge can start the thermal check while
        # the person is still "entering" here.
        self.declare_parameter('edge_margin_frac', 0.05)
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('publish_annotated', True)
        self.declare_parameter('annotated_topic', 'camera/person_detections_image')
        self.declare_parameter('detections_topic', 'camera/person_detections')
        # Humans still at the frame edge ("entering"), on their own topic so
        # they can never raise a contact. The search node reads them to fly
        # over and get the person fully in view when the sweep would
        # otherwise pass them by - see _on_edge_detections in search_node.py.
        self.declare_parameter('edge_detections_topic', 'camera/edge_detections')
        # PyTorch defaults to using every CPU thread on the machine for each
        # inference call. On this shared VM (no GPU - checked earlier this
        # session) that starves everything else: verified live, with no
        # cap, this one node alone held 456% CPU (4.5+ of 8 cores) and load
        # average hit 12.4, which was enough to stop Gazebo's camera bridge
        # and the ground station GUI from getting CPU time to keep up, even
        # though the simulation and the model itself were both fine. 2
        # leaves meaningful headroom for Gazebo, the ROS bridges, and any
        # GUI running alongside this node.
        self.declare_parameter('cpu_threads', 2)

        self.robot_name = str(self.get_parameter('robot_name').value)
        weights_path = os.path.expanduser(str(self.get_parameter('weights_path').value))
        self.target_class_name = str(self.get_parameter('target_class_name').value).lower()
        self.confidence_threshold = float(self.get_parameter('confidence_threshold').value)
        self.edge_margin_frac = float(self.get_parameter('edge_margin_frac').value)
        self.device = str(self.get_parameter('device').value)
        self.publish_annotated = bool(self.get_parameter('publish_annotated').value)
        cpu_threads = int(self.get_parameter('cpu_threads').value)

        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f'weights_path {weights_path!r} does not exist. Pass your actual '
                f'.pt file with -p weights_path:=/path/to/weights.pt'
            )

        # Imported here, not at module level, so --help / argument errors on
        # a bad weights_path surface immediately rather than after the
        # (slow) torch/ultralytics import.
        import torch
        from ultralytics import YOLO
        if cpu_threads > 0:
            torch.set_num_threads(cpu_threads)
            cv2.setNumThreads(cpu_threads)
        self.get_logger().info(
            f'Loading YOLO weights from {weights_path} '
            f'(device={self.device}, cpu_threads={cpu_threads})...'
        )
        self.model = YOLO(weights_path)
        self.get_logger().info(f'Loaded. Model classes: {self.model.names}')

        ignore_names = [
            n.strip().lower()
            for n in str(self.get_parameter('ignore_class_names').value).split(',')
            if n.strip()
        ]
        self.target_class_ids, self.ignored_class_ids = classify_class_ids(
            self.model.names, self.target_class_name, ignore_names)
        if not self.target_class_ids:
            self.get_logger().warning(
                f'No class name containing "{self.target_class_name}" in this model '
                f'(classes: {self.model.names}) - accepting detections of any class '
                f'except the ignored ones instead.'
            )

        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image, self.get_parameter('image_topic').value, self._image_callback, 1
        )
        self.detections_pub = self.create_publisher(
            Detection2DArray, self.get_parameter('detections_topic').value, 10
        )
        self.ignored_pub = self.create_publisher(
            Detection2DArray, self.get_parameter('ignored_detections_topic').value, 10
        )
        self.edge_pub = self.create_publisher(
            Detection2DArray, self.get_parameter('edge_detections_topic').value, 10
        )
        self.annotated_pub = None
        if self.publish_annotated:
            self.annotated_pub = self.create_publisher(
                Image, self.get_parameter('annotated_topic').value, 1
            )

        self.get_logger().info(
            f'yolo_person_detector started for "{self.robot_name}": '
            f'watching {self.image_sub.topic_name}, target class name contains '
            f'"{self.target_class_name}" (matched ids={sorted(self.target_class_ids)}), '
            f'ignoring ids={sorted(self.ignored_class_ids)} (shown, never a contact), '
            f'confidence >= {self.confidence_threshold}.'
        )

    def _to_array(self, header, matches) -> Detection2DArray:
        detections = Detection2DArray()
        detections.header = header
        for cls_id, conf, (x1, y1, x2, y2) in matches:
            det = Detection2D()
            det.header = header
            det.bbox.center.position.x = (x1 + x2) / 2.0
            det.bbox.center.position.y = (y1 + y2) / 2.0
            det.bbox.size_x = x2 - x1
            det.bbox.size_y = y2 - y1
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(self.model.names[cls_id])
            hypothesis.hypothesis.score = conf
            det.results.append(hypothesis)
            detections.detections.append(det)
        return detections

    def _image_callback(self, msg: Image) -> None:
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        results = self.model(frame, device=self.device, verbose=False)[0]

        boxes = [
            (int(box.cls[0]), float(box.conf[0]), box.xyxy[0].tolist())
            for box in results.boxes
        ]
        matches, ignored = split_detections(
            boxes, self.confidence_threshold,
            self.target_class_ids, self.ignored_class_ids)
        height, width = frame.shape[:2]
        entering = [m for m in matches
                    if not fully_in_frame(m[2], width, height, self.edge_margin_frac)]
        matches = [m for m in matches if m not in entering]

        # Always published, empty or not: the ground station treats a silent
        # topic as the detector being down.
        self.detections_pub.publish(self._to_array(msg.header, matches))
        self.ignored_pub.publish(self._to_array(msg.header, ignored))
        if entering:
            self.edge_pub.publish(self._to_array(msg.header, entering))
        if matches:
            conf_str = ', '.join(f'{c:.2f}' for _, c, _ in matches)
            self.get_logger().info(f'{len(matches)} detection(s), confidence: {conf_str}')
        if ignored:
            seen = ', '.join(f'{self.model.names[i]} {c:.2f}' for i, c, _ in ignored)
            self.get_logger().info(f'Ignored (not a contact): {seen}')

        if self.annotated_pub is not None:
            annotated = frame.copy()
            # Humans green; ignored classes amber and humans still at the
            # frame edge yellow, each labelled, so the operator can tell at
            # a glance which boxes count.
            for colour, suffix, group in (((0, 191, 255), ' (ignored)', ignored),
                                          ((0, 255, 255), ' (entering)', entering),
                                          ((0, 255, 0), '', matches)):
                for cls_id, conf, (x1, y1, x2, y2) in group:
                    p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                    cv2.rectangle(annotated, p1, p2, colour, 2)
                    label = f'{self.model.names[cls_id]} {conf:.2f}{suffix}'
                    cv2.putText(
                        annotated, label, (p1[0], max(p1[1] - 6, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA,
                    )
            out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            out_msg.header = msg.header
            self.annotated_pub.publish(out_msg)


def classify_class_ids(names, target_name, ignore_names):
    """Split the model's class ids into (target ids, ignored ids).

    Both are case-insensitive substring matches on the class name. A class
    that matches an ignore name is never a target, even if it also matches
    target_name, so an ignored class can never raise a contact.
    """
    ignored = {
        int(i) for i, n in names.items()
        if any(ig in str(n).lower() for ig in ignore_names)
    }
    target = {
        int(i) for i, n in names.items()
        if target_name in str(n).lower() and int(i) not in ignored
    }
    return target, ignored


def fully_in_frame(xyxy, width, height, margin_frac):
    """True if the box is at least margin_frac of the image size in from
    every edge, i.e. the whole object is in view, not cut off by the frame."""
    x1, y1, x2, y2 = xyxy
    mx, my = margin_frac * width, margin_frac * height
    return x1 >= mx and y1 >= my and x2 <= width - mx and y2 <= height - my


def split_detections(boxes, threshold, target_ids, ignored_ids):
    """Sort (cls_id, conf, xyxy) boxes into (matches, ignored).

    Below-threshold boxes are dropped from both. With no target ids (the
    model has no class matching target_class_name) every class that is not
    ignored counts as a match. Classes that are neither are dropped.
    """
    matches, ignored = [], []
    for cls_id, conf, xyxy in boxes:
        if conf < threshold:
            continue
        if cls_id in ignored_ids:
            ignored.append((cls_id, conf, xyxy))
        elif not target_ids or cls_id in target_ids:
            matches.append((cls_id, conf, xyxy))
    return matches, ignored


def main(args=None):
    rclpy.init(args=args)
    node = YoloPersonDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
