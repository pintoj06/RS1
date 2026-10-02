#!/usr/bin/env python3
"""Run a YOLO model on the drone's camera feed to detect people live.

Self-contained on purpose (one file, no other repo files touched): loads
your trained weights (e.g. a YOLO26n .pt file) via the `ultralytics`
package, subscribes to the camera topic, runs inference on each frame, and
publishes both an annotated debug image and structured detections.

Setup (one-time, not a repo change - installed into your user site-packages
so the rest of the ROS2/cv_bridge stack, which needs numpy 1.x, is
untouched):

    python3 -m pip install --user torch --index-url https://download.pytorch.org/whl/cpu
    python3 -m pip install --user ultralytics "opencv-python<5" "numpy<2"

(That exact numpy/opencv-python pinning matters: ultralytics pulls in
numpy>=2 and a numpy2-only opencv-python by default, which silently breaks
cv_bridge - it's compiled against the system's numpy 1.21. Pin both back
down or every other script in this package that uses cv_bridge, including
the dataset-capture scripts, breaks too, in the same Python environment.)

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
regardless of class.

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
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('publish_annotated', True)
        self.declare_parameter('annotated_topic', 'camera/person_detections_image')
        self.declare_parameter('detections_topic', 'camera/person_detections')
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

        self.target_class_ids = {
            int(class_id) for class_id, name in self.model.names.items()
            if self.target_class_name in str(name).lower()
        }
        if not self.target_class_ids:
            self.get_logger().warning(
                f'No class name containing "{self.target_class_name}" in this model '
                f'(classes: {self.model.names}) - accepting detections of any class instead.'
            )

        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image, self.get_parameter('image_topic').value, self._image_callback, 1
        )
        self.detections_pub = self.create_publisher(
            Detection2DArray, self.get_parameter('detections_topic').value, 10
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
            f'confidence >= {self.confidence_threshold}.'
        )

    def _image_callback(self, msg: Image) -> None:
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        results = self.model(frame, device=self.device, verbose=False)[0]

        detections = Detection2DArray()
        detections.header = msg.header

        matches = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < self.confidence_threshold:
                continue
            if self.target_class_ids and cls_id not in self.target_class_ids:
                continue
            matches.append((cls_id, conf, box.xyxy[0].tolist()))

        for cls_id, conf, (x1, y1, x2, y2) in matches:
            det = Detection2D()
            det.header = msg.header
            det.bbox.center.position.x = (x1 + x2) / 2.0
            det.bbox.center.position.y = (y1 + y2) / 2.0
            det.bbox.size_x = x2 - x1
            det.bbox.size_y = y2 - y1
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(self.model.names[cls_id])
            hypothesis.hypothesis.score = conf
            det.results.append(hypothesis)
            detections.detections.append(det)

        self.detections_pub.publish(detections)
        if matches:
            conf_str = ', '.join(f'{c:.2f}' for _, c, _ in matches)
            self.get_logger().info(f'{len(matches)} detection(s), confidence: {conf_str}')

        if self.annotated_pub is not None:
            annotated = frame.copy()
            for cls_id, conf, (x1, y1, x2, y2) in matches:
                p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                cv2.rectangle(annotated, p1, p2, (0, 255, 0), 2)
                label = f'{self.model.names[cls_id]} {conf:.2f}'
                cv2.putText(
                    annotated, label, (p1[0], max(p1[1] - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
                )
            out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            out_msg.header = msg.header
            self.annotated_pub.publish(out_msg)


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
