# ~/ros2_ws/src/pallet_detection_node/pallet_detection_node/pallet_detector.py

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np
import os

class PalletDetectorNode(Node):
    def __init__(self):
        super().__init__('pallet_detector')

        # qos_profile = QoSProfile(depth=10)
        # qos_profile.reliability = ReliabilityPolicy.RELIABLE

        # qos_profile = QoSProfile(
        #     reliability= ReliabilityPolicy.RELIABLE, 
        #     history=QoSHistoryPolicy.KEEP_LAST,
        #     depth=10
        # )
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  
            depth=10
        )

        self.subscriber = self.create_subscription(
            Image,
            '/robot1/zed2i/left/image_rect_color',
            self.listener_callback,
            qos_profile
        )

        self.publisher = self.create_publisher(Image, '/pallet_detection', 10)
        self.br = CvBridge()

        model_path = os.path.expanduser('~/ros2_ws/src/pallet_detection_node/models/best.pt')
        self.get_logger().info(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)

    def listener_callback(self, msg):
        try:
            self.get_logger().info("Image received — running inference...")
            frame = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            results = self.model(frame)[0]

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{self.model.names[cls]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            out_msg = self.br.cv2_to_imgmsg(frame, encoding='bgr8')
            out_msg.header = msg.header
            self.publisher.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = PalletDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


