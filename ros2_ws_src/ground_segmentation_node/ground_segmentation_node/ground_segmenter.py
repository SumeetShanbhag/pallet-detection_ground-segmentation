import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
from ultralytics import YOLO

class GroundSegmentationNode(Node):
    def __init__(self):
        super().__init__('ground_segmenter')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # Match ZED2i publisher
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.subscriber = self.create_subscription(
            Image,
            '/robot1/zed2i/left/image_rect_color',
            self.listener_callback,
            qos_profile
        )

        self.publisher = self.create_publisher(Image, '/ground_segmentation', 10)
        self.br = CvBridge()

        model_path = os.path.expanduser('~/ros2_ws/src/ground_segmentation_node/models/best.pt')
        self.get_logger().info(f"Loading YOLOv8 segmentation model from: {model_path}")
        self.model = YOLO(model_path)

    def listener_callback(self, msg):
        try:
            self.get_logger().info("Image received — running ground segmentation inference...")
            frame = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            results = self.model.predict(frame, imgsz=640, conf=0.3)[0]

            annotated = frame.copy()

            if results.masks is not None and results.masks.data is not None:
                masks = results.masks.data.cpu().numpy()
                boxes = results.boxes
                classes = boxes.cls.cpu().numpy() if boxes is not None else [0] * len(masks)
                scores = boxes.conf.cpu().numpy() if boxes is not None else [1.0] * len(masks)

                for i, mask in enumerate(masks):
                    # Resize mask to match image size
                    mask_resized = cv2.resize((mask * 255).astype(np.uint8), (frame.shape[1], frame.shape[0]))
                    mask_rgb = cv2.merge([mask_resized] * 3)  # Convert to 3-channel

                    # Blend the mask with the image
                    color_mask = np.zeros_like(frame)
                    color_mask[:, :, 2] = mask_resized  # Red channel only
                    annotated = cv2.addWeighted(annotated, 1.0, color_mask, 0.4, 0)

                    # Add label
                    label = f"{self.model.names[int(classes[i])]} {scores[i]:.2f}"
                    ys, xs = np.where(mask_resized > 127)
                    if len(xs) > 0 and len(ys) > 0:
                        x, y = int(xs[0]), int(ys[0])
                        cv2.putText(annotated, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            out_msg = self.br.cv2_to_imgmsg(annotated, encoding='bgr8')
            out_msg.header = msg.header
            self.publisher.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f"Callback error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = GroundSegmentationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

