# ~/ros2_ws/src/pallet_detection_node/launch/pallet_detection_launch.py

from launch import LaunchDescription
from launch.actions import ExecuteProcess

def generate_launch_description():
    return LaunchDescription([
        ExecuteProcess(
            cmd=['python3', '/home/sumeet/ros2_ws/src/pallet_detection_node/pallet_detection_node/pallet_detector.py'],
            output='screen'
        )
    ])

