# ~/ros2_ws/src/pallet_detection_node/setup.py

from setuptools import setup

package_name = 'pallet_detection_node'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/pallet_detection_launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Sumeet',
    maintainer_email='you@example.com',
    description='Pallet detection ROS2 node using YOLOv8',
    license='MIT',
    entry_points={
        'console_scripts': [
            'pallet_detector = pallet_detection_node.pallet_detector:main',
        ],
    },
)

