from setuptools import setup

package_name = 'ground_segmentation_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Sumeet',
    maintainer_email='sumeet@example.com',
    description='Ground segmentation using YOLOv8 in ROS2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ground_segmenter = ground_segmentation_node.ground_segmenter:main',
        ],
    },
)

