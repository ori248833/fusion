import os
from pathlib import Path

from setuptools import find_packages, setup

package_name = 'cone_detector'
package_root = Path(__file__).resolve().parent
model_root = package_root.parents[1] / 'models'
model_files = [
    # colcon requires data_files source paths to be relative to setup.py.
    str(Path(os.path.relpath(path, start=package_root)))
    for path in model_root.glob('*.engine')
]

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/models', model_files),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='juziwei',
    maintainer_email='w062177@tongji.edu.cn',
    description='YOLO-based cone detection with LiDAR fusion',
    license='Apache License 2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'yolo_detector=cone_detector.yolo_detector:main',
        ],
    },
)
