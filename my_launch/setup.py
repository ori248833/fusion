import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'my_launch'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    package_data={package_name: ['frame_report.html', 'frame_metrics.js']},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'configs'), glob('configs/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ori',
    maintainer_email='ori15055966135@qq.com',
    description='Unified launch files package',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'frame_monitor = my_launch.frame_monitor:main',
            'frame_report = my_launch.frame_report:main',
        ],
    },
)
