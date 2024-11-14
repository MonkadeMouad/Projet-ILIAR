from setuptools import setup
import os
from glob import glob

package_name = 'iliar_solution'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.xml')),
        ('share/' + package_name + '/config', glob('config/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Amrani',
    maintainer_email='hamza.amrani8@etu.univ-lorraine.fr',
    description='Description',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            
        ],
    },
)

