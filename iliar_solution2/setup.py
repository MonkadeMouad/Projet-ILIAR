from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'iliar_solution2'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(where='src'),  # Utilise find_packages pour trouver les packages dans 'src'
    package_dir={'': 'src'},              # Indique que le code source se trouve dans 'src'
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Inclure les fichiers de lancement
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.*')),
        # Inclure les fichiers de configuration
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Package iliar_solution2 for ROS 2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'teleop_command = iliar_solution2.teleop_command:main',
            'record_dataset=iliar_solution2.record_dataset:main',
            'autopilot=iliar_solution2.autopilot:main'

        ],
    },
)
