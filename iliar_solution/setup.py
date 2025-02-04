from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'iliar_solution'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(where='src'),  # Trouve les packages dans le dossier 'src'
    package_dir={'': 'src'},              # Déclare que 'src' contient le code source
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),   # Fichier pour indexation Ament
        ('share/' + package_name, ['package.xml']),  # Package.xml nécessaire pour ROS
        # Inclure les fichiers de lancement
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.*')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # Inclure les fichiers de configuration
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Package iliar_solution for ROS 2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Mappez les exécutables aux fonctions principales dans vos scripts
            'teleop_command = iliar_solution.teleop_command:main',
            'record_dataset = iliar_solution.record_dataset:main',
            'autopilot = iliar_solution.autopilot:main',
            'command_mux = iliar_solution.command_mux:main',
            'path_publisher = iliar_solution.path_publisher:main',
            'path_dilater = iliar_solution.path_dilater:main',
            'dist_to_path = iliar_solution.dist_to_path:main',
            'scorer = iliar_solution.scorer:main',
            'traveled_distance = iliar_solution.traveled_distance:main',
        ],
    },
)
