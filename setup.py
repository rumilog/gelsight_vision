from setuptools import find_packages, setup

package_name = 'gelsight_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    package_data={
        'gelsight_vision.gsrobotics': ['models/*.pt', 'LICENSE'],
        'gelsight_vision.tools': ['models/*.pt', 'models/*.json'],
    },
    include_package_data=True,
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ppak',
    maintainer_email='ppak10@gmail.com',
    description='Manages Gelsight and vision from D555 Realsense camera',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'motion_collect_sample = gelsight_vision.motion_collect_sample:main',
            'pipeline_visuo_tactile = gelsight_vision.pipeline_visuo_tactile:main',
            'set_plate = gelsight_vision.set_plate:main',
        ],
    },
)
