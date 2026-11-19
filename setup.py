from setuptools import find_packages, setup

package_name = 'gelsight_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    package_data={
        'gelsight_vision.gsrobotics': ['models/*.pt', 'LICENSE'],
        'gelsight_vision.tools': [
            'models/*.pt', 'models/*.json',
            'palette_models/checkpoint/2000_Network.pth',
            'palette_models/checkpoint/config.json',
            'palette_models/README.md',
        ],
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
            'motion_collect_overlap = gelsight_vision.motion_collect_overlap:main',
            'pipeline_visuo_tactile = gelsight_vision.pipeline_visuo_tactile:main',
            'set_plate = gelsight_vision.set_plate:main',
            'touch_bead_stations = gelsight_vision.touch_bead_stations:main',
            'calib_spot_touch = gelsight_vision.calib_spot_touch:main',
            'calib_bead_sweep = gelsight_vision.calib_bead_sweep:main',
        ],
    },
)
