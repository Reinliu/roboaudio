from setuptools import setup

package_name = "spatial_audio_processor"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/test_vectors", [
            "test_vectors/generated_synthetic_sig_spliced.wav"
        ]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Your Name",
    maintainer_email="your@email.com",
    description="ROS2 multichannel audio -> DoA + Whisper ASR + diarization",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # ros2 run spatial_audio_processor main_ros
            "main_ros = spatial_audio_processor.main_ros:main",
        ],
    },
)
