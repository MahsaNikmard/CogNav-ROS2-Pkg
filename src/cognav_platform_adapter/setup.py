import os
from glob import glob

from setuptools import setup

package_name = "cognav_platform_adapter"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        (os.path.join("share", package_name), ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*launch.[pxy][yma]*")),
        (os.path.join("share", package_name, "config"), glob("config/*")),
        # Run by the habitat conda environment rather than as a ROS node.
        (os.path.join("share", package_name, "scripts"),
         ["cognav_platform_adapter/habitat_sim_server.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Mahsa Nikmard",
    maintainer_email="mahsa.nikmard@gssi.it",
    description="Platform adapters for CogNav: Habitat-Sim bridge and TurtleBot4.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "namespace_bridge = cognav_platform_adapter.namespace_bridge:main",
            "habitat_bridge = cognav_platform_adapter.habitat_bridge:main",
            "lab_bridge = cognav_platform_adapter.lab_bridge:main",
        ],
    },
)
