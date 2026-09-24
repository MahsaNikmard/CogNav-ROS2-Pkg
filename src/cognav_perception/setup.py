import os
from glob import glob

from setuptools import setup

package_name = "cognav_perception"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name, package_name + ".calibration"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        (os.path.join("share", package_name), ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*")),
        *[
            (os.path.join("share", package_name, os.path.dirname(path)), [path])
            for path in glob("intrinsic/**/*.npy", recursive=True)
        ],
        # Whichever DA2 checkpoints are present; the image downloads the metric one.
        *[
            (os.path.join("share", package_name, os.path.dirname(path)), [path])
            for path in glob("model_weights/*.pth")
        ],
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Mahsa Nikmard",
    maintainer_email="mahsa.nikmard@gssi.it",
    description="Monocular depth (Depth Anything V2) to polar distance vector for CogNav.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "cognav_perception = cognav_perception.cognav_perception:main",
        ],
    },
)
