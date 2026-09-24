import os
from glob import glob

from setuptools import setup

package_name = "cognav_bt_runner"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*launch.[pxy][yma]*")),
        (os.path.join("share", package_name, "config"), glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Mahsa Nikmard",
    maintainer_email="mahsa.nikmard@gssi.it",
    description="Behaviour tree runner that ticks a CogNav mission once per perception frame.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={"console_scripts": ["bt_runner = cognav_bt_runner.bt_runner:main"]},
)
