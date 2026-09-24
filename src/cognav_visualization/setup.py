import os

from setuptools import setup

package_name = "cognav_visualization"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        (os.path.join("share", package_name), ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Mahsa Nikmard",
    maintainer_email="mahsa.nikmard@gssi.it",
    description="RViz marker and message helpers for CogNav.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={"console_scripts": []},
)
