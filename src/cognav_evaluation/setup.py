import os

from setuptools import setup

package_name = "cognav_evaluation"

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
    description="Offline evaluation tools for CogNav runs.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={"console_scripts": [
        "timing_probe = cognav_evaluation.timing_probe:main",
        "cognav_metrics = cognav_evaluation.metrics:main",
        "cognav_episodes = cognav_evaluation.episodes:main",
        "cognav_compare = cognav_evaluation.compare:main",
        "cognav_footprint = cognav_evaluation.footprint:main",
        "cognav_coverage_curve = cognav_evaluation.coverage_curve:main",
        "cognav_arena = cognav_evaluation.arena:main",
    ]},
)
