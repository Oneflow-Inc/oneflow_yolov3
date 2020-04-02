#!/usr/bin/env python3
from setuptools import setup, find_packages

package_data = {"oneflow_yolov3": ["liboneflow_yolov3.so"]}
setup(
    name="oneflow_yolov3", packages=find_packages(), package_data=package_data,
)
