#!/usr/bin/env python3
from setuptools import setup

package_data = {"oneflow_yolov3": ["liboneflow_yolov3.so"]}
setup(name="oneflow_yolov3", package_data=package_data)
