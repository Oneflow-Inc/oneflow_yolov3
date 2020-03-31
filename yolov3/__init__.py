import os
import glob


def lib_path():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    libs = glob.glob(os.path.join(dir_path, "liboneflow_yolov3.so"))
    assert len(libs) > 0, "no .so found in {dir_path}"
    return libs[0]

