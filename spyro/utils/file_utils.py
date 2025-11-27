import os

from time import strftime, gmtime


def mkdir_timestamp(basename):
    name = basename + "_" + strftime("%Y%m%d_%H%M%S", gmtime())
    if not os.path.exists(name):
        os.mkdir(name)
    return name
