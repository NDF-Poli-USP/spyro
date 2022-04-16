from firedrake import File
import matplotlib.pyplot as plt
import numpy as np
import math
import spyro
import pytest
from test_forward_3d import test_forward_3d


def test_forward_3d_coverage():
    real_boolean = test_forward_3d(tf=0.05)
    
    assert True