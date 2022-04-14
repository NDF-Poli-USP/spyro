from numbers import Real
import pytest
import numpy as np
import math
import spyro


@pytest.mark.mpi(min_size=5)
def test_forward_5shots():
    