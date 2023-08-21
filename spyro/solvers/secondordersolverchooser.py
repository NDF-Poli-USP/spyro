from .acousticNoPML import AcousticWaveNoPML
from .acousticPML import AcousticWavePML
from .mms_acoustic import AcousticWaveMMS


def get_abc_type(dictionary):
    if "BCs" in dictionary:
        return "PML"
    elif "absorving_boundary_conditions" in dictionary:
        return dictionary["absorving_boundary_conditions"]["damping_type"]
    else:
        return None


def AcousticWave(dictionary=None):
    if dictionary["acquisition"]["source_type"] == "MMS":
        return AcousticWaveMMS(dictionary=dictionary)

    has_abc = False
    if "BCs" in dictionary:
        has_abc = dictionary["BCs"]["status"]
    elif "absorving_boundary_conditions" in dictionary:
        has_abc = dictionary["absorving_boundary_conditions"]["status"]

    if has_abc:
        abc_type = get_abc_type(dictionary)
    else:
        abc_type = None

    if has_abc is False:
        return AcousticWaveNoPML(dictionary=dictionary)
    elif has_abc and abc_type == "PML":
        return AcousticWavePML(dictionary=dictionary)
    elif has_abc and abc_type == "HABC":
        raise NotImplementedError("HABC not implemented yet")
