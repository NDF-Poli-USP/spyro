from .time_integration_central_difference import central_difference
from .time_integration_central_difference import mixed_space_central_difference
from .time_integration_central_difference import central_difference_MMS


def time_integrator(Wave_object, source_ids=[0]):
    if Wave_object.source_type == "ricker":
        return time_integrator_ricker(Wave_object, source_ids=source_ids)
    elif Wave_object.source_type == "MMS":
        return time_integrator_mms(Wave_object, source_ids=source_ids)


def time_integrator_ricker(Wave_object, source_ids=[0]):
    if Wave_object.time_integrator == "central_difference":
        return central_difference(Wave_object, source_ids=source_ids)
    elif Wave_object.time_integrator == "mixed_space_central_difference":
        return mixed_space_central_difference(Wave_object, source_ids=source_ids)
    else:
        raise ValueError("The time integrator specified is not implemented yet")


def time_integrator_mms(Wave_object, source_ids=[0]):
    if Wave_object.time_integrator == "central_difference":
        return central_difference_MMS(Wave_object, source_ids=source_ids)
    else:
        raise ValueError("The time integrator specified is not implemented yet")
