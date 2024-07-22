from .time_integration_central_difference import central_difference

def time_integrator(Wave_object, source_id=0):
    return central_difference(Wave_object, source_id=source_id)
