import sys
sys.path.append('/home/alexandre/Development/Spyro-main/spyro')
import spyro
import time

print("===================================================", flush = True)
frequency = 5.0
method = 'KMV'
degree = 3
experient_type = 'homogeneous'
minimum_mesh_velocity = 2.0
print("Running with "+method+ " and p =" + str(degree), flush = True)

start_time= time.time()
print("Starting initial method check", flush = True)

model = spyro.tools.create_model_for_grid_point_calculation(frequency, degree, method, minimum_mesh_velocity, experiment_type = experient_type, receiver_type = 'near')
#print("Model built at time "+str(time.time()-start_time), flush = True)
comm = spyro.utils.mpi_init(model)
#print("Comm built at time "+str(time.time()-start_time), flush = True)

p1 = spyro.tools.wave_solver(model, G =15, comm = comm)
#p1 = spyro.tools.p_filter(p1)
print("p1 finished at time "+str(time.time()-start_time), flush = True)
p2 = spyro.tools.wave_solver(model, G =10, comm = comm)
#p2 = spyro.tools.p_filter(p2)
print("p2 at time "+str(time.time()-start_time), flush = True)
error = spyro.tools.error_calc(p1,p2,model, comm=comm)
print("Error of  "+str(error)+" with G = 10", flush = True)


spyro.plots.plot_receiver_difference(model, p1, p2, 1, appear=True)

print('Fim', flush = True)