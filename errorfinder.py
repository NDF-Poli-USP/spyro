import sys
sys.path.append('/home/alexandre/Development/Spyro-3workingBranch')
from datetime import datetime
import spyro
import numpy as np
import matplotlib.pyplot as plt

## Selecting sweep parameters
# Reference solution values
ft = 'PDF'
G_reference = 15
p_reference = 5

# Experiment parameters
experiment_type = 'homogeneous'
method = 'KMV'
minimum_mesh_velocity = 1.0
frequency = 5.0

## Generating comm
model = spyro.tools.create_model_for_grid_point_calculation(frequency,1,method,minimum_mesh_velocity,experiment_type=experiment_type)
comm = spyro.utils.mpi_init(model)

## Calculating reference solution with p=5 and g=15:
degree = p_reference
model = spyro.tools.create_model_for_grid_point_calculation(frequency, degree, method, minimum_mesh_velocity, experiment_type = experiment_type, receiver_type = 'near')
print('Calculating reference solution of G = '+str(G_reference)+' and p = '+str(p_reference), flush = True)
p_exact = spyro.tools.wave_solver(model, G =G_reference, comm = comm)
comm.comm.barrier()

## Calculating P=2G=12
degree = 2
G = 12
print('\nFor p of '+str(degree), flush = True)

model = spyro.tools.create_model_for_grid_point_calculation(frequency, degree, method, minimum_mesh_velocity, experiment_type = experiment_type, receiver_type = 'near')
print('G of '+str(G), flush = True)
p2 = spyro.tools.wave_solver(model, G =G, comm = comm)
error = spyro.tools.error_calc(p_exact, p2, model, comm = comm)
print('With P of '+ str(degree) +' and G of '+str(G)+' Error = '+str(error), flush = True)

## Calculating P=3G=12
degree = 3
G = 12
print('\nFor p of '+str(degree), flush = True)

model = spyro.tools.create_model_for_grid_point_calculation(frequency, degree, method, minimum_mesh_velocity, experiment_type = experiment_type, receiver_type = 'near')
print('G of '+str(G), flush = True)
p3 = spyro.tools.wave_solver(model, G =G, comm = comm)
error = spyro.tools.error_calc(p_exact, p3, model, comm = comm)
print('With P of '+ str(degree) +' and G of '+str(G)+' Error = '+str(error), flush = True)


## Calculating P=4G=12
degree = 4
G = 12
print('\nFor p of '+str(degree), flush = True)

model = spyro.tools.create_model_for_grid_point_calculation(frequency, degree, method, minimum_mesh_velocity, experiment_type = experiment_type, receiver_type = 'near')
print('G of '+str(G), flush = True)
p4 = spyro.tools.wave_solver(model, G =G, comm = comm)
error = spyro.tools.error_calc(p_exact, p4, model, comm = comm)
print('With P of '+ str(degree) +' and G of '+str(G)+' Error = '+str(error), flush = True)

final_time = model["timeaxis"]["tf"]
    
# Check if shapes are matching
times0, receivers0 = p_exact.shape
times2, receivers2 = p2.shape
times3, receivers3 = p3.shape
times4, receivers4 = p4.shape

dt0 = final_time/times0
dt2 = final_time/times2
dt3 = final_time/times3
dt4 = final_time/times4

nt0 = round(final_time / dt0)  # number of timesteps
nt2 = round(final_time / dt2)  # number of timesteps
nt3 = round(final_time / dt3)  # number of timesteps
nt4 = round(final_time / dt4)  # number of timesteps

id = 1

time_vector0 = np.linspace(0.0, final_time, nt0)
time_vector2 = np.linspace(0.0, final_time, nt2)
time_vector3 = np.linspace(0.0, final_time, nt3)
time_vector4 = np.linspace(0.0, final_time, nt4)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.tick_params(reset=True, direction="in", which="both")
plt.rc("legend", **{"fontsize": 18})
plt.plot(time_vector0, p_exact[:, id], 'bo', time_vector2, p2[:,id], 'go' , time_vector3, p3[:,id], 'ro' , time_vector4, p4[:,id], 'ko' ) 
ax.yaxis.get_offset_text().set_fontsize(18)
plt.subplots_adjust(left=0.18, right=0.95, bottom=0.14, top=0.95)
plt.legend(loc="best")
plt.xlabel("time (s)", fontsize=18)
plt.ylabel("value ", fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim(model['acquisition']['delay']*0.8, final_time)
#plt.ylim(tf, 0)
plt.savefig("Receivers." + ft, format=ft)
# plt.axis("image")
plt.show()
plt.close()

