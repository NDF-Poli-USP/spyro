import sys
sys.path.append('/home/alexandre/Development/OLD_branches_backup_23_04_2021/Spyro-new_source')
from datetime import datetime
import spyro
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams['font.size']   = 12

def saving_source_and_receiver_location_in_csv(model):
    file_name = 'experiment/sources.txt'
    file_obj = open(file_name,'w')
    file_obj.write('Z,\tX \n')
    for source in model['acquisition']['source_pos']:
        z, x = source
        string = str(z)+',\t'+str(x)+' \n'
        file_obj.write(string)
    file_obj.close()

    file_name = 'experiment/receivers.txt'
    file_obj = open(file_name,'w')
    file_obj.write('Z,\tX \n')
    for receiver in model['acquisition']['receiver_locations']:
        z, x = receiver
        string = str(z)+',\t'+str(x)+' \n'
        file_obj.write(string)
    file_obj.close()

    return None

# Comparison data
ref_degree = 5
p20more = True
stop_point = 10.0 #offset
start_time = 1.0
stop_time  = 3.5 
# Shots

Gs_no20P = [11.7, 10.5, 10.5, 8.4]
Gs_20p = [g*1.2 for g in Gs_no20P]
degrees = [2, 3, 4, 5]

if p20more:
    Gs = Gs_20p
else:
    Gs = Gs_no20P


# Experiment parameters
experiment_type = 'heterogeneous'
method = 'KMV'
minimum_mesh_velocity = 1.429
frequency = 5.0

## Generating comm and model
model = spyro.tools.create_model_for_grid_point_calculation(frequency,ref_degree,method,minimum_mesh_velocity,experiment_type=experiment_type, receiver_type = 'line')
comm = spyro.utils.mpi_init(model)

## Loading comparison data
p_exact = spyro.io.load_shots('experiment/p_'+str(ref_degree)+'_referenceLONGTIME.pck')

num_times, num_receivers = np.shape(p_exact)
receiver_locations = model["acquisition"]["receiver_locations"]


## Running forward model
fig, ax = plt.subplots(len(degrees))
cont = 0
for degree in degrees:
    degree_id = degrees.index(degree)
    p = spyro.io.load_shots('experiment/TESTINGp_' + str(degree) + '_G_LONGTIME' + str(Gs[degree_id]) + '.pck')
    #spyro.io.save_shots('experiment/TESTINGp_' + str(degree) + '_G_' + str(G) + '.pck', p)

    error = 100*spyro.tools.error_calc(p_exact, p, model, comm = comm)

    print(error)
    cont += 1
    #plt.yscale("log")
    #plt.show()

    # #spyro.plots.plot_receiver_difference(model,p_exact,p,np.argmax(error), appear= True)
    # p_receiver0 = p_exact
    # p_receiver1 = p
    # id = np.argmax(error)
    # ft = 'PDF'
    # final_time = model["timeaxis"]["tf"]
    # # Check if shapes are matching
    # times0, receivers0 = p_receiver0.shape
    # times1, receivers1 = p_receiver1.shape

    # dt0 = final_time/times0
    # dt1 = final_time/times1

    # nt0 = round(final_time / dt0)  # number of timesteps
    # nt1 = round(final_time / dt1)  # number of timesteps

    # time_vector0 = np.linspace(0.0, final_time, nt0)
    # time_vector1 = np.linspace(0.0, final_time, nt1)


