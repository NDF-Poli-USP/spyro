import sys
sys.path.append('/home/alexandre/Development/OLD_branches_backup_23_04_2021/Spyro-new_source')
from datetime import datetime
import spyro
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import copy
from scipy import interpolate

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

def time_interpolation_line(p_old, p_exact, model):
    times, = p_exact.shape
    dt = model["timeaxis"]['tf']/times

    times_old, = p_old.shape
    dt_old = model["timeaxis"]['tf']/times_old
    time_vector_old = np.zeros((1,times_old))
    for ite in range(times_old):
        time_vector_old[0,ite] = dt_old*ite

    time_vector_new = np.zeros((1,times))
    for ite in range(times):
        time_vector_new[0,ite] = dt*ite

    p = np.zeros((times,))
    f = interpolate.interp1d(time_vector_old[0,:], p_old[:] )
    p[:] = f(time_vector_new[0,:])

    return p

def error_calc_line(p_exact, p, model, comm = False):
    # p0 doesn't necessarily have the same dt as p_exact
    # therefore we have to interpolate the missing points
    # to have them at the same length
    # testing shape
    times_p_exact, = p_exact.shape
    times_p, = p.shape
    if times_p_exact > times_p: #then we interpolate p_exact
        times,= p.shape
        dt = model["timeaxis"]['tf']/times
        p_exact = time_interpolation_line(p_exact, p, model)
    elif times_p_exact < times_p: #then we interpolate p
        times,= p_exact.shape
        dt = model["timeaxis"]['tf']/times
        p = time_interpolation_line(p, p_exact, model)
    else: #then we dont need to interpolate
        times, = p.shape
        dt = model["timeaxis"]['tf']/times


    if comm.ensemble_comm.rank ==0:
        numerator_time_int = 0.0
        denominator_time_int = 0.0
        # Integrating with trapezoidal rule
        for t in range(times-1):
            numerator_time_int   += (p_exact[t]-p[t])**2
            denominator_time_int += (p_exact[t])**2
        numerator_time_int -= ((p_exact[0]-p[0])**2 + (p_exact[times-1]-p[times-1])**2)/2
        numerator_time_int *= dt
        denominator_time_int -= (p_exact[0]**2+p_exact[times-1]**2)/2
        denominator_time_int *= dt
	
        #if denominator_time_int > 1e-15:
        error = np.sqrt(numerator_time_int/denominator_time_int)

        #if numerator_time_int < 1e-15:
         #   print('Warning: error too small to measure correctly.', flush = True)
            #error = 0.0
        if denominator_time_int < 1e-15:
            print("Warning: receivers don't appear to register a shot.", flush = True)
            error = 0.0

    return error

def plot_recs(p_receiver0, p_receiver1, id):
    p_receiver0 = p_exact
    p_receiver1 = p
    ft = 'PDF'
    final_time = model["timeaxis"]["tf"]
    # Check if shapes are matching
    times0, receivers0 = p_receiver0.shape
    times1, receivers1 = p_receiver1.shape

    dt0 = final_time/times0
    dt1 = final_time/times1

    nt0 = round(final_time / dt0)  # number of timesteps
    nt1 = round(final_time / dt1)  # number of timesteps

    time_vector0 = np.linspace(0.0, final_time, nt0)
    time_vector1 = np.linspace(0.0, final_time, nt1)

    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #plt.tick_params(reset=True, direction="in", which="both")
    #plt.rc("legend")
    fig, ax = plt.subplots()
    ax.plot(time_vector0, p_receiver0[:, id], label = 'Reference', c = 'b', linewidth = 2.0 ) 
    ax.plot(time_vector1, p_receiver1[:, id], label = 'With $\\tilde{C}$ = 2.03', c = 'g', linewidth = 2.0 ) 
    ax.legend(loc="best")
    ## Cutting of to zoom in at time interval start_time-stop_time
    ax.set_xlim((start_time,stop_time))

    #ax2.subplots_adjust(left=0.18, right=0.95, bottom=0.14, top=0.95)
    ax.set(xlabel = "time (s)", ylabel = "measured pressure")
    #plt.xlim(model['acquisition']['delay']*0.8, final_time)
    #plt.ylim(tf, 0)
    #plt.savefig("Receivers." + ft, format=ft)
    # plt.axis("image")

    fig.set_size_inches(10,5)
    plt.show()

# Comparison data
ref_degree = 5
p20more = True
stop_point = 10.0 #offset
start_time = 2.4
stop_time  = 2.8
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

for receiver_id in range(num_receivers):
    if (receiver_locations[receiver_id,1]-model["acquisition"]["source_pos"][0][1]) < stop_point :
        stop_id = receiver_id
stop_id += 1

degree = 5
degree_id = degrees.index(degree)
p = spyro.io.load_shots('experiment/TESTINGp_' + str(degree) + '_G_LONGTIME' + str(Gs[degree_id]) + '.pck')

## Locating receiver
error = np.zeros( (stop_id,1) )

for receiver_id in range(stop_id):
    error[receiver_id] = 100*spyro.tools.error_calc_line(p_exact[:,receiver_id], p[:,receiver_id], model, comm = comm)

id = np.argmax(error)


## Error percentage with reflected wave
r_exact = p_exact[:,id]
r = p[:,id]
times_p_exact, = r_exact.shape
times_p, = r.shape
if times_p_exact > times_p: #then we interpolate p_exact
    times,= r.shape
    dt = model["timeaxis"]['tf']/times
    r_exact = time_interpolation_line(r_exact, r, model)
elif times_p_exact < times_p: #then we interpolate p
    times,= r_exact.shape
    dt = model["timeaxis"]['tf']/times
    r = time_interpolation_line(r, r_exact, model)
else: #then we dont need to interpolate
    times, = r.shape
    dt = model["timeaxis"]['tf']/times

error_complete = 100*spyro.tools.error_calc_line(r_exact, r, model, comm = comm)
print('Error with reflected wave')
print(error_complete)

## PLotting it
#plot_recs(p_receiver0, p_receiver1, id)

## Error without reflected wave
wave_start = 2.4
wave_end   = 2.8
r_exact_no_reflected = copy.deepcopy(r_exact)
r_no_reflected = copy.deepcopy(r)

for t in range(times-1):
    if dt*t > wave_start and dt*t < wave_end:
        r_exact_no_reflected[t] = 0.0
        r_no_reflected[t] = 0.0
    
error_complete = 100*spyro.tools.error_calc_line(r_exact_no_reflected, r_no_reflected, model, comm = comm)
print('Error without reflected wave')
print(error_complete)