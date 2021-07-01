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

Gs_no20P = [8.4]#[11.7, 10.5, 10.5, 8.4]
Gs_20p = [g*1.2 for g in Gs_no20P]
degrees = [5]

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

## Running forward model
#saving_source_and_receiver_location_in_csv(model)
#p = spyro.tools.wave_solver(model, G =G, comm = comm)
# Saving for later testing
fig, ax = plt.subplots(len(degrees))
cont = 0
for degree in degrees:
    degree_id = degrees.index(degree)
    p = spyro.io.load_shots('experiment/TESTINGp_' + str(degree) + '_G_LONGTIME' + str(Gs[degree_id]) + '.pck')
    #spyro.io.save_shots('experiment/TESTINGp_' + str(degree) + '_G_' + str(G) + '.pck', p)
    error = np.zeros( (stop_id,1) )
    xs    = np.zeros( (stop_id,1) )

    for receiver_id in range(stop_id):
        error[receiver_id] = 100*spyro.tools.error_calc_line(p_exact[:,receiver_id], p[:,receiver_id], model, comm = comm)
        xs[receiver_id] = -model["acquisition"]["source_pos"][0][1] + receiver_locations[receiver_id,1]

    ax.plot(xs, error, "-")
    ax.plot([2.0,10.0],[5, 5], 'k--')
    max_error_location = xs[np.argmax(error)][0]
    max_error = np.amax(error)
    ax.plot([max_error_location ,max_error_location , 2.0 ],[0.0  , max_error  , max_error ], '--')
    print("Max error location is "+str(max_error_location)+ " km.")
    print("Max error value is "+ str(max_error)+" %")

    ax.set_xlim((2.0,10.0))
    ax.set_ylim((0.0,9.0))
    xticks = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    yticks = [1, 3, 5, 7, 9]
    # modifying xticks and yticks to have the max error and location and take out nearest tick to not get crowded
    temp = np.argmin(abs(xticks-max_error_location))
    xticks[temp] = max_error_location
    temp = np.argmin(abs(yticks-max_error))
    yticks[temp] = max_error

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set(xlabel = "offset (km)", ylabel = "$E_{x = (-1.0, 36.5)}$ %")
    if p20more:
        withoffset = ' with 20% more G'
    else:
        withoffset = ' without 20% more G'
    ax.set_title("Error with varying offset for KMV"+str(degree)+"tri on BP2004 velocity model")
    cont += 1
    fig.set_size_inches(13,5)
    #plt.yscale("log")
#plt.show()

#     # #spyro.plots.plot_receiver_difference(model,p_exact,p,np.argmax(error), appear= True)
#     # p_receiver0 = p_exact
#     # p_receiver1 = p
#     # id = np.argmax(error)
#     # ft = 'PDF'
#     # final_time = model["timeaxis"]["tf"]
#     # # Check if shapes are matching
#     # times0, receivers0 = p_receiver0.shape
#     # times1, receivers1 = p_receiver1.shape

#     # dt0 = final_time/times0
#     # dt1 = final_time/times1

#     # nt0 = round(final_time / dt0)  # number of timesteps
#     # nt1 = round(final_time / dt1)  # number of timesteps

#     # time_vector0 = np.linspace(0.0, final_time, nt0)
#     # time_vector1 = np.linspace(0.0, final_time, nt1)

#     # #fig = plt.figure()
#     # #ax = fig.add_subplot(111)
#     # #plt.tick_params(reset=True, direction="in", which="both")
#     # #plt.rc("legend")
#     # ax2.plot(time_vector0, p_receiver0[:, id], label = 'Reference', c = 'b', linewidth = 2.0 ) 
#     # ax2.plot(time_vector1, p_receiver1[:, id], label = 'With C = ', c = 'g', linewidth = 2.0 ) 

#     # ## Cutting of to zoom in at time interval start_time-stop_time
#     # ax2.set_xlim((start_time,stop_time))

#     # #ax2.subplots_adjust(left=0.18, right=0.95, bottom=0.14, top=0.95)
#     # #ax2.legend(loc="best")
#     # ax2.set(xlabel = "time (s)", ylabel = "measured pressure")
#     # ax2.set_title("Pressure at receiver on "+str(np.round(max_error_location,2))+ " km offset")
#     # #plt.xlim(model['acquisition']['delay']*0.8, final_time)
#     # #plt.ylim(tf, 0)
#     # #plt.savefig("Receivers." + ft, format=ft)
#     # # plt.axis("image")
#     fig.set_size_inches(10,13)
#     fig.tight_layout(pad=3)

# degree = 3
# degree_id = degrees.index(degree)
# p = spyro.io.load_shots('experiment/TESTINGp_' + str(degree) + '_G_LONGTIME' + str(Gs[degree_id]) + '.pck')
# error = np.zeros( (stop_id,1) )
# xs    = np.zeros( (stop_id,1) )

# for receiver_id in range(stop_id):
#     error[receiver_id] = 100*spyro.tools.error_calc_line(p_exact[:,receiver_id], p[:,receiver_id], model, comm = comm)
#     xs[receiver_id] = -model["acquisition"]["source_pos"][0][1] + receiver_locations[receiver_id,1]

# p_receiver0 = p_exact
# p_receiver1 = p
# id = 0#np.argmax(error)
# fig, ax = plt.subplots()
# times0, receivers0 = p_receiver0.shape
# times1, receivers1 = p_receiver1.shape
# final_time = model["timeaxis"]["tf"]
# dt0 = final_time/times0
# dt1 = final_time/times1

# nt0 = round(final_time / dt0)  # number of timesteps
# nt1 = round(final_time / dt1)  # number of timesteps
# max_error_location = xs[np.argmax(error)][0]

# time_vector0 = np.linspace(0.0, final_time, nt0)
# time_vector1 = np.linspace(0.0, final_time, nt1)
# time_vector0 = np.linspace(0.0, final_time, nt0)
# colors = ['b', 'y', 'g', 'r']
# ax.plot(time_vector0, p_receiver0[:, id], label = 'Reference', c = 'k', linewidth = 0.5 ) 

# cont = 0
# for degree in degrees:
#     degree_id = degrees.index(degree)
#     p = spyro.io.load_shots('experiment/TESTINGp_' + str(degree) + '_G_LONGTIME' + str(Gs[degree_id]) + '.pck')
#     for receiver_id in range(stop_id):
#         error[receiver_id] = 100*spyro.tools.error_calc_line(p_exact[:,receiver_id], p[:,receiver_id], model, comm = comm)
#         xs[receiver_id] = -model["acquisition"]["source_pos"][0][1] + receiver_locations[receiver_id,1]

#     p_receiver0 = p_exact
#     p_receiver1 = p
#     ft = 'PDF'
#     final_time = model["timeaxis"]["tf"]
#     max_error = np.amax(error)
#     # Check if shapes are matching
#     times0, receivers0 = p_receiver0.shape
#     times1, receivers1 = p_receiver1.shape

#     dt0 = final_time/times0
#     dt1 = final_time/times1

#     nt0 = round(final_time / dt0)  # number of timesteps
#     nt1 = round(final_time / dt1)  # number of timesteps

#     time_vector0 = np.linspace(0.0, final_time, nt0)
#     time_vector1 = np.linspace(0.0, final_time, nt1)

#     #fig = plt.figure()
#     #ax = fig.add_subplot(111)
#     #plt.tick_params(reset=True, direction="in", which="both")
#     #plt.rc("legend")
#     #fig, ax = plt.subplots()
    
#     ax.plot(time_vector1, p_receiver1[:, id], label = 'KMV'+str(degree)+'tri', c = colors[cont], linewidth = 0.5 ) 
#     ax.legend(loc="best")
#     ## Cutting of to zoom in at time interval start_time-stop_time
#     ax.set_xlim((start_time,stop_time))

#     #ax2.subplots_adjust(left=0.18, right=0.95, bottom=0.14, top=0.95)
#     ax.set(xlabel = "simulated time (s)", ylabel = "pressure")
#     #ax.set_title("Pressure at receiver on "+str(np.round(max_error_location,2))+ " km offset")
#     ax.set_title("Pressure at receiver with 2 km offset")
#     #plt.xlim(model['acquisition']['delay']*0.8, final_time)
#     #plt.ylim(tf, 0)
#     #plt.savefig("Receivers." + ft, format=ft)
#     # plt.axis("image")

#     fig.set_size_inches(10,5)
#     cont += 1
plt.show()
