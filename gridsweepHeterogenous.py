import sys
sys.path.append('/home/alexandre/Development/OLD_branches_backup_23_04_2021/Spyro-new_source')
from datetime import datetime
import spyro

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

## Selecting sweep parameters
# Reference solution values
G_reference = 15
P_reference = 5
calc_ref = False

# Degrees and Gs for sweep
Gs = [9.0, 9.1, 9.2, 9.3, 9.4, 9.5]#, 14, 15, 16, 17, 18, 19, 20]
degrees = [5]#[5]#, 5]

# Experiment parameters
experiment_type = 'heterogeneous'
method = 'KMV'
minimum_mesh_velocity = 1.429
frequency = 5.0
receiver_type = 'line'

## Generating comm
model = spyro.tools.create_model_for_grid_point_calculation(frequency,1,method,minimum_mesh_velocity,experiment_type=experiment_type, receiver_type = receiver_type)
comm = spyro.utils.mpi_init(model)

## Output file for saving data
date = datetime.today().strftime('%Y_%m_%d')
filename = "output_"+experiment_type+"p2_finding"+date
text_file = open(filename+".txt", "w")
text_file.write(experiment_type+' and '+method+' \n')

## Generating csv file for visualizing receiver and source position in paraview
saving_source_and_receiver_location_in_csv(model)

if calc_ref == True:
    model = spyro.tools.create_model_for_grid_point_calculation(frequency, P_reference, method, minimum_mesh_velocity, experiment_type = experiment_type, receiver_type = receiver_type)
    print('Calculating reference solution of G = '+str(G_reference)+' and p = '+str(P_reference), flush = True)
    p_exact = spyro.tools.wave_solver(model, G =G_reference, comm = comm)
    spyro.io.save_shots("experiment/heterogeneous_reference_p"+str(P_reference)+"g"+str(G_reference)+"line.pck", p_exact)
else:
    p_exact = spyro.io.load_shots("experiment/heterogeneous_reference_p"+str(P_reference)+"g"+str(G_reference)+"line.pck")

## Starting sweep
for degree in degrees:
    print('\nFor p of '+str(degree), flush = True)
    text_file.write('For p of '+str(degree)+'\n')
    print('Starting sweep:', flush = True)
    ## Calculating reference solution with p=5 and g=15:
    comm.comm.barrier()
    #p_exact=spyro.io.load_shots('experiment/p_'+str(degree)+'CG_reference.pck')
    text_file.write('\tG\t\tError \n')
    for G in Gs:
        model = spyro.tools.create_model_for_grid_point_calculation(frequency, degree, method, minimum_mesh_velocity, experiment_type = experiment_type, receiver_type = receiver_type)
        print('G of '+str(G), flush = True)
        p_0 = spyro.tools.wave_solver(model, G =G, comm = comm)
        error = spyro.tools.error_calc(p_exact, p_0, model, comm = comm)
        print('With P of '+ str(degree) +' and G of '+str(G)+' Error = '+str(error), flush = True)
        text_file.write('\t'+ str(G) +'\t\t'+str(error)+' \n')
        #spyro.plots.plot_receiver_difference(model, p_exact, p_0, 1, appear=True)

text_file.close()

