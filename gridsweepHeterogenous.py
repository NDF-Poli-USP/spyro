import sys
sys.path.append('/home/alexandre/Development/Spyro-3workingBranch')
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

# Degrees and Gs for sweep
Gs = [8, 9, 10, 11, 12]
degrees = [2,3,4,5]

# Experiment parameters
experiment_type = 'heterogeneous'
method = 'KMV'
minimum_mesh_velocity = False
frequency = 5.0

## Generating comm
model = spyro.tools.create_model_for_grid_point_calculation(frequency,1,method,minimum_mesh_velocity,experiment_type=experiment_type)
comm = spyro.utils.mpi_init(model)

## Output file for saving data
date = datetime.today().strftime('%Y_%m_%d')
text_file = open("output_heterogeneous_immersedSourceSigma500itselfreference_NOFILTER_correctError"+date+".txt", "w")
text_file.write('Heterogeneous and KMV \n')

## Generating csv file for visualizing receiver and source position in paraview
saving_source_and_receiver_location_in_csv(model)

## Starting sweep
for degree in degrees:
    print('\nFor p of '+str(degree), flush = True)
    text_file.write('For p of '+str(degree)+'\n')
    print('Starting sweep:', flush = True)
    ## Calculating reference solution with p=5 and g=15:
    model = spyro.tools.create_model_for_grid_point_calculation(frequency, degree, method, minimum_mesh_velocity, experiment_type = experiment_type, receiver_type = 'near')
    print('Calculating reference solution of G = '+str(G_reference)+' and p = '+str(degree), flush = True)
    p_exact = spyro.tools.wave_solver(model, G =G_reference, comm = comm)
    comm.comm.barrier()
    text_file.write('\tG\t\tError \n')
    for G in Gs:
        model = spyro.tools.create_model_for_grid_point_calculation(frequency, degree, method, minimum_mesh_velocity, experiment_type = experiment_type, receiver_type = 'near')
        print('G of '+str(G), flush = True)
        p_0 = spyro.tools.wave_solver(model, G =G, comm = comm)
        error = spyro.tools.error_calc(p_exact, p_0, model, comm = comm)
        print('With P of '+ str(degree) +' and G of '+str(G)+' Error = '+str(error), flush = True)
        text_file.write('\t'+ str(G) +'\t\t'+str(error)+' \n')
        #spyro.plots.plot_receiver_difference(model, p_exact, p_0, 1, appear=True)

text_file.close()

