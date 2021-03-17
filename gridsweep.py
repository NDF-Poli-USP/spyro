import sys
sys.path.append('/home/alexandre/Development/Spyro-main/spyro')
import spyro

G_reference = 15
dg = 0.1
Gs = [x for x in range(4,12) ]
degrees = [1,2,3,4,5]

experiment_type = 'homogeneous'
method = 'KMV'
minimum_mesh_velocity = 1.0
frequency = 5.0
comm = spyro.utils.mpi_init(model)

for degree in degrees:
    model = spyro.tools.create_model_for_grid_point_calculation(frequency, degree, method, minimum_mesh_velocity, experiment_type = experient_type, receiver_type = 'near')
    print('Calculating reference solution of G = '+str(G_reference), flush = True)
    p_exact = wave_solver(model, G =G_reference, comm = comm)
    comm.comm.barrier()
    print('Starting sweep:', flush = True)
    for G in Gs:
        print('G of '+str(G), flush = True)
        p_0 = wave_solver(model, G =G, comm = comm)
        error = error_calc(p_exact, p_0, model, comm = comm)
        print('With G of '+str(G)+'Error = '+str(error), flush = True)

