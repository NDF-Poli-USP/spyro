import firedrake as fire
import math
import spyro
import matplotlib.pyplot as plt
from collections import defaultdict
import csv

def mms_h_convergence(dt, h):
    u1 = lambda x, t: (x[0]**2 + x[0])*(x[1]**2 - x[1])*t
    u2 = lambda x, t: (2*x[0]**2 + 2*x[0])*(-x[1]**2 + x[1])*t
    u = lambda x, t: fire.as_vector([u1(x, t), u2(x, t)])

    b1 = lambda x, t: -(2*x[0]**2 + 6*x[1]**2 - 16*x[0]*x[1] + 10*x[0] - 14*x[1] + 4)*t
    b2 = lambda x, t: -(-12*x[0]**2 - 4*x[1]**2 + 8*x[0]*x[1] - 16*x[0] + 8*x[1] - 2)*t
    b = lambda x, t: fire.as_vector([b1(x, t), b2(x, t)])

    fo = int(0.1/dt)

    dictionary = {}
    dictionary["options"] = {
        "cell_type": "Q",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
        "variant": "lumped",  # lumped, equispaced or DG, default is lumped "method":"MLT", # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral) You can either specify a cell_type+variant or a method
        "degree": 4,  # p order
        "dimension": 2,  # dimension
    }
    dictionary["parallelism"] = {
        "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
    }
    dictionary["mesh"] = {
        "Lz": 1.0,  # depth in km - always positive
        "Lx": 1.0,  # width in km - always positive
        "Ly": 0.0,  # thickness in km - always positive
        "mesh_type": "firedrake_mesh",  # options: firedrake_mesh or user_mesh
        "mesh_file": None,  # specify the mesh file
    }
    dictionary["acquisition"] = {
        "source_type": "MMS",
        "source_locations": [(-1.0, 1.0)],
        "frequency": 5.0,
        "delay": 1.5,
        "receiver_locations": [(-0.0, 0.5)],
        "body_forces": b,
    }
    dictionary["time_axis"] = {
        "initial_condition": u,
        "initial_time": 0.0,  # Initial time for event
        "final_time": 1.0,  # Final time for event
        "dt": dt,  # timestep size
        "amplitude": 1,  # the Ricker has an amplitude of 1.
        "output_frequency": fo,  # how frequently to output solution to pvds
        "gradient_sampling_frequency": 1,  # how frequently to save solution to RAM
    }

    dictionary["visualization"] = {
        "forward_output": False,
        "output_filename": "results/forward_output.pvd",
        "fwi_velocity_model_output": False,
        "velocity_model_filename": None,
        "gradient_output": False,
        "gradient_filename": None,
    }

    dictionary["synthetic_data"] = {
        "type": "object",
        "density": 1,
        "lambda": 1,
        "mu": 1,
        "real_velocity_file": None,
    }
    dictionary["boundary_conditions"] = [
        ("uz", "on_boundary", 0),
        ("ux", "on_boundary", 0),
    ]

    wave = spyro.IsotropicWave(dictionary)
    wave.set_mesh(input_mesh_parameters={"edge_length": h})
    # Lets build de matrix operator outside of the forward solve so we can
    # set previous timesteps for the MMS problem
    wave._initialize_model_parameters()
    wave.matrix_building()
    x = wave.get_spatial_coordinates()
    # wave.u_nm1.interpolate(u(x, 0.0 - 2*dt))
    # wave.u_n.interpolate(u(x, 0.0 - dt))
    # wave.u_nm2.interpolate(u(x, 0.0 - 2*dt))

    wave.forward_solve(build_matrix_operator=False)

    u_an = fire.Function(wave.function_space)
    t = wave.current_time
    u_an.interpolate(u(x, t))

    e1 = fire.errornorm(wave.u_n.sub(0), u_an.sub(0)) / fire.norm(u_an.sub(0))
    e2 = fire.errornorm(wave.u_n.sub(1), u_an.sub(1)) / fire.norm(u_an.sub(1))

    if e1 > 1e3 or e2 > 1e3:
        raise ValueError("ERROR")

    print(f"For dt: {dt}, h: {h}, e1 = {e1}, e2 = {e2}")
    return e1, e2

if __name__ == "__main__":
    dts = [1e-3, 1e-4, 1e-5]
    # dts = [1e-2, 1e-3]
    hs = [0.125]#, 0.1, 0.05, 0.025]
    
    # Dictionary to store results: results[dt][h] = (e1, e2) or "Error"
    results = defaultdict(dict)
    
    for dt in dts:
        for h in hs:
            # try:
            e1, e2 = mms_h_convergence(dt, h)
            results[dt][h] = (e1, e2)
            # except Exception as e:
            #     print(f"Error at dt: {dt}, h: {h}: {str(e)}")
            #     results[dt][h] = "Error"
    
    # Print results in table format
    print("\n" + "="*80)
    print("RESULTS TABLE")
    print("="*80)
    
    for dt in dts:
        print(f"\ndt = {dt}")
        print("-" * 60)
        print(f"{'h':<10} {'e1':<25} {'e2':<25}")
        print("-" * 60)
        for h in hs:
            if h in results[dt]:
                if results[dt][h] == "Error":
                    print(f"{h:<10} {'Failed':<25} {'Failed':<25}")
                else:
                    e1, e2 = results[dt][h]
                    print(f"{h:<10} {e1:<25.6e} {e2:<25.6e}")
        print()
    
    # Save results to CSV file
    print("\nSaving results to CSV file...")
    with open('mms_results.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['dt', 'h', 'e1', 'e2', 'status'])
        
        for dt in dts:
            for h in sorted(results[dt].keys()):
                if results[dt][h] == "Error":
                    csvwriter.writerow([dt, h, 'Error', 'Error', 'Failed'])
                else:
                    e1, e2 = results[dt][h]
                    csvwriter.writerow([dt, h, f'{e1:.6e}', f'{e2:.6e}', 'Success'])
    
    print("Saved: mms_results.csv")
    
    # Save results to TXT file
    print("Saving results to TXT file...")
    with open('mms_results.txt', 'w') as txtfile:
        txtfile.write("="*80 + "\n")
        txtfile.write("MMS CONVERGENCE TEST RESULTS\n")
        txtfile.write("="*80 + "\n\n")
        
        for dt in dts:
            txtfile.write(f"dt = {dt}\n")
            txtfile.write("-" * 60 + "\n")
            txtfile.write(f"{'h':<10} {'e1':<25} {'e2':<25}\n")
            txtfile.write("-" * 60 + "\n")
            for h in sorted(results[dt].keys()):
                if results[dt][h] == "Error":
                    txtfile.write(f"{h:<10} {'Failed':<25} {'Failed':<25}\n")
                else:
                    e1, e2 = results[dt][h]
                    txtfile.write(f"{h:<10} {e1:<25.6e} {e2:<25.6e}\n")
            txtfile.write("\n")
    
    print("Saved: mms_results.txt")
    
    # Create plots for each dt
    print("\nGenerating convergence plots...")
    
    for dt in dts:
        # Filter out errors and get successful runs
        successful_h = []
        successful_e1 = []
        successful_e2 = []
        
        for h in sorted(results[dt].keys()):
            if results[dt][h] != "Error":
                e1, e2 = results[dt][h]
                successful_h.append(h)
                successful_e1.append(e1)
                successful_e2.append(e2)
        
        if successful_h:  # Only plot if we have data
            plt.figure(figsize=(10, 6))
            plt.loglog(successful_h, successful_e1, 'o-', label='e1 error', linewidth=2, markersize=8)
            plt.loglog(successful_h, successful_e2, 's-', label='e2 error', linewidth=2, markersize=8)
            plt.xlabel('Mesh size h', fontsize=12)
            plt.ylabel('Error', fontsize=12)
            plt.title(f'MMS Convergence for dt = {dt}', fontsize=14)
            plt.legend(fontsize=11)
            plt.grid(True, which="both", ls="-", alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'mms_convergence_dt_{dt}.png', dpi=150)
            print(f"Saved plot: mms_convergence_dt_{dt}.png")
            plt.close()
    
    print("\nAll plots generated successfully!")
