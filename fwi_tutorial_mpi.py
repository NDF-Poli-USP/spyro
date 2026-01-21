#!/usr/bin/env python3
"""
FWI Tutorial Script - MPI Version
Run with: mpiexec -n 8 python3 fwi_tutorial_mpi.py

This script contains the complete FWI tutorial converted for MPI execution.
Based on the Jupyter notebook tutorial by Alexandre Olender.
"""

import spyro
import firedrake as fire
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def main():
    """Main FWI tutorial function."""
    
    # Define problem parameters
    degree = 4
    frequency = 5.0
    final_time = 1.3
    
    # Initialize the main dictionary for FWI parameters
    dictionary = {}
    
    # Finite element options
    dictionary["options"] = {
        "cell_type": "T",
        "variant": "lumped",
        "degree": degree,
        "dimension": 2,
    }
    
    # Parallelism settings - important for MPI
    dictionary["parallelism"] = {
        "type": "spatial",
    }
    
    # Mesh parameters
    dictionary["mesh"] = {
        "length_z": 2.0,
        "length_x": 2.0,
        "length_y": 0.0,
    }
    
    # Acquisition geometry
    dictionary["acquisition"] = {
        "source_type": "ricker",
        "source_locations": spyro.create_transect((-0.35, 0.5), (-0.35, 1.5), 8),
        "frequency": frequency,
        "delay": 1.0/frequency,
        "delay_type": "time",
        "receiver_locations": spyro.create_transect((-1.65, 0.5), (-1.65, 1.5), 200),
    }
    
    # Absorbing boundary conditions
    dictionary["absorving_boundary_conditions"] = {
        "status": True,
        "damping_type": "local",
    }
    
    # Time domain parameters
    dictionary["time_axis"] = {
        "initial_time": 0.0,
        "final_time": final_time,
        "dt": 0.0001,
        "amplitude": 1,
        "output_frequency": 100,
        "gradient_sampling_frequency": 1,
    }
    
    # Visualization and output settings
    dictionary["visualization"] = {
        "forward_output": True,
        "forward_output_filename": "results/forward_output.pvd",
        "fwi_velocity_model_output": False,
        "velocity_model_filename": None,
        "gradient_output": False,
        "gradient_filename": "results/Gradient.pvd",
        "adjoint_output": False,
        "adjoint_filename": None,
        "debug_output": False,
    }
    
    # Generate real data (synthetic example)
    def generate_real_data():
        fwi = spyro.FullWaveformInversion(dictionary=dictionary)
        
        fwi.set_real_mesh(input_mesh_parameters={
            "edge_length": 0.05,
            "mesh_type": "firedrake_mesh"
        })
        
        # Get mesh coordinates
        mesh_z = fwi.mesh_z
        mesh_x = fwi.mesh_x
        
        # Define anomalies
        center_z = -1.0
        center_x = 1.0
        radius = 0.4
        
        square_top_z = -0.8
        square_bot_z = -1.2
        square_left_x = 0.8
        square_right_x = 1.2
        
        # Create velocity model with anomalies
        cond = fire.conditional(
            (mesh_z - center_z)**2 + (mesh_x - center_x)**2 < radius**2, 
            3.0,  # Circular anomaly
            2.5   # Background
        )
        
        cond = fire.conditional(
            fire.And(
                fire.And(mesh_z < square_top_z, mesh_z > square_bot_z),
                fire.And(mesh_x > square_left_x, mesh_x < square_right_x)
            ),
            3.5,  # Rectangular anomaly
            cond
        )
        
        return fwi, cond
    
    print("\nStep 1: Generating synthetic 'observed' data...")
    
    # Generate real data setup
    fwi_real, velocity_model = generate_real_data()
    
    # Set the true velocity model and generate shot records
    fwi_real.set_real_velocity_model(
        conditional=velocity_model, 
        output=True,
        dg_velocity_model=False
    )
    
    shot_filename = f"shots/shot_record_f{frequency}_"
    
    print("   Generating synthetic shot records (this may take a few minutes)...")
    
    fwi_real.generate_real_shot_record(
        plot_model=True,
        save_shot_record=True,
        shot_filename=shot_filename,
        high_resolution_model=True
    )
    
    print(f"    Synthetic data saved with prefix: {shot_filename}")
    
    # Configure inversion
    dictionary["inversion"] = {
        "perform_fwi": True,
        "real_shot_record_file": shot_filename,
    }
    
    print("\n Step 2: Setting up FWI inversion...")
    
    # Create FWI object for inversion
    fwi = spyro.FullWaveformInversion(dictionary=dictionary)
    
    # Set up initial guess
    fwi.set_guess_mesh(input_mesh_parameters={
        "mesh_type": "firedrake_mesh", 
        "edge_length": 0.05
    })
    
    initial_velocity = 1.5
    fwi.set_guess_velocity_model(constant=initial_velocity)
    
    # Convert to grid format
    grid_data = spyro.utils.velocity_to_grid(fwi, 0.01, output=True)
    
    # Set up final mesh with gradient mask
    mask_boundaries = {
        "z_min": -1.55,
        "z_max": -0.45,
        "x_min": 0.45,
        "x_max": 1.55,
    }
    
    fwi.set_guess_mesh(input_mesh_parameters={
        "mesh_type": "spyro_mesh",
        "cells_per_wavelength": 2.7,
        "grid_velocity_data": grid_data,
        "gradient_mask": mask_boundaries,
        "output_filename": "test.vtk"
    })
    
    inversion_initial_velocity = 2.5
    fwi.set_guess_velocity_model(constant=inversion_initial_velocity)
    
    print(f"   Initial guess: {inversion_initial_velocity} km/s")
    print("   Mesh and gradient mask configured")
    
    # Run FWI
    vmin = 2.5
    vmax = 3.5
    maxiter = 30
    
    print(f"\n Step 3: Running Full Waveform Inversion...")
    print(f"   Velocity bounds: [{vmin}, {vmax}] km/s")
    print(f"   Maximum iterations: {maxiter}")
    print("   " + "="*50)
    
    # Record start time
    t_start = time.time()
    
    # Run the FWI algorithm
    fwi.run_fwi(vmin=vmin, vmax=vmax, maxiter=maxiter)
    
    # Record end time
    t_end = time.time()
    total_time = t_end - t_start
    

    print("   " + "="*50)
    print(f"   ✅ FWI completed in {total_time:.2f} seconds!")
    
    # Results analysis (only on rank 0)

    print(f"\n📊 RESULTS SUMMARY")
    print("="*50)
        
    final_velocity = fwi.vp
    if hasattr(final_velocity, 'dat'):
        velocity_data = final_velocity.dat.data[:]
        print(f"Velocity statistics:")
        print(f"  • Minimum: {np.min(velocity_data):.3f} km/s")
        print(f"  • Maximum: {np.max(velocity_data):.3f} km/s")
        print(f"  • Mean: {np.mean(velocity_data):.3f} km/s")
        
        recovered_range = np.max(velocity_data) - np.min(velocity_data)
        true_range = 3.5 - 2.5
        print(f"  • Recovered range: {recovered_range:.3f} km/s")
        print(f"  • True range: {true_range:.3f} km/s")
        print(f"  • Range recovery: {(recovered_range/true_range)*100:.1f}%")
        
        print(f"\nRuntime: {total_time:.2f} seconds")
        print("="*50)
        print("FWI Tutorial completed successfully!")
        
        # Try to plot results
        try:
            spyro.plots.plot_model(fwi, 
                                  filename="inverted_model.png", 
                                  flip_axis=False, 
                                  show=False)  # Don't show in headless mode
            print("Inverted model saved as 'inverted_model.png'")
        except Exception as e:
            print(f"Plot generation skipped: {e}")
            print("   Use ParaView to visualize .pvd files")

if __name__ == "__main__":
    main()