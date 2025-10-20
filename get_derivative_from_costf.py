import numpy as np
import re

def read_functional_values(filename):
    """Read functional values from the text file."""
    iterations = []
    functionals = []
    
    with open(filename, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                # Extract iteration and functional using regex
                match = re.match(r'Iteration:\s*(\d+),\s*Functional:\s*([\d.e-]+)', line.strip())
                if match:
                    iteration = int(match.group(1))
                    functional = float(match.group(2))
                    iterations.append(iteration)
                    functionals.append(functional)
    
    return np.array(iterations), np.array(functionals)

def calculate_finite_difference_derivatives(functionals):
    """Calculate first-order finite difference derivatives."""
    n = len(functionals)
    derivatives = np.zeros(n)
    methods = []
    
    for i in range(n):
        if i == 0:  # Forward difference for first point
            derivatives[i] = functionals[i+1] - functionals[i]
            methods.append("Forward difference")
        else:  # i == n-1:  # Backward difference for last point
            derivatives[i] = functionals[i] - functionals[i-1]
            methods.append("Backward difference")
        # else:  # Central difference for interior points
        #     derivatives[i] = (functionals[i+1] - functionals[i-1]) / 2
        #     methods.append("Central difference")
    
    return derivatives, methods

def calculate_normalized_derivatives(functionals, derivatives):
    """Calculate normalized derivatives (derivative / functional value)."""
    return derivatives / functionals

def main():
    # Read data from file
    filename = 'functional_values.txt'
    iterations, functionals = read_functional_values(filename)
    
    # Calculate derivatives
    derivatives, methods = calculate_finite_difference_derivatives(functionals)
    normalized_derivatives = calculate_normalized_derivatives(functionals, derivatives)
    
    # Print results
    print("Normalized first-order finite difference derivatives:")
    print("-" * 100)
    print(f"{'Iteration':<10} {'Functional Value':<20} {'Derivative':<20} {'Normalized Derivative':<20} {'Method'}")
    print("-" * 100)
    
    for i in range(len(iterations)):
        print(f"{iterations[i]:<10} {functionals[i]:<20.6e} {derivatives[i]:<20.6e} "
              f"{normalized_derivatives[i]:<20.4f} {methods[i]}")
    
    # Additional analysis
    print("\nKey observations:")
    print(f"- Steepest absolute decrease: Iteration {np.argmin(derivatives)} (derivative: {np.min(derivatives):.6e})")
    print(f"- Largest normalized decrease: Iteration {np.argmin(normalized_derivatives)} (normalized: {np.min(normalized_derivatives):.4f})")
    print(f"- Any increases: {np.any(derivatives > 0)}")
    if np.any(derivatives > 0):
        increasing_points = np.where(derivatives > 0)[0]
        print(f"  Increasing at iterations: {increasing_points}")
    
    # Save results to file
    output_filename = 'derivative_analysis.txt'
    with open(output_filename, 'w') as f:
        f.write("Normalized first-order finite difference derivatives:\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Iteration':<10} {'Functional Value':<20} {'Derivative':<20} {'Normalized Derivative':<20} {'Method'}\n")
        f.write("-" * 100 + "\n")
        
        for i in range(len(iterations)):
            f.write(f"{iterations[i]:<10} {functionals[i]:<20.6e} {derivatives[i]:<20.6e} "
                   f"{normalized_derivatives[i]:<20.4f} {methods[i]}\n")
    
    print(f"\nResults saved to {output_filename}")

if __name__ == "__main__":
    main()