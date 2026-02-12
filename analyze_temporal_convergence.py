"""
Analyze temporal convergence of MMS results.
For a central difference scheme in time, we expect second-order convergence (rate = 2).
This means error should decrease by ~4x when dt is halved.
"""

import numpy as np
import re
import sys
from collections import defaultdict


class TeeOutput:
    """Write to both console and file simultaneously."""
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.stdout = sys.stdout
    
    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
    
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    
    def close(self):
        self.file.close()


def parse_mms_results(filename):
    """Parse the MMS results file and organize data by h value."""
    data = defaultdict(lambda: {"dt": [], "e1": [], "e2": []})
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    current_dt = None
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for dt value
        if line.startswith("dt = "):
            dt_str = line.split("=")[1].strip()
            current_dt = float(dt_str)
            i += 1
            continue
        
        # Skip separator lines and headers
        if line.startswith("-") or line.startswith("h ") or not line or line.startswith("="):
            i += 1
            continue
        
        # Parse data lines (h, e1, e2)
        if current_dt is not None:
            parts = line.split()
            if len(parts) >= 3 and parts[1] != "Failed" and parts[2] != "Failed":
                h = float(parts[0])
                e1 = float(parts[1])
                e2 = float(parts[2])
                
                data[h]["dt"].append(current_dt)
                data[h]["e1"].append(e1)
                data[h]["e2"].append(e2)
        
        i += 1
    
    return data


def calculate_convergence_rate(dt_values, errors):
    """
    Calculate convergence rate between consecutive refinements.
    For second-order convergence: error ~ dt^p, where p should be ~2
    Rate = log(error1/error2) / log(dt1/dt2)
    """
    rates = []
    for i in range(len(dt_values) - 1):
        if errors[i] > 0 and errors[i+1] > 0:
            rate = np.log(errors[i] / errors[i+1]) / np.log(dt_values[i] / dt_values[i+1])
            rates.append(rate)
        else:
            rates.append(np.nan)
    return rates


def analyze_convergence(data, expected_rate=2.0, tolerance=0.1):
    """
    Analyze convergence for each h value.
    
    Parameters:
    -----------
    data : dict
        Dictionary with h values as keys
    expected_rate : float
        Expected convergence rate (2.0 for second-order)
    tolerance : float
        Acceptable deviation from expected rate
    """
    print("=" * 80)
    print("TEMPORAL CONVERGENCE ANALYSIS")
    print("Expected convergence rate: {:.1f} (second-order in time)".format(expected_rate))
    print("Tolerance: ± {:.1f}".format(tolerance))
    print("=" * 80)
    print()
    
    h_values = sorted(data.keys())
    
    for h in h_values:
        print(f"h = {h}")
        print("-" * 80)
        
        # Sort by dt (largest to smallest)
        dt_vals = np.array(data[h]["dt"])
        e1_vals = np.array(data[h]["e1"])
        e2_vals = np.array(data[h]["e2"])
        
        sorted_idx = np.argsort(dt_vals)[::-1]
        dt_sorted = dt_vals[sorted_idx]
        e1_sorted = e1_vals[sorted_idx]
        e2_sorted = e2_vals[sorted_idx]
        
        # Calculate convergence rates
        rates_e1 = calculate_convergence_rate(dt_sorted, e1_sorted)
        rates_e2 = calculate_convergence_rate(dt_sorted, e2_sorted)
        
        # Print table
        print(f"{'dt':<12} {'e1':<15} {'rate(e1)':<12} {'e2':<15} {'rate(e2)':<12} {'Status':<15}")
        print("-" * 80)
        
        for i in range(len(dt_sorted)):
            dt_str = f"{dt_sorted[i]:.5g}"
            e1_str = f"{e1_sorted[i]:.6e}"
            e2_str = f"{e2_sorted[i]:.6e}"
            
            if i < len(rates_e1):
                rate_e1_str = f"{rates_e1[i]:.2f}"
                rate_e2_str = f"{rates_e2[i]:.2f}"
                
                # Check if converging at expected rate
                e1_converging = abs(rates_e1[i] - expected_rate) <= tolerance
                e2_converging = abs(rates_e2[i] - expected_rate) <= tolerance
                
                if e1_converging and e2_converging:
                    status = "✓ PASS"
                elif e1_converging or e2_converging:
                    status = "~ PARTIAL"
                else:
                    status = "✗ FAIL"
            else:
                rate_e1_str = "-"
                rate_e2_str = "-"
                status = "-"
            
            print(f"{dt_str:<12} {e1_str:<15} {rate_e1_str:<12} {e2_str:<15} {rate_e2_str:<12} {status:<15}")
        
        # Summary for this h
        if len(rates_e1) > 0:
            valid_rates_e1 = [r for r in rates_e1 if not np.isnan(r)]
            valid_rates_e2 = [r for r in rates_e2 if not np.isnan(r)]
            
            if valid_rates_e1:
                avg_rate_e1 = np.mean(valid_rates_e1)
                avg_rate_e2 = np.mean(valid_rates_e2)
                
                print()
                print(f"Average convergence rate (e1): {avg_rate_e1:.2f}")
                print(f"Average convergence rate (e2): {avg_rate_e2:.2f}")
                
                e1_ok = abs(avg_rate_e1 - expected_rate) <= tolerance
                e2_ok = abs(avg_rate_e2 - expected_rate) <= tolerance
                
                if e1_ok and e2_ok:
                    print(f"Overall for h={h}: ✓ CONVERGING at expected second-order rate")
                else:
                    print(f"Overall for h={h}: ✗ NOT converging at expected rate")
        
        print()
        print()


if __name__ == "__main__":
    input_filename = "mms_results.txt"
    output_filename = "temporal_convergence_analysis.txt"
    
    # Parse data
    data = parse_mms_results(input_filename)
    
    if not data:
        print(f"Error: No valid data found in {input_filename}")
    else:
        # Redirect output to both console and file
        tee = TeeOutput(output_filename)
        sys.stdout = tee
        
        try:
            # Analyze convergence
            analyze_convergence(data, expected_rate=2.0, tolerance=0.1)
        finally:
            # Restore stdout and close file
            sys.stdout = tee.stdout
            tee.close()
        
        print(f"\nAnalysis complete. Results saved to {output_filename}")
