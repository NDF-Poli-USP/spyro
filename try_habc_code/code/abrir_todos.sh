#!/bin/bash

# Number of results per ParaView instance (default 10)
n=${1:-10}

# Get total number of files
total=$(python3 -c "import glob; print(len(glob.glob('**/*fields.xdmf', recursive=True)))")

echo "Total files: $total"
echo "Opening ParaView instances with $n results each"

# Simple loop
for ((start=0; start<total; start+=$n)); do
    end=$((start + n))
    [ $end -gt $total ] && end=$total
    
    echo "Opening: files $start to $((end-1))"
    PV_START=$start PV_END=$end /home/andre/Software/ParaView-5.11.0-RC1-MPI-Linux-Python3.9-x86_64/bin/paraview --state=paraview_open_transform.py &
    
    sleep 30
done

echo "All ParaView instances started!"