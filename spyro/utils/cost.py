from numpy import savetxt
from os import getcwd
from time import perf_counter  # For runtime
from tracemalloc import get_traced_memory, start, stop  # For memory usage


def comp_cost(flag, tRef=None, user_name=None):
    '''
    Estimate runtime and used memory and save them to a *.txt file.

    Parameters
    ----------
    flag : `str`
        Flag to indicate the action to be performed
        - 'tini' to start the timer
        - 'tfin' to finish the timer and print the results
    tRef : `float`, optional
        Reference time in seconds. Default is None
    user_name: `str`, optional
        User name or path to save the computational cost data

    Returns
    -------
    tRef : float
        Reference time in seconds. Only returned if flag is 'tini'
    '''

    if flag == 'tini':
        # Start memory usage
        start()

        # Reference time. Don't move this line!
        tRef = perf_counter()

        return tRef

    elif flag == 'tfin':

        # Separation format
        hifem_draw = 62

        # Total time
        print('\n' + hifem_draw * '-')
        print("Estimating Runtime and Used Memory")
        tTotal = perf_counter() - tRef
        val_time = [tTotal, tTotal/60, tTotal/3600]
        cad_time = 'Runtime: (s):{:3.3f}, (m):{:3.3f}, (h):{:3.3f}'
        print(cad_time.format(*val_time))

        # Memory usage
        curr, peak = get_traced_memory()
        val_memo = [curr/1024**2, peak/1024**2]
        cad_memo = "Used Memory: Current (MB):{:3.3f}, Peak (MB):{:3.3f}"
        print(cad_memo.format(*val_memo))
        print(hifem_draw * '-' + '\n')
        stop()

        # Save file for resource usage
        file_name = 'cost.txt'
        path_file = getcwd() + "/" if user_name is None else user_name
        path_cost = path_file + file_name
        savetxt(path_cost, (*val_time, *val_memo), delimiter='\t')
