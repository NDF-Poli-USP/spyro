import pstats
p = pstats.Stats("results/profile_coupled.prof")
p.sort_stats("cumulative").print_stats(20)