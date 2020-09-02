
import numpy as np

def get_plot_ranges(start=0, end=10, n=3):
    distance = end - start
    for i in np.arange(start, end, np.floor(distance/n)) :
        yield (int(i), int(np.minimum(end, np.floor(distance/n)+i)))

for start, stop in get_plot_ranges(0, 10, 3):
    print(start, stop)


'''
print(list(get_plot_ranges(0, 10, 1)))
print(list(get_plot_ranges(0, 10, 2)))
print(list(get_plot_ranges(0, 10, 3)))
print(list(get_plot_ranges(0, 10, 4)))
print(list(get_plot_ranges(0, 10, 5)))
print(list(get_plot_ranges(0, 10, 6)))
print(list(get_plot_ranges(0, 10, 7)))
'''