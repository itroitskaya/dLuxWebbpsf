import jax.numpy as np

"""
Utility functions to do a grid search
"""

__all__ = [
    "get_grid",
    "grid_search",
]

def get_grid(xsize, xmin, xmax, ysize, ymin, ymax):
    """
    Returns a grid of x and y coordinates.

    Parameters:
    xsize (int): Number of x coordinates.
    xmin (float): Minimum x coordinate.
    xmax (float): Maximum x coordinate.
    ysize (int): Number of y coordinates.
    ymin (float): Minimum y coordinate.
    ymax (float): Maximum y coordinate.

    Returns:
    xr (ndarray): Array of x coordinates.
    yr (ndarray): Array of y coordinates.
    """
    allsize = xsize*ysize

    xx, yy = np.meshgrid(np.linspace(xmin, xmax, xsize), np.linspace(ymin, ymax, ysize))

    xr = xx.reshape((allsize))
    yr = yy.reshape((allsize))

    return xr, yr

def grid_search(get_data, x0, y0, grid_size, grid_steps, niter = 5, oversteps = 5):
    """
    Performs a grid search to find the maximum value of a function.

    Parameters:
    get_data (function): Function to be optimized.
    x0 (float): Initial x coordinate.
    y0 (float): Initial y coordinate.
    grid_size (float): Size of the grid.
    grid_steps (int): Number of steps in the grid.
    niter (int): Number of iterations.
    oversteps (int): Number of grid cells to search beyond the found center.

    Returns:
    x_found (float): x coordinate of the maximum.
    y_found (float): y coordinate of the maximum.
    traces (list): List of dictionaries containing information about each iteration.
    """
    i = 0
    x_found = None
    y_found = None
    traces = []
    
    if x0 is None:
        x0 = 0
    if y0 is None:
        y0 = 0
    
    xsteps = grid_steps
    ysteps = grid_steps
    xmin = x0 - grid_size / 2
    xmax = x0 + grid_size / 2
    ymin = y0 - grid_size / 2
    ymax = y0 + grid_size / 2
    
    while i < niter:
        
        xr, yr = get_grid(xsteps, xmin, xmax, ysteps, ymin, ymax)
        
        xstep = xr[1] - xr[0]
        ystep = yr[xsteps] - yr[0]
        
        out = get_data(xr, yr)
        
        likelihoods = np.array(out).reshape((xsteps, ysteps))

        ind_argmax = out.argmax()
        x_found = xr[ind_argmax]
        y_found = yr[ind_argmax]
        
        trace = {
            'x': x_found,
            'y': y_found,
            'xstep': xstep,
            'ystep': ystep,
            'likelihoods': likelihoods
        }
        
        xmin = x_found - xstep*oversteps
        xmax = x_found + xstep*oversteps
        ymin = y_found - ystep*oversteps
        ymax = y_found + ystep*oversteps
        
        traces.append(trace)
        i = i + 1

    return x_found, y_found, traces