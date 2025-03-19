import json
import scipy as sp
import numpy as np


def write_to_json_file(data, filename):
    '''
    Writes the given data to a JSON file.

    Parameters:
    data (dict): The data to write to the file.
    filename (str): The name of the file to write the data to.
    '''
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def read_from_json_file(filename):
    '''
    Reads data from a JSON file.

    Parameters:
    filename (str): The name of the file to read the data from.

    Returns:
    dict: The data read from the file.
    '''
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
    return data


def clean_nordic_catalog(inp_name, out_name,
    skip_list=['FOCMEC', 'BIN', 'IAML', 'ACTION', 'GAP']):
    '''
    Cleanup earthquake catalog in nordic format.
    '''
    with open(inp_name, 'r') as inp_file:
        with open(out_name, 'w') as out_file:
            for line in inp_file:
                # If the line is empty.
                if line.strip() == '':
                    continue
                if '-' not in line:
                    continue
                # If specific character exist in the line.
                skip = False
                for char in skip_list:
                    if char in line:
                        skip = True
                        break
                if skip:
                    continue
                # Writing in the output file.
                out_file.write(line)


def resample_fault_trace(*args, step=None, num_points=None, kind='linear'):
    """
    Interpolates a set of 2D points and generates new points either by a fixed step size or by specifying the number of points.

    Parameters:
        *args: 
            - If given as (x, y), it should be two separate lists/arrays of x and y coordinates.
            - If given as a single argument, it must be a list of (x, y) tuples.
        step (float, optional): 
            - Distance between interpolated points. Uses `np.arange`.
            - If provided, `num_points` should be None.
        num_points (int, optional): 
            - The desired number of points. Uses `np.linspace`.
            - If provided, `step` should be None.
        kind (str, optional): 
            - Type of interpolation used in `scipy.interpolate.interp1d`. 
            - Default is `'linear'`. Other options include `'nearest'`, `'cubic'`, `'quadratic'`, etc.

    Returns:
        tuple: 
            - `new_x` (np.ndarray): Interpolated x values.
            - `new_y` (np.ndarray): Interpolated y values.
    
    Example Usage:
        >>> x = [0, 1, 2, 3, 4]
        >>> y = [0, 2, 3, 5, 7]
        >>> new_x, new_y = resample_fault_trace(x, y, step=0.5)
        >>> new_x, new_y = resample_fault_trace(list(zip(x, y)), num_points=50)
    """
    if len(args) == 1 and isinstance(args[0], (list, tuple, np.ndarray)):  
        x, y = zip(*args[0])
    elif len(args) == 2:  
        x, y = args
    else:
        raise ValueError("Invalid input format. Provide either (x, y) lists or a list of (x, y) tuples.")
    
    x, y = np.asarray(x), np.asarray(y)

    # Compute cumulative distances
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    cumulative_dist = np.insert(np.cumsum(distances), 0, 0)

    # Create interpolation functions
    fx = sp.interpolate.interp1d(cumulative_dist, x, kind=kind)
    fy = sp.interpolate.interp1d(cumulative_dist, y, kind=kind)

    # Generate new distances using either np.arange or np.linspace
    if num_points is not None:
        new_distances = np.linspace(0, cumulative_dist[-1], num_points)
    elif step is not None:
        new_distances = np.arange(0, cumulative_dist[-1], step)
        if new_distances[-1] < cumulative_dist[-1]:  # Ensure the last point is included
            new_distances = np.append(new_distances, cumulative_dist[-1])
    else:
        raise ValueError("Either 'step' or 'num_points' must be provided.")

    new_x = fx(new_distances)
    new_y = fy(new_distances)
    return new_x, new_y

