import json


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
