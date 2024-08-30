from obspy import read
from obspy import Stream
import glob


def read_gcf(path: str):
    '''
    There is a problem in some .gcf files that cause an error
    when we try to read them with ObsPy.
    
    Docs ???
    '''
    lst = glob.glob(path)
    st = Stream()
    for fname in lst:
        try:
            st += read(fname)
        except Exception as error:
            print('Error', fname, error)
    return st


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