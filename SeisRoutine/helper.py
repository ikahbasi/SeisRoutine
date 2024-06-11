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
