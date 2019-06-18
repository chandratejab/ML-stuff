from settings import INPUT_DATA
from src.data.data_pull import read_file

args = {
    'input_file': INPUT_DATA
}

def pull():
    return read_file(args)
