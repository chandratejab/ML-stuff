import pandas as pd


def read_file(args):
    data = pd.read_csv(args['input_file'])
    return data
