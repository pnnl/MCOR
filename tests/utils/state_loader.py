# -*- coding: utf-8 -*-
""""
Functions to save and load the outputs of a AlternativeSolarProfiles instance.

Files are saved as human-readable json files.

save_state and load_state functions handle the dictionaries which store the
grouped state probability transition dataframes.

save_trials and load_trials functions handle the list of dataframes which store
the calculated trial solar data.

save_date_ranges and load_date_ranges functions handle the list of datetime
indexes from asp.generate_trial_date_ranges.

Pickled objects are no longer being used so as to avoid repeating issues with
Pandas altering pickled formats and needing to create a custom unpickler.
"""

import json
import ast

import pandas as pd


def save_state(dict_dfs, file_name):
    """
    Function to save dict of pandas dataframes

    Adapted from here: https://stackoverflow.com/questions/
    33061302/dictionary-of-pandas-dataframe-to-json"""

    # convert dataframes to dictionary
    dict_jsons = {}

    for key in dict_dfs.keys():
        if type(key) == int:
            stored_key = key
        elif type(key) == tuple:
            stored_key = key.__str__()
        else:
            raise TypeError('dictionary key must be tuple or int')
        # add the index to the dataframe dict to be retrieved in load_state
        # column renamed to avoid pandas loading in as a RangeIndex instead of an int64 index
        df = dict_dfs[key].reset_index().rename(columns={'index': 'index_'})
        dict_jsons[stored_key] = df.to_dict(orient='records')

    # add entry for dtypes and column order
    # this assumes the column order and types are equal for all dfs
    first_key = list(dict_dfs.keys())[0]
    column_order = dict_dfs[first_key].columns.tolist() + ['index_']
    dtypes = dict_dfs[first_key].dtypes.apply(lambda x: x.name).to_dict()

    if type(first_key) == int:
        dtypes['index_'] = 'int64'
    else:
        dtypes['index_'] = 'object'

    dict_jsons['c_order'] = column_order
    dict_jsons['dtypes'] = dtypes

    with open(file_name, 'w') as fp:
        json.dump(dict_jsons, fp, indent=4)


def load_state(file_name):
    with open(file_name, 'r') as fp:
        dict_jsons = json.load(fp)

    column_order = dict_jsons['c_order']
    dtypes = dict_jsons['dtypes']

    dict_dfs = {}
    for key in dict_jsons:
        # skip the column order column
        if key == 'c_order' or key == 'dtypes':
            continue
        # convert json to dataframe
        dataframe = pd.DataFrame(dict_jsons[key]).astype(dtypes)
        # reorder columns to match original
        dataframe = dataframe[column_order]

        # put into the dictionary
        if key.isdigit():
            stored_key = int(key)
        else:
            stored_key = ast.literal_eval(key)
        dict_dfs[stored_key] = dataframe.set_index('index_')

    return dict_dfs


def save_trials(list_dfs, file_name):
    # convert dataframes to dictionary
    dict_jsons = {}

    for ind, df in enumerate(list_dfs):
        # add the index to the dataframe dict to be retrieved in load_state
        df = df.reset_index().rename(columns={'index': 'index_'})
        df['index_'] = df['index_'].dt.strftime('%Y-%m-%d %H:%M:%S')
        dict_jsons[ind] = df.to_dict(orient='records')

    # add entry for dtypes and column order
    # this assumes the column order and types are equal for all dfs
    first_df = list_dfs[0]

    column_order = first_df.columns.tolist() + ['index_']
    dtypes = first_df.dtypes.apply(lambda x: x.name).to_dict()
    dtypes['index_'] = 'object'

    dict_jsons['c_order'] = column_order
    dict_jsons['dtypes'] = dtypes

    with open(file_name, 'w') as fp:
        json.dump(dict_jsons, fp, indent=4)


def load_trials(file_name):
    with open(file_name, 'r') as fp:
        dict_jsons = json.load(fp)

    column_order = dict_jsons['c_order']
    dtypes = dict_jsons['dtypes']

    list_dfs = []
    for key in dict_jsons:
        # skip the column order column
        if key == 'c_order' or key == 'dtypes':
            continue
        # convert json to dataframe
        dataframe = pd.DataFrame(dict_jsons[key]).astype(dtypes)
        dataframe['index_'] = pd.to_datetime(dataframe['index_'], format='%Y-%m-%d %H:%M:%S')
        # reorder columns to match original
        dataframe = dataframe[column_order]
        list_dfs.append(dataframe.set_index('index_'))

    return list_dfs


def save_date_ranges(list_ranges, file_name):
    # convert indexes to dictionary
    dict_jsons = {}

    for ind, date_range in enumerate(list_ranges):
        dict_jsons[ind] = date_range.strftime('%Y-%m-%d %H:%M:%S').tolist()

    with open(file_name, 'w') as fp:
        json.dump(dict_jsons, fp, indent=4)


def load_date_ranges(file_name):
    with open(file_name, 'r') as fp:
        dict_jsons = json.load(fp)

    list_date_range = []
    for key in dict_jsons:
        # convert json to dataframe
        datetime_index = pd.DatetimeIndex(dict_jsons[key])
        list_date_range.append(datetime_index)

    return list_date_range
