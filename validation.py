# -*- coding: utf-8 -*-
"""
Includes classes and functions for error handling and parameter
    validation

Classes:
- Error: base error class
- ParamValidationError: class for parameter validation errors

Functions:
- validate_parameter: performs parameter validation
- log_error: logs errors and warnings to file

"""

import os
import datetime as dt
from pydoc import locate
from copy import deepcopy
import pandas as pd
import numpy as np
import logging
from scipy.stats import norm
from scipy.signal import periodogram
from pvlib import pvsystem
import pytz
from config import DATA_DIR

VALIDATION_TYPE_STRINGS = {
    "is_none": "not be blank",
    "data_type": "be of type",
    "max_val": "be <=",
    "min_val": "be >=",
    "enums": "be one of",
    "size": "have size"
}

# Load validation constraints csv
CONSTRAINTS_DF = pd.read_csv(os.path.join(DATA_DIR, 'parameter_validation.csv'),
                             index_col=0, usecols=range(9),
                             skiprows=1, encoding="ISO-8859-1",
                             names=['parameter', 'data_type', 'min_val',
                                    'max_val', 'enums', 'size', 'custom_func',
                                    'custom_args', 'custom_message'])

# Format columns
CONSTRAINTS_DF['data_type'] = CONSTRAINTS_DF['data_type'].apply(lambda x: x.split(','))
CONSTRAINTS_DF['enums'] = CONSTRAINTS_DF['enums'].apply(
    lambda x: x.split(',') if isinstance(x, str) else x)

# Turn into dict
CONSTRAINTS_DICT = {param_key: {col_key: col_val
                                for col_key, col_val in param_val.items()
                                if not (isinstance(col_val, float)
                                        and not np.isfinite(col_val))}
                    for param_key, param_val
                    in CONSTRAINTS_DF.to_dict(orient='index').items()}


def log_error(error_message):
    """
    Logs error message to file
    :param error_message: Error message

    """

    logging.basicConfig(filename='MCOR_error_messages.log',
                        level=logging.ERROR,
                        format='\n%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.exception(error_message)


class Error(Exception):
    """Base class for exceptions in this module."""

    def __init__(self, *args, **kwargs):
        pass


class ParamValidationError(Exception):
    """
    Custom exception that is raised for parameter validation errors. Saves results to a
        logger.
    """

    def __init__(self, param_name, violation_type, violation_value,
                 user_value, custom_message=None):
        """
        Creates a message based on the input arguments and saves to a logger.
        :param param_name: Name of the parameter
        :param violation_type: Type of violation, options are:
            ["is_none", "data_type", "max_val", "min_val", "enums", "size"]
        :param violation_value: The value that the parameter is violating
        :param custom_message: A custom message that can be specified instead of the common
            format
       """

        # Create error message
        if custom_message is not None:
            message = custom_message
        else:
            message = "The parameter {} must {} {}. You entered {}".format(
                param_name, VALIDATION_TYPE_STRINGS[violation_type],
                violation_value, user_value)

        # Log error
        log_error(message)
        super(ParamValidationError, self).__init__(message)


def validate_all_parameters(args_dict):
    """ Validate all parameters given a dict.
        The format of args_dict is:
            {"param_name": param}
    """

    # Validate each parameter
    for param_name, param in args_dict.items():
        kwargs = deepcopy(CONSTRAINTS_DICT[param_name])
        if 'custom_func' in kwargs.keys():
            if 'custom_args' in kwargs.keys():
                kwargs['custom_args'] = {key: args_dict[key] for key
                                         in kwargs['custom_args'].split(',')}
            else:
                kwargs['custom_args'] = {}

        validate_parameter(param_name, param, **kwargs)


def validate_parameter(param_name, param_value, data_type=None, max_val=None,
                       min_val=None, enums=None, size=None, custom_func=None,
                       custom_args=None, custom_message=None):
    """
    Performs parameter validation based on optional inputs for constraints:

    :param param_name: the name of the parameter to be validated
    :param param_value: the value of the parameter to be validated
    :param data_type: required data type
    :param max_val: required maximum value (inclusive)
    :param min_val: required minimum value (inclusive)
    :param enums: set of allowed values
    :param size: allowable size
    :param custom_func: a custom function specifying a validation test
    :param custom_args: arguments associated with the custom function
    :param custom_message: error message associated with the custom function
   """

    # First check that the parameter is not None
    try:
        assert param_value is not None
    except AssertionError:
        raise ParamValidationError(param_name, "is_none", "", param_value)

    # Perform try, except with assert statement on all specified args
    if data_type is not None:
        try:
            assert isinstance(param_value, tuple([locate(dtype) for dtype in data_type]))
        except AssertionError:
            raise ParamValidationError(param_name, "data_type", data_type, param_value)

    if max_val is not None:
        try:
            assert param_value <= max_val
        except AssertionError:
            raise ParamValidationError(param_name, "max_val", max_val, param_value)

    if min_val is not None:
        try:
            assert param_value >= min_val
        except AssertionError:
            raise ParamValidationError(param_name, "min_val", min_val, param_value)

    if enums is not None:
        try:
            assert param_value in enums
        except AssertionError:
            raise ParamValidationError(param_name, "enums", enums, param_value)

    if size is not None:
        try:
            assert len(param_value) == size
        except AssertionError:
            raise ParamValidationError(param_name, "size", size, param_value)

    if custom_func is not None:
        try:
            assert VALIDATION_FUNCS[custom_func](param_value, **custom_args)
        except AssertionError:
            raise ParamValidationError(param_name, "", "", param_value,
                                       custom_message=custom_message)


# Custom parameter validation functions
def check_path(path):
    """ Check that the path exists. """
    try:
        _ = os.listdir(path)
        return True
    except FileNotFoundError:
        return False


def check_sitename(sitename, path, start_year, end_year):
    """ Check that data for the corresponding sitename has been downloaded.
    """
    dirs = os.listdir(os.path.join(path, 'nrel'))
    if sitename not in dirs:
        return False
    files = os.listdir(os.path.join(path, 'nrel', sitename))
    for year in range(start_year, end_year+1):
        if '{}_{}.csv'.format(sitename, year) not in files:
            return False
    return True


def check_solar_profile(solar_profile):
    """ Check that solar_profile has the necessary columns and index type.
    """

    # Check that solar_profile has the necessary columns
    if len({'ghi', 'dni'} - set(solar_profile.columns)):
        if len({'dhi', 'dni'} - set(solar_profile.columns)):
            return False

    # Check the type of the index
    if not isinstance(solar_profile.index, pd.DatetimeIndex):
        return False

    return True


def check_temp_profile(temp_profile):
    """ Check that temp_profile has the necessary columns and index type.
    """

    # Check that solar_profile has the necessary columns
    if 'temp_celcius' not in temp_profile.columns:
        return False

    # Check the type of the index
    if not isinstance(temp_profile.index, pd.DatetimeIndex):
        return False
    return True


def check_power_profile(power_profile):
    """ Check that power_profile has the necessary index type. """

    # Check the type of the index
    return isinstance(power_profile.index, pd.DatetimeIndex)


def check_night_profile(night_profile):
    """ Check that night_profile has the necessary columns and index type.
    """

    # Check that night_profile has the necessary columns
    if len({'is_night', 'is_first_hour_of_night', 'night_duration'} -
           set(night_profile.columns)):
        return False

    # Check the type of the index
    if not isinstance(night_profile.index, pd.DatetimeIndex):
        return False
    return True


def check_strings(strings):
    """ Check that strings has the appropriate keys and that those fields have valid values.
    """

    if len({'mods_per_string', 'strings_per_inv'} - set(strings.keys())):
        return False

    validate_parameter('mods_per_string', strings['mods_per_string'],
                       **CONSTRAINTS_DICT['mods_per_string'])
    validate_parameter('strings_per_inv', strings['strings_per_inv'],
                       **CONSTRAINTS_DICT['strings_per_inv'])

    return True


def check_night_duration(night_duration, is_night):
    """ Check that the night duration is > 0 if it is night. """

    if is_night:
        return night_duration > 0
    else:
        return night_duration == 0


def check_fuel_curve_model(fuel_curve_model):
    """ Check that expected keys are present.
    """
    return len({'1/4 Load (gal/hr)', '1/2 Load (gal/hr)', '3/4 Load (gal/hr)',
                'Full Load (gal/hr)'} - set(fuel_curve_model.keys())) == 0


def check_outputs(outputs):
    """ Several checks on the aggregated system output. """

    # Check that required keys are included
    if len({'pv_percent', 'batt_percent', 'gen_percent',
            'storage_recovery_percent', 'fuel_used_gal',
            'generator_power_kW'} - set(outputs.keys())) > 0:
        return False
    return True


def check_existing_components(existing_components):
    """ Check that each element in existing components is a Component of the appropriate type.
    """

    # Check that keys are allowable
    if len(set(existing_components.keys()) - {'pv', 'mre', 'batt', 'generator', 'fuel_tank'}) > 0:
        return False

    # Check the datatype for each elem
    type_key = {'pv': 'microgrid_system.PV',
                'batt': 'microgrid_system.Battery',
                'mre': 'microgrid_system.MRE',
                'generator': 'microgrid_system.Generator',
                'fuel_tank': 'microgrid_system.FuelTank'}
    for key, val in existing_components.items():
        validate_parameter(key, val, data_type=[type_key[key]])
    return True


def check_net_metering_limits(net_metering_limits):
    """ Check that net metering limits have the format:
        {type: ['capacity_cap' or 'percent_of_load'], value: [<kW value> or <percentage>]}
    """

    # Check that it has the required keys
    if 'type' not in net_metering_limits.keys():
        return False

    # Check the data type and range of the value parameter
    if net_metering_limits['type'] == 'no_nm_use_battery':
        return True
    else:
        if 'value' not in net_metering_limits.keys():
            return False
        if net_metering_limits['type'] == 'capacity_cap':
            validate_parameter('net_metering_capacity_cap',
                               net_metering_limits['value'],
                               data_type=['int', 'float'],
                               min_val=0)
            return True
        elif net_metering_limits['type'] == 'percent_of_load':
            validate_parameter('net_metering_percent',
                               net_metering_limits['value'],
                               data_type=['int', 'float'],
                               min_val=0,
                               max_val=200)
            return True
        # If the type is not one of the above, raise an error
        else:
            return False


def check_location(location):
    """ Check that location has the following keys and value datatypes:
        {'longitude': float, 'latitude': float, 'timezone': string,
        'altitude': float}
    """

    if len({'longitude', 'latitude', 'timezone', 'altitude'} - set(location.keys())) > 0:
        return False
    validate_parameter('longitude', location['longitude'],
                       **CONSTRAINTS_DICT['longitude'])
    validate_parameter('latitude', location['latitude'],
                       **CONSTRAINTS_DICT['latitude'])
    validate_parameter('timezone', location['timezone'], custom_args={},
                       **CONSTRAINTS_DICT['timezone'])
    validate_parameter('altitude', location['altitude'],
                       **CONSTRAINTS_DICT['altitude'])
    return True


def check_system_costs(system_costs):
    """ Check that system costs has the following keys:
        {'generator_costs', 'fuel_tank_costs', 'pv_costs', 'battery_costs', 'om_costs'}
        and run checks on each element.
    """

    if len({'generator_costs', 'fuel_tank_costs', 'pv_costs', 'battery_costs',
            'om_costs'} - set(system_costs.keys())) > 0:
        return False

    validate_parameter('generator_costs', system_costs['generator_costs'],
                       custom_args={}, **CONSTRAINTS_DICT['generator_costs'])
    validate_parameter('fuel_tank_costs', system_costs['fuel_tank_costs'],
                       custom_args={}, **CONSTRAINTS_DICT['fuel_tank_costs'])
    validate_parameter('pv_costs', system_costs['pv_costs'],
                       custom_args={}, **CONSTRAINTS_DICT['pv_costs'])
    validate_parameter('battery_costs', system_costs['battery_costs'],
                       custom_args={}, **CONSTRAINTS_DICT['battery_costs'])
    validate_parameter('om_costs', system_costs['om_costs'],
                       custom_args={}, **CONSTRAINTS_DICT['om_costs'])
    return True


def check_generator_costs(generator_costs):
    """ Check that generator costs has the required columns. """

    return not len({'1/4 Load (gal/hr)', '1/2 Load (gal/hr)', '3/4 Load (gal/hr)',
                    'Full Load (gal/hr)', 'Cost (USD)'} -
                   set(generator_costs.columns))


def check_battery_costs(battery_costs):
    """ Check that battery costs have the required columns and acceptable values. """

    if len({'Battery System', 'Inverter'} - set(battery_costs.columns)) > 0:
        return False
    validate_parameter('batt_cost_per_Wh',
                       battery_costs['Battery System'].values[1],
                       **CONSTRAINTS_DICT['batt_cost_per_Wh'])
    validate_parameter('inverter_cost_per_W',
                       battery_costs['Inverter'].values[1],
                       **CONSTRAINTS_DICT['inverter_cost_per_W'])
    return True


def check_pv_costs(pv_costs):
    """ Check that PV costs have the appropriate columns and acceptable values. """

    # Check that column names are numbers
    if not all([isinstance(col, (int, float)) for col in pv_costs.columns]):
        return False

    # Check that price values are acceptable
    for price in np.ravel(pv_costs.iloc[1:].values):
        validate_parameter('pv_cost_per_W', price, **CONSTRAINTS_DICT['pv_cost_per_W'])
    return True


def check_fuel_tank_costs(fuel_tank_costs):
    """ Check that fuel tank costs have the required columns and acceptable values. """

    if len({'Cost (USD)'} - set(fuel_tank_costs.columns)):
        return False

    for size in fuel_tank_costs.index:
        validate_parameter('fuel_tank_size', size, **CONSTRAINTS_DICT['fuel_tank_size'])

    for cost in fuel_tank_costs['Cost (USD)']:
        validate_parameter('fuel_tank_cost', cost, **CONSTRAINTS_DICT['fuel_tank_cost'])
    return True


def check_om_costs(om_costs):
    """ Check that om_costs have the required columns and acceptable values. """

    if len({'Generator_scalar', 'Generator_exp', 'Battery', 'PV_ground;fixed',
            'PV_ground;single_axis', 'PV_roof;fixed', 'PV_carport;fixed'} -
           set(om_costs.columns)):
        return False

    # Check that the om costs have acceptable values
    for cost in om_costs.loc['Cost'].values:
        validate_parameter('om_cost', cost, **CONSTRAINTS_DICT['om_cost'])
    return True


def check_gen_power(gen_power):
    """ Check that gen_power has a DateTimeIndex. """

    return isinstance(gen_power.index, pd.DatetimeIndex)


def check_grouped_load(grouped_load):
    """ Check that grouped load has the necessary columns. """

    return not len({'num_hours', 'binned_load'} - set(grouped_load.columns))


def check_power_profiles(power_profiles, renewable_resources):
    """ Check that each power profile conforms to standards. """

    for key, profiles in power_profiles.items():
        if isinstance(profiles, list):
            for profile in profiles:
                validate_parameter('power_profile', profile, custom_args={},
                                **CONSTRAINTS_DICT['power_profile'])
        else:
            validate_parameter('power_profile', profiles, custom_args={},
                                **CONSTRAINTS_DICT['power_profile'])
        
    # Check that the keys match the items from the renewable_resources list
    if len(set(renewable_resources) - set(power_profiles.keys())):
        return False
    
    # Check that a night profile is included if pv is included
    if 'pv' in power_profiles.keys():
        if 'night' not in power_profiles.keys():
            return False
        
    return True


def check_temp_profiles(temp_profiles):
    """ Check that each temperature profile conforms to standards. """

    for profile in temp_profiles:
        validate_parameter('temp_profile', profile, custom_args={},
                           **CONSTRAINTS_DICT['temp_profile'])
    return True


def check_night_profiles(night_profiles):
    """ Check that each temperature profile conforms to standards. """

    for profile in night_profiles:
        validate_parameter('night_profile', profile, custom_args={},
                           **CONSTRAINTS_DICT['night_profile'])
    return True


def check_pv_params(pv_params):
    """ Check that pv_params has the required keys and value formats. """

    if len({'tilt', 'azimuth', 'module_capacity', 'module_area',
            'advanced_inputs', 'spacing_buffer', 'pv_tracking', 'pv_racking'} -
           set(pv_params.keys())) > 0:
        return False
    validate_parameter('tilt', pv_params['tilt'], **CONSTRAINTS_DICT['tilt'])
    validate_parameter('azimuth', pv_params['azimuth'],
                       **CONSTRAINTS_DICT['azimuth'])
    validate_parameter('module_capacity', pv_params['module_capacity'],
                       **CONSTRAINTS_DICT['module_capacity'])
    validate_parameter('module_area_in2', pv_params['module_area'],
                       **CONSTRAINTS_DICT['module_area_in2'])
    validate_parameter('spacing_buffer', pv_params['spacing_buffer'],
                       **CONSTRAINTS_DICT['spacing_buffer'])
    validate_parameter('pv_tracking', pv_params['pv_tracking'],
                       **CONSTRAINTS_DICT['pv_tracking'])
    validate_parameter('pv_racking', pv_params['pv_racking'],
                       **CONSTRAINTS_DICT['pv_racking'])
    return True


def check_battery_params(battery_params):
    """ Check that battery_params has the required keys and value formats. """

    if len({'battery_power_to_energy', 'initial_soc',
            'one_way_battery_efficiency', 'one_way_inverter_efficiency',
            'soc_upper_limit', 'soc_lower_limit'} -
           set(battery_params.keys())) > 0:
        return False
    validate_parameter('battery_power_to_energy',
                       battery_params['battery_power_to_energy'],
                       **CONSTRAINTS_DICT['battery_power_to_energy'])
    validate_parameter('one_way_battery_efficiency',
                       battery_params['one_way_battery_efficiency'],
                       **CONSTRAINTS_DICT['one_way_battery_efficiency'])
    validate_parameter('one_way_inverter_efficiency',
                       battery_params['one_way_inverter_efficiency'],
                       **CONSTRAINTS_DICT['one_way_inverter_efficiency'])
    validate_parameter('soc_upper_limit', battery_params['soc_upper_limit'],
                       **CONSTRAINTS_DICT['soc_upper_limit'])
    validate_parameter('soc_lower_limit', battery_params['soc_lower_limit'],
                       **CONSTRAINTS_DICT['soc_lower_limit'])

    initial_soc_params = CONSTRAINTS_DICT['initial_soc']
    initial_soc_params['custom_args'] = {'soc_upper_limit': battery_params['soc_upper_limit'],
                                         'soc_lower_limit': battery_params['soc_lower_limit']}
    validate_parameter('initial_soc', battery_params['initial_soc'], **initial_soc_params)
    return True


def check_initial_soc(initial_soc, soc_upper_limit, soc_lower_limit):
    """ Check that the initial SOC is within the upper and lower soc limits. """

    if initial_soc > soc_upper_limit or initial_soc < soc_lower_limit:
        return False
    return True


def check_gen_power_percent(gen_power_percent):
    """ Check that each value in gen load percent is between 0 and 100. """

    for val in gen_power_percent:
        validate_parameter('gen_power_percent', val, data_type=['int', 'float'], min_val=0,
                           max_val=100)
    return True


def check_filter_constraints(filter_constraints):
    """ Check that filter constraints contains allowable elements. """

    if len(filter_constraints):
        for elem in filter_constraints:
            if not isinstance(elem, dict):
                return False
            if len({'parameter', 'type', 'value'} - set(elem.keys())):
                return False
            if elem['parameter'] not in ['capital_cost_usd', 'pv_area_ft2',
                                         'annual_benefits_usd',
                                         'simple_payback_yr', 'pv_capacity',
                                         'fuel_used_gal mean',
                                         'fuel_used_gal most-conservative',
                                         'pv_percent mean',
                                         'gen_percent mean']:
                return False
            if elem['type'] not in ['max', 'min']:
                return False
            if not isinstance(elem['value'], (float, int)):
                return False
    return True


def check_ranking_criteria(ranking_criteria):
    """ Check that filter constraints contains allowable elements. """

    if len(ranking_criteria):
        for elem in ranking_criteria:
            if not isinstance(elem, dict):
                return False
            if len({'parameter', 'order_type'} - set(elem.keys())):
                return False
            if elem['parameter'] not in ['capital_cost_usd',
                                         'annual_benefits_usd',
                                         'simple_payback_yr',
                                         'fuel_used_gal mean',
                                         'fuel_used_gal most-conservative']:
                return False
            if elem['order_type'] not in ['ascending', 'descending']:
                return False
    return True


def check_include_pv(include_pv):
    """ Check that each element of include_pv is valid. """

    for elem in include_pv:
        validate_parameter('pv_capacity', elem, data_type=('int', 'float'), min_val=0)
    return True


def check_include_mre(include_mre):
    """ Check that each element of include_mre is valid. """

    for elem in include_mre:
        validate_parameter('num_turbines', elem, data_type=('int',), min_val=0)
    return True


def check_include_batt(include_batt):
    """ Check that each element of include_batt is valid. """

    for elem in include_batt:
        validate_parameter('battery size', elem, data_type=('tuple',), size=2)
        validate_parameter('battery capacity', elem[0], data_type=('int', 'float'), min_val=0)
        validate_parameter('battery capacity', elem[1], data_type=('int', 'float'), min_val=0)
    return True


# Custom high-level validation
def check_annual_load_profile(annual_load_profile, duration):
    """ Checks that annual_load_profile has a datetime index, has values for every hour (or
            corresponding duration) of the year, and has non-negative values.
    """

    # Check that the index can be converted to datetime
    try:
        converted_index = pd.to_datetime(annual_load_profile.index).map(
            lambda x: x.replace(year=2017))
    except:
        message = 'The annual load profile must have a datetime index, and not contain ' \
                  'leap days.'
        log_error(message)
        raise Exception(message)

    # Check that the index has all of the expected values
    comp_index = pd.date_range(start='1/1/2017', end='1/1/2018',
                               freq='{}S'.format(int(duration)))[:-1]
    if len(set(comp_index).symmetric_difference(set(converted_index))):
        message = 'The annual load profile must begin on January 1 at ' \
                  '00:00:00 and have no missing values.'
        log_error(message)
        return False

    # Check that values are > 0
    if len(annual_load_profile[annual_load_profile < 0]):
        message = 'The annual load profile must not have any negative values.'
        log_error(message)
        return False

    return True


def check_off_grid_load_profile(off_grid_load_profile, duration):
    """ Checks that off_grid_load_profile has a datetime index, has values for every hour (or
            corresponding duration) of the year, and has non-negative values.
    """

    # Check that the index can be converted to datetime
    try:
        converted_index = pd.to_datetime(off_grid_load_profile.index).map(
            lambda x: x.replace(year=2017))
    except:
        message = 'The off-grid load profile must have a datetime index, ' \
                  'and not contain leap days.'
        log_error(message)
        raise Exception(message)

    # Check that the index has all of the expected values
    comp_index = pd.date_range(start='1/1/2017', end='1/1/2018',
                               freq='{}S'.format(int(duration)))[:-1]
    if len(set(comp_index).symmetric_difference(set(converted_index))):
        message = 'The off-grid load profile must begin on January 1 at ' \
                  '00:00:00 and have no missing values.'
        log_error(message)
        return False

    # Check that values are > 0
    if len(off_grid_load_profile[off_grid_load_profile < 0]):
        message = 'The off-grid load profile must not have any negative values.'
        log_error(message)
        return False

    return True


def check_annual_production(annual_production):
    """ Checks that annual_production has a datetime index, has values for every hour of the
            year, and has non-negative values.
    """

    # Check that the index can be converted to datetime
    try:
        converted_index = pd.to_datetime(annual_production.index).map(
            lambda x: x.replace(year=2017))
    except:
        message = 'The annual production profile must have a datetime index,' \
                  ' and not contain leap days.'
        log_error(message)
        raise Exception(message)

    # Check that the index has all of the expected values
    comp_index = pd.date_range(start='1/1/2017', end='1/1/2018', freq='H')[:-1]
    if len(set(comp_index).symmetric_difference(set(converted_index))):
        message = 'The annual production profile must begin on January 1 at ' \
                  '00:00:00 and have no missing values.'
        log_error(message)
        return False

    # Check that values are > 0
    if len(annual_production[annual_production < 0]):
        message = 'The annual production profile must not have any negative values.'
        log_error(message)
        return False

    return True


def check_temperature(temperature):
    """ Checks that temperature profile has a datetime index, has values for every hour of the
            year, and has non-negative values.
    """

    # Check that the index can be converted to datetime
    try:
        converted_index = pd.to_datetime(temperature.index).map(
            lambda x: x.replace(year=2017))
    except:
        message = 'The annual temperature profile must have a datetime index,'\
                  ' and not contain leap days.'
        log_error(message)
        raise Exception(message)

    # Check that the index has all of the expected values
    comp_index = pd.date_range(start='1/1/2017', end='1/1/2018', freq='H')[:-1]
    if len(set(comp_index).symmetric_difference(set(converted_index))):
        message = 'The annual temperature profile must begin on January 1 at '\
                  '00:00:00 and have no missing values.'
        log_error(message)
        return False

    # Check that values are > 0
    if len(temperature[temperature < 0]):
        message = 'The annual production profile must not have any negative values.'
        log_error(message)
        return False

    return True


def check_load_profile(load_profile):
    """ Checks that load profile has a datetime index and non-negative values. """

    # Check that it has a datetime index
    if not isinstance(load_profile.index, pd.core.indexes.datetimes.DatetimeIndex):
        return False

    # Check that all values are positive
    if len(load_profile[load_profile < 0]):
        return False

    return True


def check_spg_advanced_inputs(advanced_inputs):
    """Checks that each field in advanced inputs is valid. """

    # Validate each of the fields in advanced_inputs
    for key, val in advanced_inputs.items():
        # Check that the field has a validation function
        try:
            kwargs = CONSTRAINTS_DICT[key]
        except KeyError:
            print("{} does not have a validation function. Skipping "
                  "validation...".format(key))
        else:
            # If the field has a custom function, include custom args
            if 'custom_func' in kwargs.keys():
                if 'custom_args' in kwargs.keys() and len(kwargs['custom_args']):
                    # Note: the following line requires that if a custom validation function
                    #   for a parameter in advanced_inputs has custom arguments then those
                    #   arguments must also be included in the advanced_inputs dictionary.
                    kwargs['custom_args'] = \
                        {key: advanced_inputs[key] for key in
                         kwargs['custom_args'].split(',')}
                else:
                    kwargs['custom_args'] = {}

            validate_parameter(key, val, **kwargs)
    return True

def check_tpg_advanced_inputs(advanced_inputs):
    """Checks that each field in advanced inputs is valid. """

    # Validate each of the fields in advanced_inputs
    for key, val in advanced_inputs.items():
        # Check that the field has a validation function
        try:
            kwargs = CONSTRAINTS_DICT[key]
        except KeyError:
            print("{} does not have a validation function. Skipping "
                  "validation...".format(key))
        else:
            # If the field has a custom function, include custom args
            if 'custom_func' in kwargs.keys():
                if 'custom_args' in kwargs.keys() and len(kwargs['custom_args']):
                    # Note: the following line requires that if a custom validation function
                    #   for a parameter in advanced_inputs has custom arguments then those
                    #   arguments must also be included in the advanced_inputs dictionary.
                    kwargs['custom_args'] = \
                        {key: advanced_inputs[key] for key in
                         kwargs['custom_args'].split(',')}
                else:
                    kwargs['custom_args'] = {}

            validate_parameter(key, val, **kwargs)
    return True


def check_module(module):
    """ Checks that all of the required fields are included, and that they have valid values.
    """

    # Check that the required fields are included
    if len({'database', 'model', 'capacity', 'area_in2'} - set(module.keys())):
        return False

    # Run validation on each field
    validate_parameter('module_database', module['database'],
                       **CONSTRAINTS_DICT['module_database'])
    validate_parameter('module_capacity', module['capacity'],
                       **CONSTRAINTS_DICT['module_capacity'])
    validate_parameter('module_area_in2', module['area_in2'],
                       **CONSTRAINTS_DICT['module_area_in2'])

    module_name_params = CONSTRAINTS_DICT['module_name']
    module_name_params['custom_args'] = {'module_database': module['database']}
    validate_parameter('module_name', module['model'], **module_name_params)
    return True


def check_module_name(module_name, module_database):
    """ Checks that the module name can be found in the database. """

    try:
        module_list = pvsystem.retrieve_sam(module_database)
        module = module_list[module_name]
    except (ValueError, KeyError):
        return False
    else:
        return True


def check_inverter(inverter):
    """ Checks that all of the required fields are included, and that they have valid values.
    """

    # Check that the required fields are included
    if len({'database', 'model'} - set(inverter.keys())):
        return False

    # Run validation on each field
    validate_parameter('inverter_database', inverter['database'],
                       **CONSTRAINTS_DICT['inverter_database'])
    inverter_name_params = CONSTRAINTS_DICT['inverter_name']
    inverter_name_params['custom_args'] = {'inverter_database':
                                           inverter['database']}
    validate_parameter('inverter_name', inverter['model'], **inverter_name_params)

    return True


def check_inverter_name(inverter_name, inverter_database):
    """ Checks that the inverter name can be found in the database. """

    try:
        inverter_list = pvsystem.retrieve_sam(inverter_database)
        inverter = inverter_list[inverter_name]
    except (ValueError, KeyError):
        return False
    else:
        return True


def check_cld_hours(cloud_hours):
    """ Checks that the elements of cloud hours are between 0 and 23. """

    for elem in cloud_hours:
        if elem < 0 or elem > 23:
            return False
    return True


def check_start_datetimes(start_datetimes, num_trials, start_year, end_year):
    """Checks that start_datetimes is the same length as num_trials, each element is a
        datetime object, and each element is within the bounds of the start and end years.
    """

    if not len(start_datetimes) == num_trials:
        return False
    for elem in start_datetimes:
        if not isinstance(elem, dt.datetime):
            return False
        if elem.year < start_year or elem.year > end_year:
            return False
    return True


def check_timezone(timezone):
    """Checks that the timezone is a valid pytz timezone. """

    if timezone in pytz.all_timezones:
        return True
    return False


def check_demand_rate_list(demand_rate_list):
    """ Checks that the demand rate is in the form of a number or a list with 12 elements and
        each element has appropriate values.
    """

    # If it is a list, check that there are 12 elements
    if isinstance(demand_rate_list, list):
        if len(demand_rate_list) != 12:
            return False

    # If it is not a list, add it to one to check elements
    else:
        demand_rate_list = [demand_rate_list]

    # Check all elements of demand rate list
    for elem in demand_rate_list:
        validate_parameter('demand_rate', elem, **CONSTRAINTS_DICT['demand_rate'])
    return True


def check_solar_source(solar_source, start_year, end_year):
    """ Checks that the years specified are covered in the specified source for the solar
            data.
    """

    if (solar_source == 'himawari') and \
            (not {start_year, end_year}.issubset(set(range(2016, 2021)))):
        message = 'Himawari dataset only covers years 2016-2020. ' \
                  'Please check the start/end years.'
        log_error(message)
        return False

    if (solar_source == 'nsrdb') and \
            (not {start_year, end_year}.issubset(set(range(1998, 2023)))):
        message = "NREL's NSRDB dataset only covers years 1998-2022. "\
                  'Please check the start/end years.'
        log_error(message)
        return False

    return True


def check_renewable_resources(renewable_resources):
    """ Checks that renewables resources contains one or more of the following strings: 
        'pv', 'mre' """
    
    if not len(renewable_resources):
        return False
    if len(set(renewable_resources) - set(['pv', 'mre'])):
        return False
    return True


def check_mre_params(mre_params):
    return True


# Parameter warning functions
def annual_load_profile_warnings(annual_load_profile):
    """ Checks to make sure the annual load profile looks realistic and outputs a warning if
            not.
    """

    # Check for 0's in the load profile
    zeros = annual_load_profile[annual_load_profile <= 0]

    # Look for hourly and daily outliers (for daily: sum, min, max) using Chauvenet's
    #   criterion
    hourly_outliers = chauvenet_outliers(annual_load_profile)
    daily_sum_outliers = chauvenet_outliers(annual_load_profile.resample('D').sum())
    daily_min_outliers = chauvenet_outliers(annual_load_profile.resample('D').min())
    daily_max_outliers = chauvenet_outliers(annual_load_profile.resample('D').max())

    # Check frequencies of data
    # Calculate load profile periodogram
    freq, power = periodogram(annual_load_profile.values, fs=1/3600)
    pg = pd.DataFrame(power, index=freq, columns=['power'])

    # Get the peak frequency
    peak_freq = pg[pg['power'] == max(pg['power'])].index[0]
    min_period = 1 / peak_freq / 3600

    # Get the next highest frequency
    pg_crop = pg[:peak_freq].iloc[5:-5]
    if len(pg_crop):
        second_freq = pg_crop[pg_crop['power'] == max(pg_crop['power'])].index[0]
        second_period = 1 / second_freq / 3600
    else:
        second_period = 0

    # Is daytime load (workday) higher than nighttime load
    daytime = annual_load_profile[(annual_load_profile.index.hour >= 9) &
                                  (annual_load_profile.index.hour <= 17
                                   )].median()
    nighttime = annual_load_profile[(annual_load_profile.index.hour <= 7) |
                                    (annual_load_profile.index.hour >= 18
                                     )].median()

    # Compile any warnings
    warning_message = ''
    if len(zeros):
        warning_message += 'There are zeros at the following times: {}. ' \
                           ''.format(list(zeros.index))
    if len(hourly_outliers):
        warning_message += 'There are suspicious values for the following ' \
                           'hours: {}. '.format(list(hourly_outliers.index))
    if len(daily_sum_outliers) or len(daily_max_outliers) or len(daily_min_outliers):
        warning_message += 'There are suspicious values for the following ' \
                           'days: {}. '.format(
                                set(pd.concat([daily_sum_outliers,
                                               daily_min_outliers,
                                               daily_max_outliers]).index))
    if int(min_period) not in [23, 24, 25] or int(second_period) not in range(165, 172):
        warning_message += 'The load profile does not have natural periods ' \
                           'at 24 hours and/or 1 week. '
    if daytime < nighttime:
        warning_message += 'The average daytime (working hours) load is ' \
                           'less than the average nighttime load.'

    if len(warning_message):
        warning_message = 'Warning: There are one or more potential issues ' \
                          'with the annual load profile: {}' \
                          ''.format(warning_message)
        print(warning_message)
        log_error(warning_message)


def normalize_profile(annual_load_profile):
    """ Normalizes a load profile by daily, weekly and seasonal trends. """

    # Decompose profile into daily, weekly, seasonal trends
    # Normalized monthly trend
    load_monthly = annual_load_profile.resample('M').median()
    load_monthly.index = load_monthly.index.month
    load_season_norm = annual_load_profile.reset_index().apply(
        lambda x: x[annual_load_profile.name] - load_monthly[x[
            annual_load_profile.index.name].month], axis=1)
    load_season_norm.index = annual_load_profile.index

    # Normalize weekly trend
    load_season_norm = load_season_norm.to_frame(name='load').reset_index()
    load_season_norm['dow'] = \
        load_season_norm[annual_load_profile.index.name].apply(lambda x: x.dayofweek)
    load_dow = load_season_norm.groupby('dow')['load'].median()
    load_season_weekly_norm = annual_load_profile.reset_index().apply(
        lambda x: x[annual_load_profile.name] -
        load_monthly[x[annual_load_profile.index.name].month] -
        load_dow[x[annual_load_profile.index.name].dayofweek], axis=1)
    load_season_weekly_norm.index = annual_load_profile.index

    # Normalize hourly trend
    load_season_weekly_norm = load_season_weekly_norm.to_frame(name='load').reset_index()
    load_season_weekly_norm['hour'] = load_season_weekly_norm[
        annual_load_profile.index.name].apply(lambda x: x.hour)
    load_hourly = load_season_weekly_norm.groupby('hour')['load'].median()
    load_normalized = annual_load_profile.reset_index().apply(
        lambda x: x[annual_load_profile.name] -
        load_monthly[x[annual_load_profile.index.name].month] -
        load_dow[x[annual_load_profile.index.name].dayofweek] -
        load_hourly[x[annual_load_profile.index.name].hour], axis=1)
    load_normalized.index = annual_load_profile.index

    return load_normalized


def chauvenet_outliers(data):
    """
        Given a Pandas Series of time series data, returns any data points that are considered
            outliers according to Chauvenet's criterion:
            P(data point) < 1/(2n), where n is the length of the series
    """

    # Create a dataframe to hold the data
    data_df = data.to_frame(name='data')

    # Calculate the mean and standard deviation of the distribution
    avg = data_df['data'].mean()
    std = data_df['data'].std()

    # For each data point, calculate the likelihood that that value is drawn
    # from the underlying distribution
    data_df['num_stds'] = data_df['data'].apply(lambda x: np.abs((x-avg)/std))
    data_df['prob'] = data_df['num_stds'].apply(lambda x: (1-norm.cdf(x))*2)

    # Determine if each point is an outlier
    data_df['outlier'] = data_df['prob'].apply(lambda x: x < 1/(2*len(data_df)))

    return data_df[data_df['outlier']]


def strings_warnings(strings, module_name, inverter_name):
    """ Checks to make sure strings configuration is realistic and consistent with inverter
            parameters and outputs a warning if not.
    """

    # Retrieve info on the module and inverter
    module_list = pvsystem.retrieve_sam(module_name['database'])
    module = module_list[module_name['model']]
    inverter_list = pvsystem.retrieve_sam(inverter_name['database'])
    inverter = inverter_list[inverter_name['model']]

    # Check that the DC/AC ratio is within a reasonable range
    string_warnings = ''
    dc = module_name['capacity'] * strings['mods_per_string'] * 1000 \
        * strings['strings_per_inv']
    ac = inverter['Paco']
    if not 1 <= dc/ac <= 1.1:
        string_warnings += 'The DC/AC ratio for your PV system is {:.2f}. We recommend a ' \
                           'value between 1 and 1.1.'.format(dc/ac)

    # Check string parameters against inverter voltage and current
    system_voltage = module['V_oc_ref'] * strings['mods_per_string']
    system_current = module['I_sc_ref'] * strings['strings_per_inv']
    if system_voltage > inverter['Vdcmax']:
        string_warnings += 'The system voltage ({:.2f}V) exceeds the ' \
                           'maximum inverter voltage ({:.2f}V). ' \
                           'Consider decreasing the number of panels ' \
                           'in series or choosing a different ' \
                           'inverter. '.format(system_voltage, inverter['Vdcmax'])

    if system_current > inverter['Idcmax']:
        string_warnings += 'The system current ({:.1f}A) exceeds the ' \
                           'maximum inverter current ({:.1f}A). ' \
                           'Consider decreasing the number of strings ' \
                           'in parallel or choosing a different ' \
                           'inverter. '.format(system_current, inverter['Idcmax'])

    if len(string_warnings):
        string_warnings = 'Warning: You have one or more issues with your ' \
                          'string sizing: {}'.format(string_warnings)
        print(string_warnings)
        log_error(string_warnings)


def location_warnings(latitude, longitude, timezone):
    """ Checks to make sure the location and timezone are consistent. """

    # Determines proper timezone based on location

    # Checks against actual timezone

    pass


VALIDATION_FUNCS = {'check_path': check_path,
                    'check_sitename': check_sitename,
                    'check_solar_profile': check_solar_profile,
                    'check_temp_profile': check_temp_profile,
                    'check_strings': check_strings,
                    'check_night_duration': check_night_duration,
                    'check_fuel_curve_model': check_fuel_curve_model,
                    'check_outputs': check_outputs,
                    'check_existing_components': check_existing_components,
                    'check_annual_load_profile': check_annual_load_profile,
                    'check_net_metering_limits': check_net_metering_limits,
                    'check_location': check_location,
                    'check_night_profile': check_night_profile,
                    'check_generator_costs': check_generator_costs,
                    'check_gen_power': check_gen_power,
                    'check_grouped_load': check_grouped_load,
                    'check_power_profile': check_power_profile,
                    'check_power_profiles': check_power_profiles,
                    'check_temp_profiles': check_temp_profiles,
                    'check_night_profiles': check_night_profiles,
                    'check_pv_params': check_pv_params,
                    'check_battery_params': check_battery_params,
                    'check_gen_power_percent': check_gen_power_percent,
                    'check_filter_constraints': check_filter_constraints,
                    'check_ranking_criteria': check_ranking_criteria,
                    'check_include_pv': check_include_pv,
                    'check_include_batt': check_include_batt,
                    'check_include_mre': check_include_mre,
                    'check_load_profile': check_load_profile,
                    'check_spg_advanced_inputs': check_spg_advanced_inputs,
                    'check_tpg_advanced_inputs': check_tpg_advanced_inputs,
                    'check_module': check_module,
                    'check_inverter': check_inverter,
                    'check_module_name': check_module_name,
                    'check_system_costs': check_system_costs,
                    'check_battery_costs': check_battery_costs,
                    'check_pv_costs': check_pv_costs,
                    'check_fuel_tank_costs': check_fuel_tank_costs,
                    'check_om_costs': check_om_costs,
                    'check_inverter_name': check_inverter_name,
                    'check_cld_hours': check_cld_hours,
                    'check_start_datetimes': check_start_datetimes,
                    'check_timezone': check_timezone,
                    'check_annual_production': check_annual_production,
                    'check_temperature': check_temperature,
                    'check_off_grid_load_profile': check_off_grid_load_profile,
                    'check_demand_rate_list': check_demand_rate_list,
                    'check_initial_soc': check_initial_soc,
                    'check_solar_source': check_solar_source,
                    'check_renewable_resources': check_renewable_resources,
                    'check_mre_params': check_mre_params}
