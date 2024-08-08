# -*- coding: utf-8 -*-
"""
Runs a sensitivity analysis with MCOR, varying one or more input parameters.
"""
import os
import pickle
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from generate_solar_profile import SolarProfileGenerator
from generate_tidal_profile import TidalProfileGenerator
from microgrid_optimizer import GridSearchOptimizer
from config import DATA_DIR
from config import OUTPUT_DIR


# Parameters that require re-running resource models
GET_SOLAR_DATA_PARAMS = ['latitude', 'longitude']
GET_SOLAR_PROFILES_PARAMS = ['latitude', 'longitude', 'timezone', 'num_trials', 'length_trials', 
                             'start_datetimes']
GET_TIDAL_PROFILE_PARAMS = ['latitude', 'longitude', 'timezone', 'start_year', 'end_year', 'depth',
                            'num_trials', 'length_trials', 'start_datetimes']
ALLOWED_SENSITIVITY_PARAMS = ['latitude', 'longitude', 'timezone', 'num_trials', 'length_trials', 
                             'start_datetimes', 'dispatch_strategy', 'size_resources_based_on_tmy', 
                             're_constraints', 'tilt', 'azimuth', 'pv_racking', 'pv_tracking',
                             'tidal_inverter_efficiency', 'tidal_turbine_losses', 'depth',
                             'battery_power_to_energy', 'initial_soc', 'one_way_battery_efficiency',
                             'one_way_inverter_efficiency', 'soc_upper_limit', 'soc_lower_limit',
                             'batt_sizing_method', 'percent_at_night']

def create_spg_object(system_inputs, pv_inputs, warning_inputs, multithreading_inputs):
    spg = SolarProfileGenerator(system_inputs['latitude'], system_inputs['longitude'], system_inputs['timezone'],
                                    system_inputs['altitude'], pv_inputs['tilt'], pv_inputs['azimuth'],
                                    float(system_inputs['num_trials']), float(system_inputs['length_trials']),
                                    start_year=pv_inputs['solar_data_start_year'],
                                    end_year=pv_inputs['solar_data_end_year'],
                                    pv_racking=pv_inputs['pv_racking'], pv_tracking=pv_inputs['pv_tracking'],
                                    suppress_warnings=warning_inputs['suppress_warnings'],
                                    multithreading=multithreading_inputs['multithreading'],
                                    solar_source=pv_inputs['solar_data_source'])
    return spg


def create_tpg_object(system_inputs, data_start_year, data_end_year, mre_inputs):
    tpg = TidalProfileGenerator(mre_inputs['marine_data_filename'], system_inputs['latitude'], system_inputs['longitude'], system_inputs['timezone'],float(system_inputs['num_trials']),
                                float(system_inputs['length_trials']), mre_inputs['tidal_turbine_rated_power'], float(mre_inputs['depth']), mre_inputs['tidal_rotor_radius'],
                                mre_inputs['tidal_rotor_number'], mre_inputs['maximum_cp'], mre_inputs['tidal_cut_in_velocity'], mre_inputs['tidal_cut_out_velocity'],
                                mre_inputs['tidal_inverter_efficiency'], mre_inputs['tidal_turbine_losses'], data_start_year, data_end_year)
    return tpg


def get_solar_data(spg):
    spg.get_solar_data()
    return spg

def build_solar_model(spg, system_inputs):
    spg.get_solar_profiles(system_inputs['start_datetimes'])
    return spg


def build_mre_model(tpg, start_datetimes):
    tpg.tidal_profiles = []
    tpg.power_profiles = []
    tpg.get_tidal_data_from_upload()
    tpg.extrapolate_tidal_epoch()
    tpg.generate_tidal_profiles(start_datetimes)
    return tpg


def run_mcor_iteration(spg, tpg, input_dict):
    # Get power profiles from solar and mre objects
    power_profiles = {}
    tmy_solar = None
    pv_params = None
    tmy_mre = None
    mre_params = None
    if 'pv' in input_dict['system_inputs']['renewable_resources']:
        spg.get_power_profiles()
        spg.get_night_duration(percent_at_night=input_dict['battery_inputs']['percent_at_night'])
        tmy_solar = spg.tmy_power_profile
        module_params = spg.get_pv_params()
        power_profiles['pv'] = spg.power_profiles
        power_profiles['night'] = spg.night_profiles
        pv_params = {'tilt': input_dict['pv_inputs']['tilt'], 
                     'azimuth': input_dict['pv_inputs']['azimuth'],
                     'module_capacity': module_params['capacity'],
                     'module_area': module_params['area_in2'],
                     'spacing_buffer': input_dict['pv_inputs']['spacing_buffer'], 'advanced_inputs': {},
                 'pv_racking': input_dict['pv_inputs']['pv_racking'], 'pv_tracking': input_dict['pv_inputs']['pv_tracking']}
    if 'mre' in input_dict['system_inputs']['renewable_resources']:
        tpg.get_power_profiles()
        tmy_mre = tpg.tmy_tidal
        mre_params = {'generator_type': 'tidal', 
                      'generator_capacity': input_dict['mre_inputs']['tidal_turbine_rated_power'],
                      'device_name': input_dict['mre_inputs']['device_name']}
        power_profiles['mre'] = tpg.power_profiles
    
    # Set up parameter dictionaries
    location = {'longitude': input_dict['system_inputs']['longitude'], 
                'latitude': input_dict['system_inputs']['latitude'],
                'timezone': input_dict['system_inputs']['timezone'], 
                'altitude': input_dict['system_inputs']['altitude']}
    
    battery_params = {'battery_power_to_energy': input_dict['battery_inputs']['battery_power_to_energy'],
                      'initial_soc': input_dict['battery_inputs']['initial_soc'],
                      'one_way_battery_efficiency': input_dict['battery_inputs']['one_way_battery_efficiency'],
                      'one_way_inverter_efficiency': input_dict['battery_inputs']['one_way_inverter_efficiency'],
                      'soc_upper_limit': input_dict['battery_inputs']['soc_upper_limit'],
                      'soc_lower_limit': input_dict['battery_inputs']['soc_lower_limit']}
    optim = GridSearchOptimizer(input_dict['system_inputs']['renewable_resources'], power_profiles, 
                                input_dict['load_inputs']['annual_load_profile'],
                                location, battery_params, 
                                input_dict['financial_inputs']['system_costs'],
                                input_dict['system_inputs']['re_constraints'],
                                tmy_solar=tmy_solar, pv_params=pv_params, tmy_mre=tmy_mre,
                                mre_params=mre_params, 
                                size_resources_based_on_tmy=input_dict['system_inputs']['size_resources_based_on_tmy'],
                                size_resources_with_battery_eff_term=input_dict['system_inputs']['size_resources_with_battery_eff_term'],
                                dispatch_strategy=input_dict['system_inputs']['dispatch_strategy'],
                                electricity_rate=input_dict['financial_inputs']['utility_rate'],
                                net_metering_rate=input_dict['net_metering_inputs']['net_metering_rate'],
                                demand_rate=input_dict['financial_inputs']['demand_rate'],
                                existing_components=input_dict['sizing_inputs']['existing_components'],
                                output_tmy=True, validate=True,
                                net_metering_limits=input_dict['net_metering_inputs']['net_metering_limits'],
                                off_grid_load_profile=input_dict['load_inputs']['off_grid_load_profile'],
                                batt_sizing_method=input_dict['battery_inputs']['batt_sizing_method'])
    optim.get_load_profiles()
    optim.define_grid(include_pv=input_dict['sizing_inputs']['include_pv'],
                      include_batt=input_dict['sizing_inputs']['include_batt'],
                      include_mre=input_dict['sizing_inputs']['include_mre'])
    optim.run_sims()
    optim.parse_results()
    return optim, spg, tpg

def plot_comparison_graphs(output_dict, sensitivity_param, comparison_param, system_label, sim_label):
    '''
    Plot comparison graph across iterations and systems

    Parameters
    ----------

    output_dict: dictionary of Optimizer objects
    sensitivity_param: dictionary with info on sensitivity param
    comparison_param: name of parameter to be compared across iterations. Must be the name of a column from the Optimizer results_grid.
    system_label: indictates which system to reference, options include: [least_fuel, least_cost, pv_only, mre_only, most_diversified]
    sim_label: indicates which simulation to reference, options include: [avg, max, min, distribution]

    '''

    # Get comparison param across iterations
    comparison_data = {}
    for iter_name, iter_data in output_dict.items():
        iter_val = iter_name.split('_')[-1]
        comparison_data[iter_val] = iter_data.get_param_value(comparison_param, system_label, sim_label)
        if comparison_data[iter_val] is None:
            comparison_data[iter_val] = np.nan

    # Plot comparison param across iterations
    plt.figure()
    # If single value
    if sim_label in ['avg', 'max', 'min']:
        sns.scatterplot(comparison_data)
        plt.xlabel(sensitivity_param['param_name'])
        plt.ylabel(comparison_param)

    # If distribution
    if sim_label == 'distribution':
        sns.boxplot(comparison_data)
        plt.xlabel(sensitivity_param['param_name'])
        plt.ylabel(comparison_param)

def save_comparison_json(output_dict, sensitivity_param, comparison_param, system_label, sim_label):
    # Get comparison param across iterations
    comparison_data = {}
    for iter_name, iter_data in output_dict.items():
        comparison_data[iter_name] = iter_data.get_param_value(comparison_param, system_label, sim_label)
        if comparison_data[iter_name] is None:
            print('Not able to save metrics, check inputs')
            return
    with open(os.path.join(OUTPUT_DIR, f'{sensitivity_param["param_name"]}_{comparison_param}_{system_label}_system_{sim_label}_sim_comparison.json'), 'w') as f:
        json.dump(comparison_data, f)


if __name__ == "__main__":
    # Define varying parameter
    # TODO - should we constrain which parameters can be varying? how many params can be varied at a time?
    sensitivity_param = {
        'param_category': 'mre_inputs',
        'param_name': 'depth',
        'param_values': [5, 10, 20]
    }

    if sensitivity_param['param_name'] not in ALLOWED_SENSITIVITY_PARAMS:
        raise Exception(f'{sensitivity_param["param_name"]} is not an allowed sensitivity parameter!')
    
    # Define static parameters
    # Note: to make it easy to swap out the sensitivity param, the static params can include the
    #   param to be varied and the sensitivity_param values will overwrite the static param value
    input_dict = {}

    # System level dictionary
    input_dict['system_inputs'] = {
        'latitude': 46.34,
        'longitude': -119.28,
        'timezone': 'US/Pacific',
        'altitude': 0,
        'num_trials': 200,
        'length_trials': 14 * 24,
        'renewable_resources': ['mre', 'pv'], # Can include 'pv' and/or 'mre', in order of dispatch',
        'dispatch_strategy': 'available_capacity',
        'size_resources_based_on_tmy': True,
        'size_resources_with_battery_eff_term': True,
        'start_datetimes': None,  # If you want to specify specific times to start the scenarios,
        're_constraints': {}  # {'total': 2000, 'pv': 2000, 'mre': 2000} Any sizing constraints for the RE system, in kW, can include keys: 'total', 'pv', or 'mre'
    }

    # PV dictionary
    input_dict['pv_inputs'] = {
        'tilt': 20,
        'azimuth': 180.,
        'spacing_buffer': 2,
        'pv_racking': 'ground',
        'pv_tracking': 'fixed',
        'solar_data_source': 'nsrdb',
        'solar_data_start_year': 1998,
        'solar_data_end_year': 2022,
        'get_solar_data': True,
        'get_solar_profiles': True
    }

    device_costs = pd.read_excel(os.path.join(DATA_DIR, 'MCOR Prices.xlsx'), sheet_name='mre_costs', index_col=0)
    device_name = "RM1"

    # MRE dictionary
    input_dict['mre_inputs'] =  {
        'marine_data_filename' : 'PortAngeles_2015_alldepths.csv',
        'generator_type': 'tidal',
        'device_name': device_name,
        'tidal_turbine_rated_power': int(device_costs.loc[device_name, 'Rated Power (kW)']),
        'tidal_rotor_radius': int(device_costs.loc[device_name, 'Rotor Diameter (m)']/2),
        'tidal_rotor_number': int(device_costs.loc[device_name, 'Rotors per Turbine']),
        'depth': 10,
        'maximum_cp': 0.42,
        'tidal_cut_in_velocity': 0.5,
        'tidal_cut_out_velocity': 3,
        'tidal_inverter_efficiency': 0.9,
        'tidal_turbine_losses': 10,
        'get_tidal_profiles': True
    }

    # Battery dictionary
    input_dict['battery_inputs'] = {
        'battery_power_to_energy': 0.25,
        'initial_soc': 1,
        'one_way_battery_efficiency': 0.9,
        'one_way_inverter_efficiency': 0.95,
        'soc_upper_limit': 1,
        'soc_lower_limit': 0.2,
        'batt_sizing_method': 'longest_night',
        'percent_at_night': 0.1
    }

    # Load (in kW) dictionary
    input_dict['load_inputs'] = {
        'annual_load_profile': pd.read_csv(os.path.join(DATA_DIR, 'sample_load_profile.csv'), 
                                           index_col=0)['Load'],
        'off_grid_load_profile': None
    }

    # Project financials
    input_dict['financial_inputs'] = {
        'utility_rate': 0.03263,                 # Utility rate in $/kWh dictionary
        'demand_rate': None,                     # Demand rate in $/kW (optional) dictionary
        'system_costs': pd.read_excel(os.path.join(DATA_DIR, 'MCOR Prices.xlsx'), 
                                      sheet_name=None, index_col=0)
    }

    # Determine if asp multithreading should be used dictionary
    input_dict['multithreading_inputs'] = {}
    input_dict['multithreading_inputs']['multithreading'] = False

    # Post-processing inputs
    input_dict['post_processing_inputs'] = {
        'filtering_constraints': [],
        'ranking_criteria': [{'parameter': 'simple_payback_yr', 'order_type': 'ascending'}],
        'dispatch_plot_scenario_criteria': 'pv',
        'dispatch_plot_scenario_num': None
    }

    # Sizing info dictionary
    input_dict['sizing_inputs'] = {
        'existing_components': {},
        'include_pv': (),  # units of kW
        'include_mre': (),  # units of number of turbines
        'include_batt': ()  # units of (kWh, kW)
    }

    # Net-metering options dictionary
    input_dict['net_metering_inputs'] = {
        'net_metering_limits': None,
        'net_metering_rate': None
    }

    # Other SolarProfileGenerator inputs dictionary
    input_dict['warning_inputs'] = {
        'suppress_warnings': False
    }

    # Set up dictionary to store outputs
    output_dict = {}

    # Create dummy vars for sim objects
    spg = None
    tpg = None

    # Run MCOR iteratively
    for i, param_value in enumerate(sensitivity_param['param_values']):
        # Print out iteration info
        print(f'Iteration: {i} of {len(sensitivity_param["param_values"])}, \
              {sensitivity_param["param_name"]} = {param_value}')
        
        # Set value for sensitivity param
        input_dict[sensitivity_param['param_category']][sensitivity_param['param_name']] = \
            sensitivity_param['param_values'][0]
        
        # If first iteration, create spg and tpg objects
        if i == 0:
            if 'pv' in input_dict['system_inputs']['renewable_resources']:
                spg = create_spg_object(input_dict['system_inputs'], input_dict['pv_inputs'], 
                                        input_dict['warning_inputs'], 
                                        input_dict['multithreading_inputs'])
            if 'mre' in input_dict['system_inputs']['renewable_resources']:
                if 'pv' in input_dict['system_inputs']['renewable_resources']:
                    data_start_year = input_dict['pv_inputs']['solar_data_start_year']
                    data_end_year = input_dict['pv_inputs']['solar_data_end_year']
                else:
                    data_start_year = None
                    data_end_year = None
                tpg = create_tpg_object(input_dict['system_inputs'], data_start_year, 
                                        data_end_year, input_dict['mre_inputs'])

        # If it's the first iteration and get_solar_data/get_solar_profiles/get_tidal_profiles
        #   is set to true or the sensitivity param requires re-running resource models, 
        #   then run those now
        if 'pv' in input_dict['system_inputs']['renewable_resources'] and \
                ((i == 0 and input_dict['pv_inputs']['get_solar_data']) 
                 or sensitivity_param['param_name'] in GET_SOLAR_DATA_PARAMS):
            spg.__setattr__(sensitivity_param['param_name'], param_value)
            spg = get_solar_data(spg)
        if 'pv' in input_dict['system_inputs']['renewable_resources'] and \
                ((i == 0 and input_dict['pv_inputs']['get_solar_profiles']) 
                 or (sensitivity_param['param_name'] in GET_SOLAR_PROFILES_PARAMS 
                 and input_dict['pv_inputs']['get_solar_profiles'])):
            spg.__setattr__(sensitivity_param['param_name'], param_value)
            spg = build_solar_model(spg, input_dict['system_inputs'])
        if 'mre' in input_dict['system_inputs']['renewable_resources'] and \
                ((i == 0 and input_dict['mre_inputs']['get_tidal_profiles'])
                 or sensitivity_param['param_name'] in GET_TIDAL_PROFILE_PARAMS):
            tpg.__setattr__(sensitivity_param['param_name'], param_value)
            if 'pv' in input_dict['system_inputs']['renewable_resources']:
                # Check to see if spg start_datetimes list has been intialized
                if not len(spg.start_datetimes) or sensitivity_param['param_name'] in GET_SOLAR_PROFILES_PARAMS:
                    spg.__setattr__(sensitivity_param['param_name'], param_value)
                    spg.get_power_profiles()
                start_datetimes = spg.start_datetimes
            else:
                start_datetimes = input_dict['system_inputs']['start_datetimes']
            tpg = build_mre_model(tpg, start_datetimes)

        # Even if the resource models don't need to be re-generated, update with the new param if
        #   applicable
        if spg is not None and (sensitivity_param['param_category'] == 'pv_inputs' or sensitivity_param['param_name'] in GET_SOLAR_PROFILES_PARAMS):
            spg.__setattr__(sensitivity_param['param_name'], param_value)
        if tpg is not None and (sensitivity_param['param_category'] == 'mre_inputs' or sensitivity_param['param_name'] in GET_TIDAL_PROFILE_PARAMS):
            tpg.__setattr__(sensitivity_param['param_name'], param_value)
            
        # Run simulation
        optim, spg, tpg = run_mcor_iteration(spg, tpg, input_dict)

        # Store outputs
        output_dict[f'{sensitivity_param["param_name"]}_{param_value}'] = optim

    # Aggregate and compare outputs
    # Create output excel spreadsheet which includes sheets for each iteration
    save_filename = 'sensitivity_run'
    optim.save_results_to_file(spg, tpg, filename=save_filename, sensitivity_data=output_dict)
        
    # Save output_dict to a pickle file
    pickle.dump(output_dict, open(os.path.join(OUTPUT_DIR, f'{save_filename}.pkl'), 'wb'))

    # Plot dispatch for across iterations
    for iter_name, iter_optim in output_dict.items():
        print(iter_name)
        iter_optim.plot_best_system()
        
    #   Plot comparison of specific params across iterations and systems, either avg of systems or specific ones, e.g. system with least fuel
    #   Inputs: 
    #   comparison metric: ['pv_capacity', 'mre_capacity', 'battery_capacity', 'battery_power', 
    #   'generator_power_kW', 'fuel_tank_size_gal', 'pv_area_ft2', 'mre_area_ft2', 
    #   'capital_cost_usd', 'pv_capital', 'mre_capital', 'battery_capital', 'generator_capital',
    #   'fuel_tank_capital', 'pv_o&m', 'mre_o&m', 'battery_o&m', 'generator_o&m',
    #   'annual_benefits_usd', 'demand_benefits_usd', 'simple_payback_yr', 'pv_avg_load', 
    #   'pv_peak_load', 'mre_avg_load', 'mre_peak_load', 'gen_avg_load', 'gen_peak_load', 
    #   'batt_avg_load', 'batt_peak_load', 'pv_percent', 'mre_percent', 'batt_percent', 
    #   'gen_percent', 'fuel_used_gal']
    #   system label: [least_fuel, least_cost, pv_only, mre_only, most_diversified]
    #   sim label: [avg, min, max, distribution]
    plot_comparison_graphs(output_dict, sensitivity_param, 'gen_total_load', 'most_diversified', 'avg')
    save_comparison_json(output_dict, sensitivity_param, 'gen_total_load', 'least_fuel', 'avg')
