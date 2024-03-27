# -*- coding: utf-8 -*-
"""
Script for running microgrid sizing tool from command line.

Created on Wed May  9 16:34:12 2018

@author: Sarah Newman
"""

import sys
import os
import pickle
import pandas as pd
from generate_solar_profile import SolarProfileGenerator
from generate_tidal_profile import TidalProfileGenerator
from microgrid_optimizer import GridSearchOptimizer
from microgrid_system import PV, SimpleLiIonBattery, Generator, FuelTank


def run_mcor(input_dict):
    system_inputs = input_dict['system_inputs']
    pv_inputs = input_dict['pv_inputs']
    mre_inputs = input_dict['mre_inputs']
    multithreading_inputs = input_dict['multithreading_inputs']
    warning_inputs = input_dict['warning_inputs']
    load_inputs = input_dict['load_inputs']
    financial_inputs = input_dict['financial_inputs']
    net_metering_inputs = input_dict['net_metering_inputs']
    sizing_inputs = input_dict['sizing_inputs']
    battery_inputs = input_dict['battery_inputs']
    post_processing_inputs = input_dict['post_processing_inputs']

    # Get renewable resource and power profiles
    power_profiles = {}
    if 'pv' in system_inputs['renewable_resources']:
        print('Creating solar profiles...')
        spg = SolarProfileGenerator(system_inputs['latitude'], system_inputs['longitude'], system_inputs['timezone'],
                                    system_inputs['altitude'], pv_inputs['tilt'], pv_inputs['azimuth'],
                                    float(system_inputs['num_trials']), float(system_inputs['length_trials']),
                                    start_year=pv_inputs['solar_data_start_year'],
                                    end_year=pv_inputs['solar_data_end_year'],
                                    pv_racking=pv_inputs['pv_racking'], pv_tracking=pv_inputs['pv_tracking'],
                                    suppress_warnings=warning_inputs['suppress_warnings'],
                                    multithreading=multithreading_inputs['multithreading'],
                                    solar_source=pv_inputs['solar_data_source'])

        if pv_inputs['get_solar_data']:
            spg.get_solar_data()

        if pv_inputs['get_solar_profiles']:
            spg.get_solar_profiles()

        print('Calculating power profiles...')
        spg.get_power_profiles()
        spg.get_night_duration(percent_at_night=battery_inputs['percent_at_night'])
        tmy_solar = spg.tmy_power_profile
        module_params = spg.get_pv_params()
        power_profiles['pv'] = spg.power_profiles
        power_profiles['night'] = spg.night_profiles

        pv_params = {'tilt': pv_inputs['tilt'], 'azimuth': pv_inputs['azimuth'],
                 'module_capacity': module_params['capacity'],
                 'module_area': module_params['area_in2'],
                 'spacing_buffer': pv_inputs['spacing_buffer'], 'advanced_inputs': {},
                 'pv_racking': pv_inputs['pv_racking'], 'pv_tracking': pv_inputs['pv_tracking']}
    else:
        spg = None
        tmy_solar=None
        pv_params = None

    # Get marine renewable power profiles
    if 'mre' in system_inputs['renewable_resources'] and mre_inputs['generator_type'] == 'tidal':
        # Run get_tidal_profile
        # Set MRE params
        tpg = TidalProfileGenerator(system_inputs['latitude'], system_inputs['longitude'], system_inputs['timezone'],
                                    float(system_inputs['num_trials']), float(system_inputs['length_trials']),
                                    advanced_inputs=mre_inputs)
        tpg.get_tidal_data_from_upload()
        tpg.extrapolate_tidal_epoch()
        tpg.generate_tidal_profiles()
        tpg.get_power_profiles()
        tmy_mre = tpg.tmy_tidal
        mre_params = {'generator_type': 'tidal', 
                      'num_generators': mre_inputs['tidal_turbine_number'],
                      'generator_capacity': mre_inputs['tidal_turbine_rated_power']}
        power_profiles['mre'] = tpg.power_profiles
    else:
        tmy_mre = None
        mre_params = None
        tpg = None

    if 'wave' in system_inputs['renewable_resources'] and mre_inputs['generator_type'] == 'wave':
        # Run get_wave_profile
        # Set MRE params
        # TODO
        pass

    # TODO - temporary solution, overwrite mre profile index to match pv index
    if 'pv' in system_inputs['renewable_resources'] and 'mre' in system_inputs['renewable_resources']:
        for mre_profile, pv_profile in zip(power_profiles['mre'], power_profiles['pv']):
            mre_profile.index = pv_profile.index

    # Set up parameter dictionaries
    location = {'longitude': system_inputs['longitude'], 'latitude': system_inputs['latitude'],
                'timezone': system_inputs['timezone'], 'altitude': system_inputs['altitude']}
    
    battery_params = {'battery_power_to_energy': battery_inputs['battery_power_to_energy'],
                      'initial_soc': battery_inputs['initial_soc'],
                      'one_way_battery_efficiency': battery_inputs['one_way_battery_efficiency'],
                      'one_way_inverter_efficiency': battery_inputs['one_way_inverter_efficiency'],
                      'soc_upper_limit': battery_inputs['soc_upper_limit'],
                      'soc_lower_limit': battery_inputs['soc_lower_limit']}

    # Create an optimizer object
    print('Running system optimization...')
    optim = GridSearchOptimizer(system_inputs['renewable_resources'], power_profiles, 
                                load_inputs['annual_load_profile'],
                                location, battery_params, financial_inputs['system_costs'],
                                tmy_solar=tmy_solar, pv_params=pv_params, tmy_mre=tmy_mre,
                                mre_params=mre_params, dispatch_strategy='available_capacity',
                                electricity_rate=financial_inputs['utility_rate'],
                                net_metering_rate=net_metering_inputs['net_metering_rate'],
                                demand_rate=financial_inputs['demand_rate'],
                                existing_components=sizing_inputs['existing_components'],
                                output_tmy=True, validate=True,
                                net_metering_limits=net_metering_inputs['net_metering_limits'],
                                off_grid_load_profile=load_inputs['off_grid_load_profile'],
                                batt_sizing_method=battery_inputs['batt_sizing_method'])

    # Create a grid of systems
    optim.define_grid(include_pv=sizing_inputs['include_pv'],
                      include_batt=sizing_inputs['include_batt'],
                      include_mre=sizing_inputs['include_mre'])

    # Get load profiles for the corresponding solar profile periods
    optim.get_load_profiles()

    # Run all simulations
    optim.run_sims_par()

    # Filter and rank results
    optim.parse_results()
    optim.filter_results(post_processing_inputs['filtering_constraints'])
    optim.rank_results(post_processing_inputs['ranking_criteria'])

    return optim, spg, tpg


if __name__ == "__main__":

    ###########################################################################
    # Define simulation parameters here
    ###########################################################################

    # Go back to main working directory
    os.chdir('..')

    days_to_hours = 24

    # Define parameters and populate dict for run_mcor()
    input_dict = {}

    # System level dictionary
    input_dict['system_inputs'] = {
        'latitude': 46.34,
        'longitude': -119.28,
        'timezone': 'US/Pacific',
        'altitude': 0,
        'num_trials': 200,
        'length_trials': 14 * days_to_hours,
        'renewable_resources': ['mre', 'pv'] # Can include 'pv' and/or 'mre', in order of dispatch'
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
        'solar_data_end_year': 2021,
        'get_solar_data': True,
        'get_solar_profiles': True
    }

    # MRE dictionary
    input_dict['mre_inputs'] =  {
        'generator_type': 'tidal',
        'tidal_turbine_rated_power': 550,
        'tidal_rotor_radius': 10,
        'tidal_rotor_number': 2,
        'tidal_turbine_number': 1,
        'maximum_cp': 0.42,
        'tidal_cut_in_velocity': 0.5,
        'tidal_cut_out_velocity': 3,
        'tidal_inverter_efficiency': 0.9,
        'tidal_turbine_losses': 10
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
        'annual_load_profile': pd.read_csv(os.path.join('data', 'sample_load_profile.csv'), 
                                           index_col=0)['Load'],
        'off_grid_load_profile': None
    }

    # Project financials
    input_dict['financial_inputs'] = {
        'utility_rate': 0.03263,                 # Utility rate in $/kWh dictionary
        'demand_rate': None,                     # Demand rate in $/kW (optional) dictionary
        'system_costs': pd.read_excel('data/MCOR Prices.xlsx', sheet_name=None, index_col=0)
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
        'include_pv': (),
        'include_mre': (),
        'include_batt': ()
    }

    # Uncomment the following to specify existing components
    # pv = PV(existing=True, pv_capacity=100, tilt=input_dict['pv_inputs']['tilt'], 
    #         azimuth=input_dict['pv_inputs']['azimuth'],
    #         module_capacity=0.360, module_area=3, spacing_buffer=2,
    #         pv_tracking='fixed', pv_racking='ground')
    # input_dict['sizing_inputs']['existing_components'] = {'pv': pv}

    # Uncomment the following to specify specific pv and battery sizes
    # input_dict['sizing_inputs']['include_pv'] = (500, 400)
    # input_dict['sizing_inputs']['include_batt'] = ((1000, 100), (1000, 500))

    # Net-metering options dictionary
    input_dict['net_metering_inputs'] = {
        'net_metering_limits': None,
        'net_metering_rate': None
    }

    # Other SolarProfileGenerator inputs dictionary
    input_dict['warning_inputs'] = {
        'suppress_warnings': False
    }

    # Output / Inputs dictionary
    input_dict['output_inputs'] = {
        'save_timeseries_json': False,
        'save_filename': 'project_name'
    }

    # Settings for dispatch plots
    # To plot the scenarios with min/max pv, set 'scenario_criteria' to 'pv', to plot
    #   scenarios with min/max fuel consumption, set 'scenario_criteria' to 'gen', or to plot
    #   a specific scenario number, set the scenario_num parameter
    scenario_criteria = 'pv'
    scenario_num = None

    save_filename = input_dict['output_inputs']['save_filename']

    # call run_mcor()
    optim, spg, tpg = run_mcor(input_dict)

    # Check resource profiles for calculation errors
    if 'pv' in input_dict['system_inputs']['renewable_resources']:
        spg.pv_checks()
    if 'mre' in input_dict['system_inputs']['renewable_resources']:
        tpg.tidal_checks()

    # Plot dispatch graphs
    # keep scenario_criteria and scenario_num defined
    # optim.plot_best_system(scenario_criteria=scenario_criteria, scenario_num=scenario_num)

    # Save results
    # optim.save_results_to_file(spg, save_filename)
    # pickle.dump(optim, open('output/{}.pkl'.format(save_filename), 'wb'))

    # if input_dict['output_inputs']['save_timeseries_json']:
    #     optim.save_timeseries_to_json(save_filename)
