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
from microgrid_optimizer import GridSearchOptimizer
from microgrid_system import PV, SimpleLiIonBattery, Generator, FuelTank


def run_mcor(input_dict):
    system_inputs = input_dict["system_inputs"]
    pv_inputs = input_dict["pv_inputs"]
    multithreading_inputs = input_dict["multithreading_inputs"]
    other_SolarProfileGenerator_inputs = input_dict["other_SolarProfileGenerator_inputs"]
    load_inputs = input_dict["load_inputs"]
    system_costs_inputs = input_dict["system_costs_inputs"]
    utility_rate_inputs = input_dict["utility_rate_inputs"]
    net_metering_inputs = input_dict["net_metering_inputs"]
    demand_rate_inputs = input_dict["demand_rate_inputs"]
    existing_components_inputs = input_dict["existing_components_inputs"]
    battery_inputs = input_dict["battery_inputs"]
    specific_pv_battery_sizes_inputs = input_dict["specific_pv_battery_sizes_inputs"]
    dispatch_plots_settings_inputs = input_dict["dispatch_plots_settings_inputs"]
    filtering_constraints_inputs = input_dict["filtering_constraints_inputs"]
    ranking_criteria_inputs = input_dict["ranking_criteria_inputs"]

    # Get solar and power profiles
    print('Creating solar profiles...')
    spg = SolarProfileGenerator(system_inputs["latitude"], system_inputs["longitude"], system_inputs["timezone"],
                                system_inputs["altitude"], pv_inputs["tilt"], pv_inputs["azimuth"],
                                float(system_inputs["num_trials"]), float(system_inputs["length_trials"]),
                                start_year=pv_inputs["solar_data_start_year"],
                                end_year=pv_inputs["solar_data_end_year"],
                                pv_racking=pv_inputs["pv_racking"], pv_tracking=pv_inputs["pv_tracking"],
                                suppress_warnings=other_SolarProfileGenerator_inputs["suppress_warnings"],
                                multithreading=multithreading_inputs["multithreading"],
                                solar_source=pv_inputs["solar_data_source"])

    if pv_inputs["get_solar_data"]:
        spg.get_solar_data()

    if pv_inputs["get_solar_profiles"]:
        spg.get_solar_profiles()

    print('Calculating power profiles...')
    spg.get_power_profiles()
    spg.get_night_duration(percent_at_night=battery_inputs["percent_at_night"])
    tmy_solar = spg.tmy_power_profile
    module_params = spg.get_pv_params()

    # Set up parameter dictionaries
    location = {'longitude': system_inputs["longitude"], 'latitude': system_inputs["latitude"],
                'timezone': system_inputs["timezone"], 'altitude': system_inputs["altitude"]}

    pv_params = {'tilt': pv_inputs["tilt"], 'azimuth': pv_inputs["azimuth"],
                 'module_capacity': module_params['capacity'],
                 'module_area': module_params['area_in2'],
                 'spacing_buffer': pv_inputs["spacing_buffer"], 'advanced_inputs': {},
                 'pv_racking': pv_inputs["pv_racking"], 'pv_tracking': pv_inputs["pv_tracking"]}

    battery_params = {'battery_power_to_energy': battery_inputs["battery_power_to_energy"],
                      'initial_soc': battery_inputs["initial_soc"],
                      'one_way_battery_efficiency': battery_inputs["one_way_battery_efficiency"],
                      'one_way_inverter_efficiency': battery_inputs["one_way_inverter_efficiency"],
                      'soc_upper_limit': battery_inputs["soc_upper_limit"],
                      'soc_lower_limit': battery_inputs["soc_lower_limit"]}

    # Create an optimizer object
    print('Running system optimization...')
    optim = GridSearchOptimizer(spg.power_profiles, spg.temp_profiles,
                                spg.night_profiles, load_inputs["annual_load_profile"],
                                location, tmy_solar, pv_params, battery_params,
                                system_costs_inputs["system_costs"],
                                electricity_rate=utility_rate_inputs["utility_rate"],
                                net_metering_rate=net_metering_inputs["net_metering_rate"],
                                demand_rate=demand_rate_inputs["demand_rate"],
                                existing_components=existing_components_inputs["existing_components"],
                                output_tmy=True,
                                validate=True,
                                net_metering_limits=net_metering_inputs["net_metering_limits"],
                                off_grid_load_profile=load_inputs["off_grid_load_profile"],
                                batt_sizing_method=battery_inputs["batt_sizing_method"],
                                gen_power_percent=())

    # Create a grid of systems
    optim.define_grid(include_pv=specific_pv_battery_sizes_inputs["include_pv"],
                      include_batt=specific_pv_battery_sizes_inputs["include_batt"])

    # Get load profiles for the corresponding solar profile periods
    optim.get_load_profiles()

    # Run all simulations
    optim.run_sims_par()

    # Filter and rank results
    optim.parse_results()
    optim.filter_results(filtering_constraints_inputs["constraints"])
    optim.rank_results(ranking_criteria_inputs["ranking_criteria"])

    return optim, spg

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
    input_dict["system_inputs"] = {}
    input_dict["system_inputs"]["latitude"] = 46.34
    input_dict["system_inputs"]["longitude"] = -119.28
    input_dict["system_inputs"]["timezone"] = 'US/Pacific'
    input_dict["system_inputs"]["altitude"] = 0
    input_dict["system_inputs"]["num_trials"] = 200
    input_dict["system_inputs"]["length_trials"] = 14 * days_to_hours

    # PV dictionary
    input_dict["pv_inputs"] = {}
    input_dict["pv_inputs"]["tilt"] = 20
    input_dict["pv_inputs"]["azimuth"] = 180.
    input_dict["pv_inputs"]["spacing_buffer"] = 2
    input_dict["pv_inputs"]["pv_racking"] = 'ground'
    input_dict["pv_inputs"]["pv_tracking"] = 'fixed'
    input_dict["pv_inputs"]["solar_data_source"] = 'nsrdb'
    input_dict["pv_inputs"]["solar_data_start_year"] = 1998
    input_dict["pv_inputs"]["solar_data_end_year"] = 2021
    input_dict["pv_inputs"]["get_solar_data"] = True
    input_dict["pv_inputs"]["get_solar_profiles"] = True

    # Battery dictionary
    input_dict["battery_inputs"] = {}
    input_dict["battery_inputs"]["battery_power_to_energy"] = 0.25
    input_dict["battery_inputs"]["initial_soc"] = 1
    input_dict["battery_inputs"]["one_way_battery_efficiency"] = 0.9
    input_dict["battery_inputs"]["one_way_inverter_efficiency"] = 0.95
    input_dict["battery_inputs"]["soc_upper_limit"] = 1
    input_dict["battery_inputs"]["soc_lower_limit"] = 0.2
    input_dict["battery_inputs"]["batt_sizing_method"] = 'longest_night'
    input_dict["battery_inputs"]["percent_at_night"] = 0.1

    # Load (in kW) dictionary
    input_dict["load_inputs"] = {}
    input_dict["load_inputs"]["annual_load_profile"] = pd.read_csv(
        os.path.join('data', 'sample_load_profile.csv'), index_col=0)['Load']
    input_dict["load_inputs"]["off_grid_load_profile"] = None
    input_dict["load_inputs"]["save_filename"] = 'project_name'

    # Utility rate in $/kWh dictionary
    input_dict["utility_rate_inputs"] = {}
    input_dict["utility_rate_inputs"]["utility_rate"] = 0.03263

    # Demand rate in $/kW (optional) dictionary
    input_dict["demand_rate_inputs"] = {}
    input_dict["demand_rate_inputs"]["demand_rate"] = None

    # Component costs and generator options dictionary
    input_dict["system_costs_inputs"] = {}
    input_dict["system_costs_inputs"]["system_costs"] = pd.read_excel('data/MCOR Prices.xlsx', sheet_name=None,
                                                                      index_col=0)

    # Determine if asp multithreading should be used dictionary
    input_dict["multithreading_inputs"] = {}
    input_dict["multithreading_inputs"]["multithreading"] = False

    # Filtering constraints dictionary
    input_dict["filtering_constraints_inputs"] = {}
    input_dict["filtering_constraints_inputs"]["constraints"] = []

    # Ranking criteria dictionary
    input_dict["ranking_criteria_inputs"] = {}
    input_dict["ranking_criteria_inputs"]["ranking_criteria"] = [{'parameter': 'simple_payback_yr',
                                                                  'order_type': 'ascending'}]

    # Settings for dispatch plots dictionary
    input_dict["dispatch_plots_settings_inputs"] = {}
    input_dict["dispatch_plots_settings_inputs"]["scenario_criteria"] = 'pv'
    input_dict["dispatch_plots_settings_inputs"]["scenario_num"] = None

    # Existing components dictionary
    input_dict["existing_components_inputs"] = {}
    input_dict["existing_components_inputs"]["existing_components"] = {}
    # Uncomment the following to specify existing components
    # input_dict["existing_components_inputs"]["pv"] = PV(existing=True, pv_capacity=100, tilt=tilt, azimuth=azimuth,
    #         module_capacity=0.360, module_area=3, spacing_buffer=2,
    #         pv_tracking='fixed', pv_racking='ground')
    # input_dict["existing_components_inputs"]["existing_components"] = {'pv': pv}

    # Specific PV and battery sizes dictionary
    input_dict["specific_pv_battery_sizes_inputs"] = {}
    input_dict["specific_pv_battery_sizes_inputs"]["include_pv"] = ()
    input_dict["specific_pv_battery_sizes_inputs"]["include_batt"] = ()
    # Uncomment the following to specify specific pv and battery sizes
    # input_dict["specific_pv_battery_sizes_inputs"]["include_pv"] = (500, 400)
    # input_dict["specific_pv_battery_sizes_inputs"]["include_batt"] = ((1000, 100), (1000, 500))

    # Net-metering options dictionary
    input_dict["net_metering_inputs"] = {}
    input_dict["net_metering_inputs"]["net_metering_limits"] = None
    input_dict["net_metering_inputs"]["net_metering_rate"] = None

    # Other SolarProfileGenerator inputs dictionary
    input_dict["other_SolarProfileGenerator_inputs"] = {}
    input_dict["other_SolarProfileGenerator_inputs"]["suppress_warnings"] = False

    # Settings for dispatch plots
    # To plot the scenarios with min/max pv, set 'scenario_criteria' to 'pv', to plot
    #   scenarios with min/max fuel consumption, set 'scenario_criteria' to 'gen', or to plot
    #   a specific scenario number, set the scenario_num parameter
    scenario_criteria = 'pv'
    scenario_num = None

    save_filename = 'project_name'

    # call run_mcor()
    optim, spg = run_mcor(input_dict)

    # Check pv profiles for calculation errors
    spg.pv_checks()

    # Plot dispatch graphs
    # keep scenario_criteria and scenario_num defined
    optim.plot_best_system(scenario_criteria=scenario_criteria, scenario_num=scenario_num)

    # Save results
    optim.save_results_to_file(spg, save_filename)
    pickle.dump(optim, open('output/{}.pkl'.format(save_filename), 'wb'))
