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

if __name__ == "__main__":

    ###########################################################################
    # Define simulation parameters here
    ###########################################################################

    # Go back to main working directory
    os.chdir('..')

    # System level
    latitude = 46.34
    longitude = -119.28
    timezone = 'US/Pacific'
    # common options = [US/Alaska, US/Aleutian, US/Arizona, US/Central,
    #   US/East-Indiana, US/Eastern, US/sHawaii, US/Indiana-Starke,
    #   US/Michigan, US/Mountain, US/Pacific, US/Pacific-New, US/Samoa]
    altitude = 0
    num_trials = 200
    length_trials = 14

    # PV
    tilt = 20
    azimuth = 180.
    spacing_buffer = 2
    pv_racking = 'ground'  # options: [ground, roof, or carport]
    pv_tracking = 'fixed'  # fixed or single_axis
    # define where the solar data is downloaded from. Valid options are:
    #   ['nsrdb', 'himawari']. The default value ('nsrdb') pulls data from
    #   NREL's NSRDB which covers CONUS, Central America, and parts of Canada
    #   and South America. It has data from 1998-2021. The Himawari dataset
    #   covers East Asia and the Pacific Islands. It has data from 2016-2020.
    solar_data_source = 'nsrdb'
    solar_data_start_year = 1998  # the first year of historical solar data
    solar_data_end_year = 2020  # the last year of historical solar data

    # Battery
    battery_power_to_energy = 0.25
    initial_soc = 1
    one_way_battery_efficiency = 0.9
    one_way_inverter_efficiency = 0.95
    soc_upper_limit = 1
    soc_lower_limit = 0.2
    batt_sizing_method = 'longest_night'  # 'longest_night' or 'no_pv_export'

    # The fraction of maximum PV power below which it is considered night,
    #   for the purposes of battery discharging
    percent_at_night = 0.1

    # Load (in kW)
    annual_load_profile = pd.read_csv(
        os.path.join('data', 'sample_load_profile.csv'), index_col=0)['Load']
    off_grid_load_profile = None
    save_filename = 'project_name'

    # Utility rate in $/kWh
    utility_rate = 0.03263

    # Demand rate in $/kW (optional)
    # The demand rate can either be a single number or a list with a rate
    # for each month.
    demand_rate = None

    # Component costs and generator options
    system_costs = pd.read_excel('data/MCOR Prices.xlsx', sheet_name=None,
                                 index_col=0)

    # Determine if asp multithreading should be used
    multithreading = False

    # Filtering constraints
    # The constraints list contains dictionaries with the format:
    #    {parameter, type, value}
    # where parameter can be any of the following:
    #    capital_cost_usd, pv_area_ft2, annual_benefits_usd, simple_payback_yr,
    #    fuel_used_gal mean, fuel_used_gal most-conservative, pv_percent mean,
    #    pv_capacity
    # and type can be [max, min]
    # and value is the maximum or minimum allowable value
    constraints = []

    # Ranking criteria
    # The ranking_criteria list is ordered from most to least important,
    # and includes dictionaries with the format:
    #    {parameter, order_type}
    # where parameter can be any of the following:
    #    capital_cost_usd, annual_benefits_usd, simple_payback_yr,
    #    fuel_used_gal mean, fuel_used_gal most-conservative
    # and order_type can be [ascending, descending]
    ranking_criteria = [{'parameter': 'simple_payback_yr',
                         'order_type': 'ascending'}]

    # Settings for dispatch plots
    # To plot the scenarios with min/max pv, set 'scenario_criteria' to 'pv', to plot
    #   scenarios with min/max fuel consumption, set 'scenario_criteria' to 'gen', or to plot
    #   a specific scenario number, set the scenario_num parameter
    scenario_criteria = 'pv'
    scenario_num = None

    # Existing components - this should be used if no smaller sizes are to be
    #   considered and equipment costs should not be included, otherwise use
    #   the include_pv option below.
    existing_components = {}
    # Uncomment the following to specify existing components
    # pv = PV(existing=True, pv_capacity=100, tilt=tilt, azimuth=azimuth,
    #         module_capacity=0.360, module_area=3, spacing_buffer=2,
    #         pv_tracking='fixed', pv_racking='ground')
    # existing_components = {'pv': pv}

    # Including specific PV and battery sizes
    include_pv = ()
    include_batt = ()
    # Uncomment the following to specify specific pv and battery sizes
    # include_pv = (500, 400)
    # include_batt = ((1000, 100), (1000, 500))

    # Net-metering options:
    #   (1) To specify a net-metering rate different from the utility rate,
    #       set net_metering_rate below
    #   (2) To specify no net-metering, set net_metering_rate to 0
    #   (3) To enforce an installed capacity cap, use the constraints
    #       parameter above with a pv_capacity limit
    #   (4) To enforce an instantaneous export power cap (in kW), set the
    #       net_metering_limits parameter below, e.g.
    #           net_metering_limits = {'type': 'capacity_cap', 'value': 100}
    #   (5) To enforce a cap as a percentage of load, set:
    #           net_metering_limits = {'type': 'percent_of_load', 'value': 100}
    #   (6) In the case where there is no net-metering, but the battery is
    #       sized to capture all excess pv generation (i.e. if
    #       batt_sizing_method = 'no_pv_export'), and you want to estimate
    #       the revenue from using the battery during normal operation to
    #       capture and use this excess generation, set the
    #       net_metering_limits parameter below, e.g.
    #           net_metering_limits = {'type': 'no_nm_use_battery'}
    #       *Note*: this will not include any additional costs due to wearing
    #       out the battery more quickly from using it daily and does not
    #       capture revenue from using the battery for grid services. It
    #       also assumes that the battery is sized with the no_pv_export
    #       method.
    net_metering_limits = None
    net_metering_rate = None

    ###########################################################################

    # Get solar and power profiles
    print('Creating solar profiles...')
    spg = SolarProfileGenerator(latitude, longitude, timezone, altitude, tilt,
                                azimuth, float(num_trials),
                                float(length_trials),
                                start_year=solar_data_start_year,
                                end_year=solar_data_end_year,
                                pv_racking=pv_racking,
                                pv_tracking=pv_tracking,
                                suppress_warnings=False,
                                multithreading=multithreading,
                                solar_source=solar_data_source)

    # Note: it is strongly recommended to comment out the following two lines
    #   after running them once for a particular site. This will save a lot
    #   of time by not having to regenerate the solar profiles
    spg.get_solar_data()
    spg.get_solar_profiles()
    print('Calculating power profiles...')
    spg.get_power_profiles()
    spg.get_night_duration(percent_at_night=percent_at_night)
    tmy_solar = spg.tmy_power_profile
    module_params = spg.get_pv_params()

    # Set up parameter dictionaries
    location = {'longitude': longitude, 'latitude': latitude,
                'timezone': timezone, 'altitude': altitude}
    pv_params = {'tilt': tilt, 'azimuth': azimuth,
                 'module_capacity': module_params['capacity'],
                 'module_area': module_params['area_in2'],
                 'spacing_buffer': spacing_buffer, 'advanced_inputs': {},
                 'pv_racking': pv_racking, 'pv_tracking': pv_tracking}
    battery_params = {'battery_power_to_energy': battery_power_to_energy,
                      'initial_soc': initial_soc,
                      'one_way_battery_efficiency': one_way_battery_efficiency,
                      'one_way_inverter_efficiency':
                          one_way_inverter_efficiency,
                      'soc_upper_limit': soc_upper_limit,
                      'soc_lower_limit': soc_lower_limit}

    # Create an optimizer object
    print('Running system optimization...')
    optim = GridSearchOptimizer(spg.power_profiles, spg.temp_profiles,
                                spg.night_profiles, annual_load_profile,
                                location, tmy_solar, pv_params, battery_params,
                                system_costs,
                                electricity_rate=utility_rate,
                                net_metering_rate=net_metering_rate,
                                demand_rate=demand_rate,
                                existing_components=existing_components,
                                output_tmy=True,
                                validate=True,
                                net_metering_limits=net_metering_limits,
                                off_grid_load_profile=off_grid_load_profile,
                                batt_sizing_method=batt_sizing_method,
                                gen_power_percent=())

    # Create a grid of systems
    optim.define_grid(include_pv=include_pv, include_batt=include_batt)

    # Get load profiles for the corresponding solar profile periods
    optim.get_load_profiles()

    # Run all simulations
    optim.run_sims_par()

    # Filter and rank results
    optim.parse_results()
    optim.filter_results(constraints)
    optim.rank_results(ranking_criteria)

    # Check pv profiles for calculation errors
    spg.pv_checks()

    # Plot dispatch graphs
    optim.plot_best_system(scenario_criteria=scenario_criteria, scenario_num=scenario_num)

    # Save results
    optim.save_results_to_file(spg, save_filename)
    pickle.dump(optim, open('output/{}.pkl'.format(save_filename), 'wb'))
