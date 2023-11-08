# -*- coding: utf-8 -*-
"""
System level testing of input parameters

Created on Tues March  24 01:01:00 2020

@author: kevin.atkinson@pnnl.gov
"""


import sys
import os
import copy
import pandas as pd
import numpy as np
import json
import unittest
from unittest.mock import patch
import generate_solar_profile
import alternative_solar_profiles
from generate_solar_profile import SolarProfileGenerator
from microgrid_optimizer import GridSearchOptimizer
from config import DATA_DIR, SYS_TESTS_DIR
from microgrid_system import PV


SYS_DATA_DIR = os.path.join(SYS_TESTS_DIR, 'data')
TEST_SOLAR_DATA_DIR = os.path.join(SYS_DATA_DIR, 'solar_data')


class TestParameters(unittest.TestCase):
    """ Integration tests that check the effect of input parameter
        changes.
    """

    @classmethod
    def setUpClass(cls):
        cls.setUp()
        cls.original_optim = cls.run_main(cls)

    @classmethod
    def setUp(self):
        test_name = 'system_test_1'
        with open(os.path.join(SYS_TESTS_DIR, 'test_configs',
                               '{}.json'.format(test_name))) as json_file:
            tc = json.load(json_file)
            system = tc['system_level']
            pv = tc['pv']
            battery = tc['battery']

        # load in test configuration
        self.latitude = system['latitude']
        self.longitude = system['longitude']
        self.timezone = system['timezone']
        self.altitude = system['altitude']
        self.tilt = pv['tilt']
        self.azimuth = pv['azimuth']
        self.solar_source = pv['solar_source']
        self.num_trials = float(system['num_trials'])
        self.length_trials = float(system['length_trials'])
        self.pv_racking = pv['pv_racking']
        self.pv_tracking = pv['pv_tracking']
        self.suppress_warnings = pv['suppress_warnings']
        self.spacing_buffer = pv['spacing_buffer']
        self.solar_data_start_year = pv['solar_data_start_year']
        self.solar_data_end_year = pv['solar_data_end_year']
        self.battery_power_to_energy = battery['battery_power_to_energy']
        self.one_way_battery_efficiency = battery['one_way_battery_efficiency']
        self.initial_soc = battery['initial_soc']
        self.one_way_inverter_efficiency = battery[
            'one_way_inverter_efficiency']
        self.soc_upper_limit = battery['soc_upper_limit']
        self.soc_lower_limit = battery['soc_lower_limit']
        self.utility_rate = tc['utility_rate']
        self.net_metering_rate = tc['net_metering_rate']
        self.demand_rate = tc['demand_rate']
        self.existing_components = tc['existing_components']
        self.net_metering_limits = tc['net_metering_limits']
        self.batt_sizing_method = battery['batt_sizing_method']
        self.include_pv = tuple(tc['include_pv'])
        self.include_batt = tuple(tc['include_batt'])
        self.percent_at_night = battery['percent_at_night']
        self.constraints = tc['constraints']
        self.ranking_criteria = tc['ranking_criteria']
        self.load_profile = tc['load_profile']
        self.off_grid_load_profile = tc['off_grid_load_profile']

        if sys.platform == 'darwin':
            self.multithreading = True
        else:
            self.multithreading = False

        # Component costs and generator options
        self.system_costs = pd.read_excel(
            os.path.join(DATA_DIR, 'MCOR Prices.xlsx'), sheet_name=None,
            index_col=0)

    @patch.object(generate_solar_profile, 'SOLAR_DATA_DIR', TEST_SOLAR_DATA_DIR)
    @patch.object(alternative_solar_profiles, 'SOLAR_DATA_DIR', TEST_SOLAR_DATA_DIR)
    def run_main(self, generate_solar_data=False,
                 generate_solar_profiles=False):
        # Load (in kW)
        self.annual_load_profile = pd.read_csv(
            os.path.join(SYS_DATA_DIR, self.load_profile), index_col=0)['Load']

        if self.off_grid_load_profile:
            self.off_grid_load_profile = pd.read_csv(
                os.path.join(SYS_DATA_DIR, self.off_grid_load_profile),
                index_col=0)['Load']

        # Get solar and power profiles
        spg = SolarProfileGenerator(self.latitude, self.longitude,
                                    self.timezone, self.altitude, self.tilt,
                                    self.azimuth, float(self.num_trials),
                                    float(self.length_trials),
                                    pv_racking=self.pv_racking,
                                    pv_tracking=self.pv_tracking,
                                    suppress_warnings=False,
                                    multithreading=self.multithreading,
                                    solar_source=self.solar_source,
                                    start_year=self.solar_data_start_year,
                                    end_year=self.solar_data_end_year)
        if generate_solar_data:
            spg.get_solar_data()

        if generate_solar_profiles:
            spg.get_solar_profiles()
        spg.get_power_profiles()
        spg.get_night_duration(percent_at_night=self.percent_at_night)
        tmy_solar = spg.tmy_power_profile
        module_params = spg.get_pv_params()

        # Set up parameter dictionaries
        location = {'longitude': self.longitude, 'latitude': self.latitude,
                    'timezone': self.timezone, 'altitude': self.altitude}
        pv_params = {'tilt': self.tilt, 'azimuth': self.azimuth,
                     'module_capacity': module_params['capacity'],
                     'module_area': module_params['area_in2'],
                     'spacing_buffer': self.spacing_buffer,
                     'advanced_inputs': {}, 'pv_racking': self.pv_racking,
                     'pv_tracking': self.pv_tracking}
        battery_params = {'battery_power_to_energy':
                              self.battery_power_to_energy,
                          'initial_soc': self.initial_soc,
                          'one_way_battery_efficiency':
                              self.one_way_battery_efficiency,
                          'one_way_inverter_efficiency':
                              self.one_way_inverter_efficiency,
                          'soc_upper_limit': self.soc_upper_limit,
                          'soc_lower_limit': self.soc_lower_limit}

        # Create an optimizer object
        optim = GridSearchOptimizer(spg.power_profiles, spg.temp_profiles,
                                    spg.night_profiles, self.
                                    annual_load_profile,
                                    location, tmy_solar, pv_params,
                                    battery_params, self.system_costs,
                                    electricity_rate=self.utility_rate,
                                    net_metering_rate=self.net_metering_rate,
                                    demand_rate=self.demand_rate,
                                    existing_components=
                                        self.existing_components,
                                    output_tmy=True,
                                    validate=True,
                                    net_metering_limits=
                                        self.net_metering_limits,
                                    off_grid_load_profile=
                                        self.off_grid_load_profile,
                                    batt_sizing_method=self.batt_sizing_method,
                                    gen_power_percent=())

        # Create a grid of systems
        optim.define_grid(include_pv=self.include_pv,
                          include_batt=self.include_batt)

        # Get load profiles for the corresponding solar profile periods
        optim.get_load_profiles()

        # Run all simulations
        optim.run_sims_par()
        optim.parse_results()

        return optim

    def compare_series(self, series_name, how, optim=None,
                       generate_solar_data=False,
                       generate_solar_profiles=False):
        """
        Compares two Pandas series
        :param series_name: column name from optim results grid dataframe
        :param how: comparison to make for all elements of the series as
            compared with the original optim object series. Options:
            ['equal', 'not_equal', 'larger', 'smaller']
        :param optim: new optim object, if none provided, the simulation is rerun
        :param generate_solar_data: boolean to re-download nrel data
        :param generate_solar_profiles: boolean to re-create solar profiles
        :return:
        """

        # If no new optim object is provided, re-run the simulation
        if optim is None:
            optim = self.run_main(generate_solar_data, generate_solar_profiles)

        new_series = optim.results_grid.loc[:, series_name]
        orig_series = self.original_optim.results_grid.loc[:, series_name]

        # Remove null or zero values in series'
        new_series = new_series[new_series != 0].dropna()
        orig_series = orig_series[orig_series != 0].dropna()

        # Determine appropriate check and run assert
        if how == 'equal':
            self.assertListEqual(list(new_series), list(orig_series))
        elif how == 'not_equal':
            with self.assertRaises(AssertionError):
                self.assertListEqual(list(new_series), list(orig_series))
        elif how == 'larger':
            series_diff = new_series.values - orig_series.values
            # Assert that all elements of new series are larger than old series
            self.assertFalse(len([elem for elem in series_diff if elem <= 0]))
        elif how == 'smaller':
            series_diff = new_series.values - orig_series.values
            # Assert that all elements of new series are smaller than old series
            self.assertFalse(len([elem for elem in series_diff if elem >= 0]))
        elif how == 'smaller_eq':
            series_diff = new_series.values - orig_series.values
            # Assert that all elements of new series are smaller than or equal to old series
            self.assertFalse(len([elem for elem in series_diff if elem > 0]))
        elif how == 'larger_eq':
            series_diff = new_series.values - orig_series.values
            # Assert that all elements of new series are larger than or equal to old series
            self.assertFalse(len([elem for elem in series_diff if elem < 0]))
        else:
            raise AssertionError

    def internal_error_catch(self, generate_solar_data=False,
                             generate_solar_profiles=False):
        with self.assertRaises(AssertionError):
            _ = self.run_main(generate_solar_data, generate_solar_profiles)

    # *****System Parameter Unit Tests****** #
    def test_add_existing_pv(self):
        pv = PV(existing=True,
                pv_capacity=100,
                tilt=self.tilt,
                azimuth=self.azimuth,
                spacing_buffer=self.spacing_buffer,
                pv_tracking=self.pv_tracking,
                pv_racking=self.pv_racking,
                module_capacity=0.360,
                module_area=3)

        self.existing_components = {'pv': pv}
        optim = self.run_main()

        # Check that original grid had 21 systems
        self.assertEqual(len(self.original_optim.results_grid), 21)

        # check that an additional size was included in the results grid
        self.assertEqual(len(optim.results_grid), 26)

        # get original capital costs for all systems
        orig_cap_cost = self.original_optim.results_grid['capital_cost_usd']

        # remove the existing pv systems to compare to original system costs
        new = optim.results_grid[optim.results_grid['pv_capacity'] != 100]

        # get new capital costs for all systems
        with_exist_cap_cost = new['capital_cost_usd']

        # remove pv_0.0kW_batt_0.0kW_0.0kWh system; cap cost equal
        orig_cap_cost = orig_cap_cost.drop(index='pv_0.0kW_batt_0.0kW_0.0kWh')
        with_exist_cap_cost.drop(index='pv_0.0kW_batt_0.0kW_0.0kWh', inplace=True)

        # Check that the capital cost for all systems is smaller
        self.assertTrue(np.all(with_exist_cap_cost < orig_cap_cost))

        # Check that existing pv system is included in the results grid
        self.assertTrue(any(['pv_100.0' in system for system in optim.results_grid.index]))

    def test_include_pv(self):
        self.include_pv = (500, 400)
        optim = self.run_main()

        # Check that original grid had 21 systems
        self.assertEqual(len(self.original_optim.results_grid), 21)

        # Check that new grid has 5 additional entries for each included pv
        self.assertEqual(len(optim.results_grid), 31)

        # Check that both pv system sizes were included in the results grid
        self.assertTrue(any(['pv_400.0' in system for system in optim.results_grid.index]))
        self.assertTrue(any(['pv_500.0' in system for system in optim.results_grid.index]))

    def test_include_battery(self):
        self.include_batt = ((1000, 100), (1000, 500))
        optim = self.run_main()

        # Check that original grid had 21 systems
        self.assertEqual(len(self.original_optim.results_grid), 21)

        # Check that new grid has 4 extra entries for each included battery
        self.assertEqual(len(optim.results_grid), 29)

        # Check that both battery sizes were included in the results grid
        self.assertTrue(any(['batt_100.0kW_1000.0kWh' in system
                             for system in optim.results_grid.index]))
        self.assertTrue(any(['batt_500.0kW_1000.0kWh' in system
                             for system in optim.results_grid.index]))

    def test_net_metering_limit_capacity_cap(self):
        """Test enforcing an instantaneous export power cap (in kW)"""
        self.net_metering_limits = {'type': 'capacity_cap', 'value': 20}
        self.compare_series('annual_benefits_usd', 'smaller')

    def test_net_metering_limit_percent(self):
        self.net_metering_limits = {'type': 'percent_of_load', 'value': 85}
        self.compare_series('annual_benefits_usd', 'smaller_eq')

    def test_net_metering_rate_zero(self):
        self.net_metering_rate = 0
        self.compare_series('annual_benefits_usd', 'smaller')

    def test_net_metering_rate_low(self):
        self.net_metering_rate = self.utility_rate * 0.5
        self.compare_series('annual_benefits_usd', 'smaller_eq')

    def test_generator_cost(self):
        self.system_cost_changes_helper('column', 'generator_costs', 'Cost (USD)', 10000,
                                        'capital_cost_usd', 'smaller')

    def test_fuel_tank_cost(self):
        self.system_cost_changes_helper('column', 'fuel_tank_costs', 'Cost (USD)',
                                        float(5000), 'capital_cost_usd', 'smaller')

    def test_pv_ground_fixed_cost(self):
        self.system_cost_changes_helper('row', 'pv_costs', 'ground;fixed', 5, 'pv_capital',
                                        'larger')

    def test_pv_ground_tracking_cost(self):
        self.pv_tracking = 'single_axis'
        self.system_cost_changes_helper('row', 'pv_costs', 'ground;single_axis', 5,
                                        'pv_capital', 'larger')

    def test_pv_roof_cost(self):
        self.pv_racking = 'roof'
        self.system_cost_changes_helper('row', 'pv_costs', 'roof;fixed', 5, 'pv_capital',
                                        'larger')

    def test_pv_carport_cost(self):
        self.pv_racking = 'carport'
        self.system_cost_changes_helper('row', 'pv_costs', 'carport;fixed', 5, 'pv_capital',
                                        'larger')

    def test_battery_cost(self):
        self.system_cost_changes_helper('row', 'battery_costs', 'Cost', 0.6,
                                        'battery_capital', 'larger_eq')

    def test_generator_om_scalar_cost(self):
        self.system_cost_changes_helper('cell', 'om_costs', 'Cost', 200, 'generator_o&m',
                                        'larger', 'Generator_scalar')

    def test_generator_om_exp_cost(self):
        self.system_cost_changes_helper('cell', 'om_costs', 'Cost', 1, 'generator_o&m',
                                        'larger', 'Generator_exp')

    def test_battery_om_cost(self):
        self.system_cost_changes_helper('cell', 'om_costs', 'Cost', 3, 'battery_o&m',
                                        'larger', 'Battery')

    def test_pv_om_fixed_cost(self):
        self.system_cost_changes_helper('cell', 'om_costs', 'Cost', 20, 'pv_o&m', 'larger',
                                        'PV_ground;fixed')

    def test_pv_om_tracking_cost(self):
        self.pv_tracking = 'single_axis'
        self.system_cost_changes_helper('cell', 'om_costs', 'Cost', 20, 'pv_o&m', 'larger',
                                        'PV_ground;single_axis')

    def test_pv_om_roof_cost(self):
        self.pv_racking = 'roof'
        self.system_cost_changes_helper('cell', 'om_costs', 'Cost', 20, 'pv_o&m', 'larger',
                                        'PV_roof;fixed')

    def test_pv_om_carport_cost(self):
        self.pv_racking = 'carport'
        self.system_cost_changes_helper('cell', 'om_costs', 'Cost', 20, 'pv_o&m', 'larger',
                                        'PV_carport;fixed')

    def system_cost_changes_helper(self, field_type, sheet, label, value, cost_column, how,
                                   column=None):
        new_system_costs = self.system_costs.copy()
        if field_type == 'row':
            new_system_costs[sheet].loc[label] = value
        elif field_type == 'column':
            new_system_costs[sheet].loc[:, label] = value
        elif field_type == 'cell':
            new_system_costs[sheet].loc[label, column] = value
        else:
            raise AttributeError
        self.system_costs = new_system_costs
        self.compare_series(cost_column, how)

    def test_constraints(self):
        # Set constraint
        constraint_value = 50000
        parameter = 'pv_area_ft2'
        constraints = [{'parameter': parameter, 'type': 'max', 'value': constraint_value}]

        # Verify that systems that violated the constraint were included in
        #   the original results
        self.assertTrue(len(self.original_optim.results_grid[
            self.original_optim.results_grid[parameter] > constraint_value]))

        # Apply constraints
        optim_copy = copy.deepcopy(self.original_optim)
        optim_copy.filter_results(constraints)

        # Verify that those systems are not in the updated results
        self.assertFalse(len(optim_copy.results_grid[
            optim_copy.results_grid[parameter] > constraint_value]))

    def test_off_grid_load_profile(self):
        # Add off grid load profile and re-run simulation
        self.off_grid_load_profile = 'sample_off_grid_load_profile.csv'
        optim = self.run_main()

        # Verify that pv capacity has not changed, but that both battery and
        #   generator capacity have decreased
        with self.subTest():
            self.compare_series('pv_capacity', 'equal', optim)
        with self.subTest():
            self.compare_series('battery_capacity', 'smaller', optim)
        with self.subTest():
            self.compare_series('generator_power', 'smaller', optim)

    def test_add_rank(self):
        # Set ranking criteria
        parameter = 'capital_cost_usd'
        ranking_criteria = [{'parameter': parameter,
                             'order_type': 'ascending'}]

        # Copy optim object and rank
        optim_copy = copy.deepcopy(self.original_optim)
        optim_copy.rank_results(ranking_criteria)

        # Verify that the same systems are included as before
        self.assertListEqual(list(self.original_optim.results_grid[
                                     parameter].sort_values()),
                             list(optim_copy.results_grid[
                                      parameter].sort_values()))

        # Verify that the new values are sorted
        self.assertListEqual(list(optim_copy.results_grid[parameter]),
                             list(optim_copy.results_grid[parameter].sort_values()))

    def test_demand_rate(self):
        # Add demand rate, should decrease payback time
        self.demand_rate = 10
        optim = self.run_main()

        # Check that payback time has decreased for each system
        # Get systems with postive payback times from original results
        orig_pb = self.original_optim.results_grid['simple_payback_yr']
        non_null_index = orig_pb.dropna().index
        new_pb = optim.results_grid['simple_payback_yr']

        series_diff = orig_pb.loc[non_null_index].values - new_pb.loc[non_null_index].values

        # Assert that all elements of new series are smaller than old series
        self.assertFalse(len([elem for elem in series_diff if elem <= 0]))

    def test_utility_rate(self):
        # Increase utility rate, should increase annual benefits
        self.utility_rate = self.utility_rate + 0.5
        self.compare_series('annual_benefits_usd', 'larger')

    def test_battery_percent_at_night(self):
        # Change battery percent at night (when battery starts discharging),
        #   should impact battery load percent
        self.percent_at_night = 0.2
        self.compare_series('batt_percent mean', 'not_equal')

    def test_battery_parameters_sizing_method(self):
        # Switch battery sizing method to no_pv_export, should only have one
        #   battery size per pv size
        self.batt_sizing_method = 'no_pv_export'
        optim = self.run_main()

        # Check that original grid had 21 systems
        self.assertEqual(len(self.original_optim.results_grid), 21)

        # Check that new grid only has 5
        self.assertEqual(len(optim.results_grid), 5)

    def test_battery_soc_upper_limit(self):
        # Decrease soc upper limit, should decrease battery load percent
        self.initial_soc = 0.8
        self.soc_upper_limit = 0.8
        self.compare_series('batt_percent mean', 'smaller')

    def test_battery_soc_lower_limit(self):
        # Increase soc lower limit, should decrease battery load percent
        self.soc_lower_limit = 0.5
        self.compare_series('batt_percent mean', 'smaller')

    def test_battery_initial_soc(self):
        # Decrease initial battery SOC, should decrease battery load percent
        self.initial_soc = 0.2
        self.compare_series('batt_percent mean', 'smaller')

    def test_battery_efficiency(self):
        # Decrease battery efficiency, should increase required battery sizes
        self.one_way_battery_efficiency = 0.8
        self.compare_series('battery_capacity', 'larger')

    def test_inverter_efficiency(self):
        # Decrease inverter efficiency, should increase required battery sizes
        self.one_way_inverter_efficiency = 0.8
        self.compare_series('battery_capacity', 'larger')

    def test_battery_parameters_power_to_energy(self):
        # Increase power to energy ratio, should increase battery power ratings
        self.battery_power_to_energy = 1
        self.compare_series('battery_power', 'larger')

    def test_load_profiles(self):
        # Use a different load profile, PV and battery sizes should be different
        self.load_profile = 'midrise_mf_flagstaff.csv'
        optim = self.run_main()
        self.compare_series('pv_capacity', 'smaller', optim)
        self.compare_series('battery_capacity', 'smaller', optim)

    def test_pv_tilt(self):
        # Decrease tilt, should increase required capacity
        self.tilt -= 10
        self.compare_series('pv_capacity', 'larger')

    def test_pv_azimuth(self):
        # Change azimuth, should increase required capacity
        self.azimuth -= 30
        self.compare_series('pv_capacity', 'larger')

    def test_pv_racking(self):
        # Use carport racking, should increase cost
        self.pv_racking = 'carport'
        self.compare_series('pv_capital', 'larger')

    def test_pv_tracking(self):
        # Use single-axis tracking, should decrease required pv capacity
        self.pv_tracking = 'single_axis'
        optim = self.run_main()
        self.compare_series('pv_capacity', 'smaller', optim)

    def test_lat_long_changes(self):
        # Check that for different latitude and longitude, pv and battery sizes are different
        self.latitude = self.latitude + 5
        self.longitude = self.longitude - 3
        self.length_trials = self.length_trials
        optim = self.run_main(generate_solar_data=True, generate_solar_profiles=True)
        self.compare_series('pv_capacity', 'not_equal', optim)
        self.compare_series('battery_capacity', 'not_equal', optim)

    def test_solar_source(self):
        # Check that if himawari is selected as the data source, the simulation runs without
        #   error
        self.latitude = -18.04
        self.longitude = 178.04
        self.length_trials = self.length_trials
        self.timezone = 'Pacific/Fiji'
        self.solar_source = 'himawari'
        self.solar_data_start_year = 2016
        self.solar_data_end_year = 2020
        optim = self.run_main(generate_solar_data=True, generate_solar_profiles=True)
        self.assertTrue(True)
