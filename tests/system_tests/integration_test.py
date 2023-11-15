"""
Integration tests for MCOR. Confirms that the results are consistent for a set of fixed
    inputs.
"""

import os
import unittest
from unittest.mock import patch
import json
import pandas as pd

import generate_solar_profile
import alternative_solar_profiles
from generate_solar_profile import SolarProfileGenerator
from microgrid_optimizer import GridSearchOptimizer
from config import DATA_DIR, SYS_TESTS_DIR
from microgrid_system import PV

SYS_DATA_DIR = os.path.join(SYS_TESTS_DIR, 'data')
TEST_SOLAR_DATA_DIR = os.path.join(SYS_DATA_DIR, 'solar_data')
rtol = 1e-2
atol = 1e-4


class TestSimulation(unittest.TestCase):
    """ Integration tests to check that results are consistent across subsequent
        simulations.
    """

    @classmethod
    def setUp(cls):
        test_name = 'system_test_2'
        with open(os.path.join(SYS_TESTS_DIR, 'test_configs',
                               '{}.json'.format(test_name))) as f:
            cls.tc = json.load(f)

        # Component costs and generator options
        cls.system_costs = pd.read_excel(
            os.path.join(DATA_DIR, 'MCOR Prices.xlsx'), sheet_name=None,
            index_col=0)

    @patch.object(generate_solar_profile, 'SOLAR_DATA_DIR', TEST_SOLAR_DATA_DIR)
    @patch.object(alternative_solar_profiles, 'SOLAR_DATA_DIR', TEST_SOLAR_DATA_DIR)
    def run_main(self):
        # Load (in kW)
        self.annual_load_profile = pd.read_csv(
            os.path.join(SYS_DATA_DIR, self.tc['load_profile']), index_col=0)['Load']

        if self.tc['off_grid_load_profile']:
            self.tc['off_grid_load_profile'] = pd.read_csv(
                os.path.join(SYS_DATA_DIR, self.tc['off_grid_load_profile']),
                index_col=0)['Load']

        # Get solar and power profiles
        spg = SolarProfileGenerator(self.tc['latitude'], self.tc['longitude'],
                                    self.tc['timezone'], self.tc['altitude'], self.tc['tilt'],
                                    self.tc['azimuth'], float(self.tc['num_trials']),
                                    float(self.tc['length_trials']),
                                    pv_racking=self.tc['pv_racking'],
                                    pv_tracking=self.tc['pv_tracking'],
                                    suppress_warnings=False,
                                    multithreading=False)

        spg.get_power_profiles()
        spg.get_night_duration(percent_at_night=self.tc['percent_at_night'])
        tmy_solar = spg.tmy_power_profile
        module_params = spg.get_pv_params()

        # Set up parameter dictionaries
        location = {'longitude': self.tc['longitude'], 'latitude': self.tc['latitude'],
                    'timezone': self.tc['timezone'], 'altitude': self.tc['altitude']}
        pv_params = {'tilt': self.tc['tilt'], 'azimuth': self.tc['azimuth'],
                     'module_capacity': module_params['capacity'],
                     'module_area': module_params['area_in2'],
                     'spacing_buffer': self.tc['spacing_buffer'],
                     'advanced_inputs': {}, 'pv_racking': self.tc['pv_racking'],
                     'pv_tracking': self.tc['pv_tracking']}
        battery_params = {'battery_power_to_energy':
                              self.tc['battery_power_to_energy'],
                          'initial_soc': self.tc['initial_soc'],
                          'one_way_battery_efficiency':
                              self.tc['one_way_battery_efficiency'],
                          'one_way_inverter_efficiency':
                              self.tc['one_way_inverter_efficiency'],
                          'soc_upper_limit': self.tc['soc_upper_limit'],
                          'soc_lower_limit': self.tc['soc_lower_limit']}

        # Create an optimizer object
        optim = GridSearchOptimizer(spg.power_profiles, spg.temp_profiles,
                                    spg.night_profiles, self.
                                    annual_load_profile,
                                    location, tmy_solar, pv_params,
                                    battery_params, self.system_costs,
                                    electricity_rate=self.tc['utility_rate'],
                                    net_metering_rate=self.tc['net_metering_rate'],
                                    demand_rate=self.tc['demand_rate'],
                                    existing_components=
                                        self.tc['existing_components'],
                                    output_tmy=True,
                                    validate=True,
                                    net_metering_limits=
                                        self.tc['net_metering_limits'],
                                    off_grid_load_profile=
                                        self.tc['off_grid_load_profile'],
                                    batt_sizing_method=self.tc['batt_sizing_method'],
                                    gen_power_percent=())

        # Create a grid of systems
        optim.define_grid(include_pv=self.tc['include_pv'],
                          include_batt=self.tc['include_batt'])

        # Get load profiles for the corresponding solar profile periods
        optim.get_load_profiles()

        # Run all simulations
        optim.run_sims_par()
        optim.parse_results()

        return optim

    def test_longest_night_battery(self):
        # Set battery sizing method to longest_night
        self.tc['batt_sizing_method'] = 'longest_night'

        # Run simulation
        optim = self.run_main()

        # Compare results to ground truth data
        ground_truth_df = pd.read_csv(os.path.join(SYS_DATA_DIR, 'ground_truth',
                                                   'longest_night_ground_truth.csv'),
                                      header=0, index_col=0)
        pd.testing.assert_frame_equal(ground_truth_df, optim.results_grid, check_exact=False,rtol=rtol, atol=atol)

    def test_no_pv_export_battery(self):
        self.tc['batt_sizing_method'] = 'no_pv_export'

        # Run simulation
        optim = self.run_main()

        # Compare results to ground truth data
        ground_truth_df = pd.read_csv(os.path.join(SYS_DATA_DIR, 'ground_truth',
                                                   'no_pv_export_ground_truth.csv'),
                                      header=0, index_col=0)
        pd.testing.assert_frame_equal(ground_truth_df, optim.results_grid, check_exact=False,rtol=rtol, atol=atol)

    def test_net_metering_limits(self):
        self.tc['net_metering_limits'] = {'type': 'percent_of_load', 'value': 100}

        # Run simulation
        optim = self.run_main()

        # Compare results to ground truth data
        ground_truth_df = pd.read_csv(os.path.join(SYS_DATA_DIR, 'ground_truth',
                                                   'nm_limit_ground_truth.csv'),
                                      header=0, index_col=0)
        pd.testing.assert_frame_equal(ground_truth_df, optim.results_grid, check_exact=False,rtol=rtol, atol=atol)

    def test_existing_equipment(self):
        pv = PV(existing=True, pv_capacity=300, tilt=self.tc['tilt'],
                azimuth=self.tc['azimuth'], module_capacity=0.360, module_area=3,
                spacing_buffer=2, pv_tracking='fixed', pv_racking='ground')
        self.tc['existing_components'] = {'pv': pv}

        # Run simulation
        optim = self.run_main()

        # Compare results to ground truth data
        ground_truth_df = pd.read_csv(os.path.join(SYS_DATA_DIR, 'ground_truth',
                                                   'existing_pv_ground_truth.csv'),
                                      header=0, index_col=0)
        pd.testing.assert_frame_equal(ground_truth_df, optim.results_grid, check_exact=False,rtol=rtol, atol=atol)

    def test_off_grid_load_profile(self):
        self.tc['off_grid_load_profile'] = 'sample_off_grid_load_profile.csv'

        # Run simulation
        optim = self.run_main()

        # Compare results to ground truth data
        ground_truth_df = pd.read_csv(os.path.join(SYS_DATA_DIR, 'ground_truth',
                                                   'off_grid_load_ground_truth.csv'),
                                      header=0, index_col=0)
        pd.testing.assert_frame_equal(ground_truth_df, optim.results_grid, check_exact=False,rtol=rtol, atol=atol)
