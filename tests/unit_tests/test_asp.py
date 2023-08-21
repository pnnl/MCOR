# -*- coding: utf-8 -*-
"""Unit test for alternative_solar_profiles.py"""

import os
import math
import multiprocessing
import unittest
from unittest.mock import patch
from datetime import timedelta, datetime
import warnings

import pandas as pd
import numpy as np

import alternative_solar_profiles
from alternative_solar_profiles import AlternativeSolarProfiles
from validation import ParamValidationError
from config import UNIT_TESTS_DIR
from tests.utils.state_loader import save_state, load_state

TEST_DATA_DIR = os.path.join(UNIT_TESTS_DIR, 'data')
EXPECTED_OUTPUTS = os.path.join(TEST_DATA_DIR, 'expected_outputs')
TEST_SOLAR_DATA_DIR = os.path.join(TEST_DATA_DIR, 'solar_data')


# noinspection PyUnresolvedReferences
def count_consecutive_zeros(df):
    """
    Calculates count of consecutive zeros

    Adapted from https://stackoverflow.com/questions/27626542/
    """
    df['consec'] = df.groupby((df.temperature !=
                               df.temperature.shift()).cumsum()).cumcount() + 1
    consecutive_zero = df[df['temperature'] == 0]
    return consecutive_zero.consec


class TestAspInit(unittest.TestCase):
    def test_init(self):
        """Test default attributes are assigned correctly"""
        self.asp = AlternativeSolarProfiles(latitude=46.34,
                                            longitude=-119.28,
                                            num_trials=10.00,
                                            length_trials=3.1,
                                            validate=True)
        self.assertEqual(self.asp.latitude, 46.34)
        self.assertEqual(self.asp.longitude, -119.28)
        self.assertIsInstance(self.asp.num_trials, int)
        self.assertEqual(self.asp.num_trials, 10)

        self.assertIsInstance(self.asp.length_trials, int)
        self.assertEqual(self.asp.length_trials, 3)
        self.assertEqual(self.asp.start_year, 1998)
        self.assertEqual(self.asp.end_year, 2016)
        self.assertEqual(self.asp.num_hourly_ghi_states, 11)
        self.assertEqual(self.asp.num_hourly_dni_states, 11)
        self.assertEqual(self.asp.num_daily_ghi_states, 11)
        self.assertEqual(self.asp.num_daily_dni_states, 11)
        self.assertEqual(self.asp.cloud_hours, (10, 17))
        self.assertEqual(self.asp.temp_bins, range(-30, 50, 3))
        self.assertEqual(self.asp.max_iter, 200)
        self.assertTrue(self.asp.multithreading)

        self.assertIsNone(self.asp.nrel_data_df)
        self.assertIsNone(self.asp.hourly_states)
        self.assertIsNone(self.asp.daily_states)
        self.assertEqual(self.asp.num_iters, [])
        self.assertIsNone(self.asp.simple_prob_hourly)
        self.assertIsNone(self.asp.state_prob_hourly)
        self.assertIsNone(self.asp.simple_prob_daily)
        self.assertIsNone(self.asp.state_prob_daily)
        self.assertIsNone(self.asp.state_prob_hourly_grouped)
        self.assertIsNone(self.asp.state_prob_daily_grouped)
        self.assertIsNone(self.asp.simple_prob_hourly_grouped)
        self.assertIsNone(self.asp.simple_prob_daily_grouped)
        self.assertEqual(self.asp.solar_trials, [])

    def test_validate(self):
        """Test for invalid parameter raising validation error"""
        with self.assertRaises(ParamValidationError):
            AlternativeSolarProfiles(latitude=46.34,
                                     longitude=-119.28,
                                     num_trials=10.00,
                                     length_trials=3.1,
                                     cloud_hours='failure')


class TestASP(unittest.TestCase):
    def setUp(self) -> None:
        """Runs before every test in test class"""
        self.asp = AlternativeSolarProfiles(latitude=46.34,
                                            longitude=-119.28,
                                            num_trials=10,
                                            length_trials=3)

    def tearDown(self) -> None:
        """Runs after each test"""
        # Ensure that everything is cleaned up after each test
        self.asp = None

    @patch.object(alternative_solar_profiles, 'SOLAR_DATA_DIR', TEST_SOLAR_DATA_DIR)
    def load_partial_nrel(self):
        """Helper function to only load partial nrel data"""
        self.asp.start_year = 2008
        self.asp.end_year = 2009
        self.asp.load_nrel_data()
        self.assertFalse(self.asp.nrel_data_df.empty)

    def load_hourly_grouped_probabilities(self):
        """Load in previously calculated hourly values"""

        simple_grouped_hourly = os.path.join(EXPECTED_OUTPUTS,
                                             'exp_simple_prob_hourly_grouped.json')
        state_grouped_hourly = os.path.join(EXPECTED_OUTPUTS,
                                            'exp_state_prob_hourly_grouped.json')
        self.asp.simple_prob_hourly_grouped = load_state(simple_grouped_hourly)
        self.asp.state_prob_hourly_grouped = load_state(state_grouped_hourly)

    def load_daily_grouped_probabilities(self):
        """Load in previously calculated daily values"""

        simple_grouped_daily = os.path.join(EXPECTED_OUTPUTS,
                                            'exp_simple_prob_daily_grouped.json')
        state_grouped_daily = os.path.join(EXPECTED_OUTPUTS,
                                           'exp_state_prob_daily_grouped.json')
        self.asp.simple_prob_daily_grouped = load_state(simple_grouped_daily)
        self.asp.state_prob_daily_grouped = load_state(state_grouped_daily)

    @patch.object(alternative_solar_profiles, 'SOLAR_DATA_DIR', TEST_SOLAR_DATA_DIR)
    def test_load_nrel_data(self):
        # load only part of the data so test runs more quickly
        self.asp.end_year = 2000
        self.asp.load_nrel_data()

        # test type requirements
        self.assertTrue(self.asp.nrel_data_df['year'].dtype == np.int64)
        self.assertTrue(self.asp.nrel_data_df['month'].dtype == np.int64)
        self.assertTrue(self.asp.nrel_data_df['day'].dtype == np.int64)
        self.assertTrue(self.asp.nrel_data_df['hour'].dtype == np.int64)
        self.assertTrue(np.issubdtype(self.asp.nrel_data_df['datetime'].dtype,
                                      np.datetime64))

        # test all years loaded into df
        expected_years = list(range(1998, self.asp.end_year + 1))
        loaded_years = self.asp.nrel_data_df['year'].unique()
        self.assertTrue(np.array_equal(loaded_years, expected_years))

        # test all final columns
        expected_cols = ['year', 'month', 'day',
                         'hour', 'dni', 'ghi', 'clear_sky_dni',
                         'clear_sky_ghi', 'cloud_type',
                         'solar_zenith_angle', 'temperature', 'datetime']
        self.assertTrue(np.array_equal(self.asp.nrel_data_df.columns,
                                       expected_cols))

    def test_bad_nrel_data_load(self):
        """Test load failure for lat/lon that haven't been downloaded"""
        self.asp.latitude = 180
        self.asp.latitude = 2
        with self.assertRaises(Exception):
            self.asp.load_nrel_data()

    def test_clean_temperature(self):
        # load only two years of data so test is faster
        self.load_partial_nrel()

        # copy of original data
        unclean = self.asp.nrel_data_df.copy()

        # assert that there were lengths of consecutive zeros >= 4
        consecutive_zeroes = count_consecutive_zeros(self.asp.nrel_data_df)
        self.assertFalse(consecutive_zeroes.max() < 4)

        self.asp.clean_temperature()

        # test that all lengths of consecutive zeros now < 4
        consecutive_zeroes = count_consecutive_zeros(self.asp.nrel_data_df)
        self.assertTrue(consecutive_zeroes.max() < 4)

        # for brevity
        cleaned = self.asp.nrel_data_df

        # assert non-zero values were not altered
        self.assertFalse(np.array_equal(unclean[unclean['temperature'] != 0],
                                        cleaned[cleaned['temperature'] != 0]))

        # assert that there are now nan values
        self.assertTrue(np.any(np.isnan(cleaned.temperature)))

        # test case where no values are equal to 0
        unclean.loc[unclean.temperature == 0, 'temperature'] = np.pi
        self.asp.nrel_data_df = unclean.copy()
        self.asp.clean_temperature()

        # test that clean temperature did nothing
        self.assertTrue(np.array_equal(unclean, self.asp.nrel_data_df))

    def test_bin_hourly_data(self):
        # load only two years of data so test is faster
        self.load_partial_nrel()

        # bin the data
        self.asp.bin_hourly_data()

        expected = pd.read_csv(os.path.join(EXPECTED_OUTPUTS, 'expected_hourly_states.csv'),
                               index_col=None)

        # np.isclose used because math.exp(1) is rounded when saved as csv
        self.assertTrue(np.all(np.isclose(expected, self.asp.hourly_states)))

        # for brevity
        num_dni = self.asp.num_hourly_dni_states
        num_ghi = self.asp.num_hourly_ghi_states

        # assert states are in the allowable ranges
        # additional state is for night states (state = math.exp(1))
        expected_dni_states = np.array([math.exp(1)] + list(range(0, num_dni)))
        dni_states = self.asp.hourly_states.dni_state.unique()
        self.assertTrue(set(dni_states).issubset(set(expected_dni_states)))

        # additional state is for night states (state = math.exp(1))
        expected_ghi_states = np.array([math.exp(1)] + list(range(0, num_ghi)))
        ghi_states = self.asp.hourly_states.ghi_state.unique()
        self.assertTrue(set(ghi_states).issubset(set(expected_ghi_states)))

        allowed_temp_bins = set(range(len(self.asp.temp_bins)))
        temp_bins = set(self.asp.hourly_states.temp_state.unique())
        self.assertTrue(temp_bins.issubset(allowed_temp_bins))

        # assert the values are binned as expected
        # set new state quantities and temp_bins
        self.asp.num_hourly_dni_states = 3
        self.asp.num_hourly_ghi_states = 4
        self.asp.temp_bins = range(-30, 60, 10)

        # generate some fake data for test
        small_test = pd.DataFrame({'clear_sky_ghi': [0, 0, 260, 570, 369],
                                   'ghi': [0, 0, 130, 570, 0],
                                   'clear_sky_dni': [0, 0, 480, 926, 293],
                                   'dni': [0, 0, 80, 926, 0],
                                   'temperature': [-5, 2, 18, 12, 25],
                                   'year': [2020] * 5,
                                   'month': [11] * 5,
                                   'day': [2] * 5,
                                   'hour': list(range(5)),
                                   'cloud_type': np.random.randint(10, size=5)
                                   })

        # assign fake data to nrel dataframe
        self.asp.nrel_data_df = small_test
        self.asp.bin_hourly_data()

        # expected results of binning
        expected_dni = np.array([math.exp(1), math.exp(1), 2, 0, 2])
        expected_ghi = np.array([math.exp(1), math.exp(1), 2, 0, 3])
        expected_temp_bins = np.array([3, 4, 5, 5, 6])

        # assert expected is equal to calculated
        self.assertTrue(np.array_equal(self.asp.hourly_states.dni_state, expected_dni))
        self.assertTrue(np.array_equal(self.asp.hourly_states.ghi_state, expected_ghi))
        self.assertTrue(np.array_equal(self.asp.hourly_states.temp_state, expected_temp_bins))

    def test_bin_daily_data(self):
        self.load_partial_nrel()
        self.asp.bin_daily_data()

        expected = pd.read_csv(os.path.join(EXPECTED_OUTPUTS, 'expected_daily_states.csv'),
                               index_col=None)

        # np.isclose used because math.exp(1) is rounded when saved as csv
        self.assertTrue(np.all(np.isclose(expected, self.asp.daily_states)))

        # for brevity
        num_ghi = self.asp.num_daily_ghi_states
        num_dni = self.asp.num_daily_dni_states

        expected_dni_states = np.array(list(range(0, num_dni))) + 1
        dni_states = self.asp.daily_states.dni_state.unique()
        self.assertTrue(set(dni_states).issubset(set(expected_dni_states)))

        expected_ghi_states = np.array(list(range(0, num_ghi))) + 1
        ghi_states = self.asp.daily_states.ghi_state.unique()
        self.assertTrue(set(ghi_states).issubset(set(expected_ghi_states)))
        self.assertTrue(self.asp.daily_states is not None)

        # create fake nrel df
        dummy_clouds = np.hstack([np.zeros(24), np.zeros(10), np.ones(8), np.zeros(6)])
        dummy_ghi = np.hstack([np.zeros(10), np.ones(8) * 36, np.zeros(6), ] * 2)
        dummy_clear_ghi = np.hstack([np.zeros(10), np.ones(8) * 36, np.zeros(6), np.zeros(10),
                                     np.ones(8) * 72, np.zeros(6)])

        small_test = pd.DataFrame({'clear_sky_ghi': dummy_clear_ghi,
                                   'ghi': dummy_ghi,
                                   'clear_sky_dni': dummy_clear_ghi,
                                   'dni': dummy_ghi,
                                   'year': [2020] * 48,
                                   'month': [11] * 48,
                                   'day': [2] * 24 + [3] * 24,
                                   'hour': list(range(24)) + list(range(24)),
                                   'cloud_type': dummy_clouds
                                   })

        # assign fake data to nrel dataframe
        self.asp.nrel_data_df = small_test
        self.asp.bin_daily_data()

        # assert expected is equal to calculated
        self.assertTrue(self.asp.daily_states is not None)
        self.assertTrue(np.array_equal(self.asp.daily_states.cloud_state, np.array([0, 1])))
        self.assertTrue(np.array_equal(self.asp.daily_states.dni_state, np.array([11, 6])))
        self.assertTrue(np.array_equal(self.asp.daily_states.ghi_state, np.array([11, 6])))

    def test_create_hourly_state_transition_matrices(self):
        # read previously calculated hourly bin data
        states = pd.read_csv(os.path.join(EXPECTED_OUTPUTS, 'expected_hourly_states.csv'),
                             index_col=None).round(10)

        # replace rounded exp with actual value
        states = states.replace(round(math.exp(1), 10), math.exp(1))
        self.asp.hourly_states = states
        self.asp.create_hourly_state_transition_matrices()

        simple_file = os.path.join(EXPECTED_OUTPUTS, 'expected_simple_prob_hourly.csv')
        state_file = os.path.join(EXPECTED_OUTPUTS, 'expected_state_prob_hourly.csv')

        expected_simple = pd.read_csv(simple_file, index_col=None).round(10)
        simple_prob_hourly = self.asp.simple_prob_hourly.round(10)
        self.assertTrue(expected_simple.equals(simple_prob_hourly))

        expected_state = pd.read_csv(state_file, index_col=None).round(10)
        state_prob_hourly = self.asp.state_prob_hourly.round(10)
        self.assertTrue(expected_state.equals(state_prob_hourly))

    def test_create_daily_state_transition_matrices(self):
        # read previously calculated daily bin data
        states = pd.read_csv(os.path.join(EXPECTED_OUTPUTS, 'expected_daily_states.csv'),
                             index_col=None)
        self.asp.daily_states = states
        self.asp.create_daily_state_transition_matrices()

        simple_file = os.path.join(EXPECTED_OUTPUTS, 'expected_simple_prob_daily.csv')
        state_file = os.path.join(EXPECTED_OUTPUTS, 'expected_state_prob_daily.csv')

        expected_simple_daily = pd.read_csv(simple_file, index_col=None).round(10)
        simple_prob_daily = self.asp.simple_prob_daily.round(10)
        self.assertTrue(expected_simple_daily.equals(simple_prob_daily))

        expected_state = pd.read_csv(state_file, index_col=None).round(10)
        state_prob_daily = self.asp.state_prob_daily.round(10)
        self.assertTrue(expected_state.equals(state_prob_daily))

    def test_create_state_transition_matrices(self):
        self.asp.start_year = 2008
        self.asp.end_year = 2009
        self.asp.create_state_transition_matrices()

        # test that all negative cloud_types have been removed
        self.assertFalse(np.any(self.asp.nrel_data_df['cloud_type'] < 0))

        # assert that values have been assigned. functions tested individually
        self.assertIsNotNone(self.asp.daily_states)
        self.assertIsNotNone(self.asp.hourly_states)
        self.assertIsNotNone(self.asp.simple_prob_daily)
        self.assertIsNotNone(self.asp.state_prob_daily)
        self.assertIsNotNone(self.asp.simple_prob_hourly)
        self.assertIsNotNone(self.asp.state_prob_hourly)

    def test_generate_trial_date_ranges(self):
        self.load_partial_nrel()
        test_ranges = self.asp.generate_trial_date_ranges()

        # assert there are as many date ranges as num_trials
        self.assertEqual(len(test_ranges), self.asp.num_trials)

        # for each date range assert they have the appropriate number of
        #   datetime objects and that they are that many hours apart
        range_length = self.asp.length_trials
        for date_range in test_ranges:
            self.assertEqual(len(date_range), range_length)
            first_hour = date_range[0]
            last_hour = date_range[-1]
            self.assertEqual(first_hour + timedelta(hours=range_length - 1),
                             last_hour)

        # test date range creation with seeded ranges and ensure a leap year
        #   would be included
        ranges_start = [datetime(year=2020, month=month, day=28)
                        for month in range(1, self.asp.num_trials + 1)]
        seeded_ranges = self.asp.generate_trial_date_ranges(ranges_start)

        for date_range in seeded_ranges:
            self.assertEqual(len(date_range), range_length)

            first_hour = date_range[0]
            last_hour = date_range[-1]

            self.assertIn(first_hour, ranges_start)

            # test leap day has been removed
            self.assertNotIn(datetime(year=2020, month=2, day=29), date_range)

            # test a day has been added to replace the leap day
            if first_hour.month == 2:
                expected_offset = range_length + 23
                self.assertEqual(first_hour + timedelta(hours=expected_offset), last_hour)
            else:
                expected_offset = range_length - 1
                self.assertEqual(first_hour + timedelta(hours=expected_offset), last_hour)

    def test_preprocess_states(self):
        # get expected output files
        simple_hourly_file = os.path.join(EXPECTED_OUTPUTS, 'expected_simple_prob_hourly.csv')
        state_hourly_file = os.path.join(EXPECTED_OUTPUTS, 'expected_state_prob_hourly.csv')
        simple_daily_file = os.path.join(EXPECTED_OUTPUTS, 'expected_simple_prob_daily.csv')
        state_daily_file = os.path.join(EXPECTED_OUTPUTS, 'expected_state_prob_daily.csv')

        # load expected outputs from file
        simple_hourly = pd.read_csv(simple_hourly_file, index_col=None).round(10)
        state_hourly = pd.read_csv(state_hourly_file, index_col=None).round(10)
        simple_daily = pd.read_csv(simple_daily_file, index_col=None).round(10)
        state_daily = pd.read_csv(state_daily_file, index_col=None).round(10)

        # set binned hourly/daily probabilities to previously calculated values
        self.asp.simple_prob_hourly = simple_hourly
        self.asp.state_prob_hourly = state_hourly
        self.asp.simple_prob_daily = simple_daily
        self.asp.state_prob_daily = state_daily
        self.asp.preprocess_states()

        self.assertIsInstance(self.asp.simple_prob_hourly_grouped, dict)
        self.assertIsInstance(self.asp.state_prob_hourly_grouped, dict)
        self.assertIsInstance(self.asp.simple_prob_daily_grouped, dict)
        self.assertIsInstance(self.asp.state_prob_daily_grouped, dict)

        simple_grouped_hourly = os.path.join(EXPECTED_OUTPUTS,
                                             'exp_simple_prob_hourly_grouped.json')
        state_grouped_hourly = os.path.join(EXPECTED_OUTPUTS,
                                            'exp_state_prob_hourly_grouped.json')
        simple_grouped_daily = os.path.join(EXPECTED_OUTPUTS,
                                            'exp_simple_prob_daily_grouped.json')
        state_grouped_daily = os.path.join(EXPECTED_OUTPUTS,
                                           'exp_state_prob_daily_grouped.json')

        test_iter = ((simple_grouped_hourly,
                      self.asp.simple_prob_hourly_grouped),
                     (state_grouped_hourly,
                      self.asp.state_prob_hourly_grouped),
                     (simple_grouped_daily,
                      self.asp.simple_prob_daily_grouped),
                     (state_grouped_daily,
                      self.asp.state_prob_daily_grouped),
                     )

        for expected_out_file, actual_output in test_iter:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                expected = load_state(expected_out_file)
            for key in expected:
                self.assertTrue(np.array_equiv(expected[key], actual_output[key]))

    def test_generate_random_state_daily(self):
        # get state transition probabilities data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            self.load_daily_grouped_probabilities()

        # create date_range to generate random states for
        range_length = self.asp.length_trials
        date_range = pd.date_range(start='2009-02-09 15:00:00', periods=range_length,
                                   freq='H')

        # generate the random states
        gen_1 = self.asp.generate_random_state_daily(date_range=date_range)

        # get the number of unique days present
        num_days = len(np.unique(date_range.date))

        # assert a random state has been generated for each date present
        self.assertEqual(len(gen_1), num_days)

        # get list of states present in daily states
        states = [state[1:] for state in self.asp.state_prob_daily_grouped.keys()]

        # assert that the generated state was present in the daily states
        for ind, row in gen_1.iterrows():
            self.assertIn(tuple(row), states)

        # generate another set of random states
        gen_2 = self.asp.generate_random_state_daily(date_range=date_range)

        # assert they are not equal
        self.assertFalse(np.array_equiv(gen_1, gen_2))

    def test_generate_random_state_hourly(self):
        # load in data needed for function
        self.load_partial_nrel()

        # get state transition probabilities data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            self.load_hourly_grouped_probabilities()

        # create date_range to generate random states for
        date = datetime(year=2009, month=2, day=10).date()
        one_day = pd.date_range(start=date, periods=24, freq='H')

        clear_sky_data = self.asp.nrel_data_df.merge(
            one_day.to_frame(),
            left_on='datetime',
            right_on=0,
            how='inner').set_index('datetime')
        one_day_clear_sky_data = clear_sky_data[clear_sky_data.index.date == date]
        night_states = one_day_clear_sky_data['clear_sky_ghi'] == 0
        current_state = [None, None, None, None]

        # returns boring result because current_state is None
        rand_state_1 = self.asp.generate_random_state_hourly(one_day, night_states,
                                                             current_state)

        # generate a second set of states using first result
        rand_state_2 = self.asp.generate_random_state_hourly(one_day, night_states,
                                                             rand_state_1)

        # get list of states present in hourly states
        states = [state[3:] for state in self.asp.state_prob_hourly_grouped.keys()]

        # assert that the generated state was present in the hourly states
        for ind, row in rand_state_2.iterrows():
            self.assertIn(tuple(row), states)

        # generate a third set of states
        rand_state_3 = self.asp.generate_random_state_hourly(one_day, night_states,
                                                             rand_state_2)
        # assert they are not equal
        self.assertFalse(np.array_equiv(rand_state_2, rand_state_3))

    def test_aggregate_hourly_states(self):
        # load in data needed for function
        self.load_partial_nrel()

        # load in previously calculated hourly states df
        hourly_state = pd.read_csv(os.path.join(EXPECTED_OUTPUTS, 'rand_hourly_state.csv'),
                                   parse_dates=True, index_col=0)

        # get df of one day clear sky data
        date = datetime(year=2009, month=2, day=10).date()
        one_day = pd.date_range(start=date, periods=24, freq='H')
        clear_sky_data = self.asp.nrel_data_df.merge(
            one_day.to_frame(),
            left_on='datetime',
            right_on=0,
            how='inner').set_index('datetime')
        one_day_clear_sky_data = clear_sky_data[clear_sky_data.index.date == date]

        # aggregate values
        aggregated = self.asp.aggregate_hourly_states(one_day_clear_sky_data, hourly_state)
        expected = (9.0, 7.0, [8])
        self.assertEqual(aggregated, expected)

    def test_calc_solar_params_from_states(self):
        # load in data needed for function
        self.load_partial_nrel()

        # load in previously calculated hourly states df
        hourly_state = pd.read_csv(os.path.join(EXPECTED_OUTPUTS, 'rand_hourly_state.csv'),
                                   parse_dates=True,
                                   index_col=0)

        # get df of one day clear sky data
        date = datetime(year=2009, month=2, day=10).date()
        one_day = pd.date_range(start=date, periods=24, freq='H')
        clear_sky_data = self.asp.nrel_data_df.merge(
            one_day.to_frame(),
            left_on='datetime',
            right_on=0,
            how='inner').set_index('datetime')
        one_day_clear_sky_data = clear_sky_data[clear_sky_data.index.date == date]

        # calculate parameters
        states = self.asp.calc_solar_params_from_states(hourly_state, one_day_clear_sky_data)

        # read in previously calculated results
        expected = pd.read_csv(os.path.join(EXPECTED_OUTPUTS, 'exp_solar_params_calc.csv'),
                               parse_dates=True, index_col=0)

        # assert they are equal - round(15) for csv rounding differences
        self.assertTrue(expected.round(10).equals(states.round(10)))

    def test_compare_hourly_daily_states(self):
        # load in data needed for function
        self.load_partial_nrel()

        # get state transition probabilities data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            self.load_hourly_grouped_probabilities()

        # load in previously calculated hourly states df
        daily_state = pd.read_csv(os.path.join(EXPECTED_OUTPUTS, 'rand_daily_states.csv'),
                                  parse_dates=True, index_col=0)

        # daily state index read in as a datetime object by default; needs
        #   to be a date object.
        daily_state.index = daily_state.index.date

        # create date_range to generate random states for
        range_length = self.asp.length_trials
        date_range = pd.date_range(start='2009-02-09 15:00:00', periods=range_length,
                                   freq='H')

        # get df of clear sky data
        clear_sky_data = self.asp.nrel_data_df.merge(
            date_range.to_frame(),
            left_on='datetime',
            right_on=0,
            how='inner').set_index('datetime')

        gen_1 = self.asp.compare_hourly_daily_states(daily_state,
                                                     date_range,
                                                     clear_sky_data,
                                                     queue=None)
        hourly_states, gen_clear_sky = gen_1

        # assert that the clear sky data has been unchanged
        self.assertTrue(clear_sky_data.equals(gen_clear_sky))

        # assert that the first and last days were not iterated through
        #   (all hours not present)
        self.assertEqual(self.asp.num_iters[0], 0)
        self.assertEqual(self.asp.num_iters[-1], 0)

        # assert that the number of iterations are less than the max num
        self.assertTrue(np.all(np.less_equal(self.asp.num_iters, self.asp.max_iter)))

        # test that hourly states have been generated for each hour in trial
        self.assertEqual(len(hourly_states), range_length)

        # test concurrent version of function
        queue = multiprocessing.Queue()
        self.asp.compare_hourly_daily_states(daily_state,
                                             date_range,
                                             clear_sky_data,
                                             queue=queue)
        mp_hourly_states, mp_clear_sky = queue.get()

        # same tests as before but with concurrent version
        self.assertTrue(clear_sky_data.equals(mp_clear_sky))
        self.assertEqual(len(mp_hourly_states), range_length)

    @patch.object(alternative_solar_profiles.AlternativeSolarProfiles,
                  'preprocess_states')
    def test_create_trial_data(self, fake_preprocess):

        # patch preprocess_states to save time
        fake_preprocess.return_value = None

        # reduce number of trials to save time
        self.asp.num_trials = 2

        # load in data needed for function
        self.load_partial_nrel()

        # get state transition probabilities data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            self.load_hourly_grouped_probabilities()
            self.load_daily_grouped_probabilities()
        self.asp.create_trial_data()

        self.assertIsNotNone(self.asp.solar_trials)
        self.assertEqual(len(self.asp.solar_trials), self.asp.num_trials)


if __name__ == '__main__':
    asp = AlternativeSolarProfiles(latitude=46.34,
                                   longitude=-119.28,
                                   num_trials=10,
                                   length_trials=3)