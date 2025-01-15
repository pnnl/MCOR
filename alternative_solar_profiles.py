# -*- coding: utf-8 -*-
"""
Alternative Solar Profiles (ASP) algorithm used for solar forecasting:
Original author in MATLAB: James Follum and Trevor Hardy

The algorithm creates a probabilistic model for the solar "state" (defined by the ghi, dni,
cloud type, and temperature values) for each month on daily and hourly timescales based on
historical data downloaded from NREL's NRSDB: https://nsrdb.nrel.gov/

To create solar trials, the algorithm randomly samples the model for a given month and hour
given the probability of transitioning from one "state" to another "state". States are first
created for each day of a trial, and then hourly states are generated iteratively until a set
of hourly states is found that matches the daily state or the maximum number of iterations
is reached. This maximum number is set to 200 by default, as testing showed that for 85% of
days, 300 iterations was sufficient to reach agreement.

File contents:
    Classes:
        AlternativeSolarProfiles

    Standalone functions:
        date_parser

"""

import numpy as np
import pandas as pd
import random 
import math
import os
import multiprocessing
from scipy import stats
import datetime as dt
from validation import validate_all_parameters, log_error
from config import SOLAR_DATA_DIR


class AlternativeSolarProfiles:
    """
    Class to create solar profiles using a probabilistic model based on historical solar data.

    Parameters
    ----------

        longitude: Site longitude in degrees

        latitude: Site latitude in degrees

        num_trials: Number of solar profiles to create

        length_trials: Length of solar profiles in hours

        start_year: Start year for solar data download

        end_year: End year for solar data download

        num_hourly_ghi_states: Number of discrete GHI states for hourly model

        num_hourly_dni_states: Number of discrete DNI states for hourly model

        num_daily_ghi_states: Number of discrete GHI states for daily model

        num_daily_dni_states: Number of discrete DNI states for daily model

        cloud_hours: Tuple containing maximum and minimum hours of the day to be used to set
            the day's cloud state

        temp_bins: Temperature bins

        max_iter: The maximum number of iterations allowed for trying to match hourly and
            daily states. After this number is reached the best set of hourly states generated
            thus far is used.

        multithreading: Whether to use multithreading to speed up the calculation. This is
            set to True by default, but should be set to False for debugging

    Methods
    ----------

        create_state_transition_matrices: Creates monthly state transition matrices from
            historical NREL solar data to hold the probability of transitioning from one solar
            state to another.

        load_nrel_data: Loads pre-downloaded NREL solar data from csv files.

        bin_hourly_data: Bins hourly historical solar data into buckets based on ghi, dni,
            cloud type, and temperature values.

        bin_daily_data: Bins daily historical solar data into buckets based on ghi, dni, cloud
            type, and temperature values.

        clean_temperature: Replaces consecutive 0s in historical temperature data with nans,
            with a threshold at 4 or more 0s in a row.

        create_hourly_state_transition_matrices: Creates an hourly state transition matrix,
            based on historical solar data. The transition matrix specifies the probability of
            transition from one solar state (determined by GHI, DNI, cloud cover type, and
            temperature) to another, for each hour and month of the year.

        create_daily_state_transition_matrices: Creates a daily state transition matrix, based
            on historical solar data. The transition matrix specifies the probability of
            transition from one solar state (determined by GHI, DNI, and cloud cover type) to
            another, for each month of the year.

        create_trial_data: Simulates solar profiles based on hourly and daily transition
            matrices, which specify the probability of transitioning from a given solar state
            (determined by GHI, DNI,temperature, and cloud cover type) to another for each
            hour and month of the year.

        calc_solar_params_from_states: Calculates solar params (ghi, dni, temp) from hourly
            states and clear sky data.

        generate_trial_date_ranges: Generates date_ranges for each trial and removes leap
            days.

        preprocess_states: Groups state dataframes to speed up creation of trials.

        generate_random_state_daily: Generates random daily states for a solar trial.

        compare_hourly_daily_states: Iteratively generates hourly states, checking for
            consistency with the daily state.

        generate_random_state_hourly: Generates random hourly states for a solar trial.

        aggregate_hourly_states: Aggregates hourly states to a daily state for comparison.

    Calculated Attributes
    ----------

        nrel_data_df: Dataframe with historical data downloaded from NREL's NRSDB

        hourly_states: Dataframe with binned hourly historical solar data

        daily_states: Dataframe with binned daily historical solar data

        simple_prob_hourly: Dataframe containing every unique set of month, hour, and
            to-states, along with the probability of getting to that to-state for that
            month/hour, independent of a from-state.

        state_prob_hourly: Dataframe containing every unique set of month, hour, from-
            (previous hour) states and to- (current hour) states, along with counts and
            probabilities. This is the probability of getting a certain to-state given the
            from-state.

        simple_prob_daily: Dataframe containing every unique set of month and to-states, along
            with the probability of getting to that to-state for that month, independent of a
            from-state.

        state_prob_daily: Dataframe containing every unique set of month, from- (previous
            hour) states, and to- (current hour) states, along with counts and probabilities.
            This is the probability of getting a certain to-state given the from-state.

        num_iters: Keeps track of the number of iterations required for the daily and hourly
            states to match.

        state_prob_hourly_grouped: Dictionary containing state_prob_hourly data grouped by
            month, hour, night state, and from-state. This is stored as a dictionary to speed
            up the calculation.

        state_prob_daily_grouped: Dictionary containing state_prob_daily data grouped by month
            and from-state.

        simple_prob_hourly_grouped: Dictionary containing simple_prob_hourly data grouped by
            month, hour, and night state.

        simple_prob_daily_grouped: Dictionary containing simple_prob_daily data grouped by
            month.

        solar_trials: Holds generated trial data.

    """

    def __init__(self, latitude, longitude, num_trials, length_trials, start_year=1998,
                 end_year=2016, num_hourly_ghi_states=11, num_hourly_dni_states=11,
                 num_daily_ghi_states=11, num_daily_dni_states=11, cloud_hours=(10, 17),
                 temp_bins=range(-30, 50, 3), max_iter=200, multithreading=True,
                 solar_data_dict = {}, validate=True):

        # Assign parameters from arguments
        self.latitude = latitude
        self.longitude = longitude
        self.num_trials = int(num_trials)
        self.length_trials = int(length_trials)
        self.start_year = int(start_year)
        self.end_year = int(end_year)
        self.num_hourly_ghi_states = num_hourly_ghi_states
        self.num_hourly_dni_states = num_hourly_dni_states
        self.num_daily_ghi_states = num_daily_ghi_states
        self.num_daily_dni_states = num_daily_dni_states
        self.cloud_hours = cloud_hours
        self.temp_bins = temp_bins
        self.max_iter = max_iter
        self.multithreading = multithreading
        self.solar_data_dict = solar_data_dict

        # Initialize other attributes
        self.nrel_data_df = None
        self.hourly_states = None
        self.daily_states = None
        self.simple_prob_hourly = None
        self.state_prob_hourly = None
        self.simple_prob_daily = None
        self.state_prob_daily = None
        self.num_iters = []
        self.state_prob_hourly_grouped = None
        self.state_prob_daily_grouped = None
        self.simple_prob_hourly_grouped = None
        self.simple_prob_daily_grouped = None
        self.solar_trials = []

        # Validate input arguments
        if validate:
            # List of initialized parameters to validate
            args_dict = {'latitude': self.latitude,
                         'longitude': self.longitude,
                         'num_trials': self.num_trials,
                         'length_trials': self.length_trials,
                         'start_year': self.start_year,
                         'end_year': self.end_year,
                         'num_ghi_states': self.num_hourly_ghi_states,
                         'num_dni_states': self.num_hourly_dni_states,
                         'num_daily_ghi_states': self.num_daily_ghi_states,
                         'num_daily_dni_states': self.num_daily_dni_states,
                         'cld_hours': self.cloud_hours,
                         'temp_bins': self.temp_bins,
                         'max_iter': self.max_iter,
                         'multithreading': self.multithreading,
                         'solar_data_dict': self.solar_data_dict}

            # Validate input parameters
            validate_all_parameters(args_dict)

    def create_state_transition_matrices(self):
        """
        Create monthly state transition matrices from historical NREL solar data.

        inputs
            self.latitude: Site latitude in degrees
            self.longitude: Site longitude in degrees
            self.start_year: Start year for solar data download
            self.end_year: End year for solar data download
            self.num_hourly_ghi_states: Number of discrete GHI states for hourly model
            self.num_hourly_dni_states: Number of discrete DNI states for hourly model
            self.temp_bins: Bins for temperature data
            self.num_daily_ghi_states: Number of discrete GHI states for daily model
            self.num_daily_dni_states: Number of discrete DNI states for daily model
            self.cloud_hours: Hours of the day (range) used to set the day's cloud state

        outputs
            self.state_prob_hourly: Dataframe containing every unique set of month, hour,
                from- (previous hour) states and to- (current hour) states, along with counts
                and probabilities. This is the probability of getting a certain to-state given
                the from-state.
            self.simple_prob_hourly: Dataframe containing every unique set of month, hour, and
                to-states, along with the probability of getting that to-state for that
                month/hour, independent of a from-state.
            self.state_prob_daily: Dataframe containing every unique set of month, from-
                (previous hour) states and to- (current hour) states, along with counts and
                probabilities. This is the probability of getting a  certain to-state given
                the from-state.
            self.simple_prob_daily: Dataframe containing every unique set of month and
                to-states, along with the probability of getting that to-state for that
                month, independent of a from-state.
        """

        # Read in historical NREL solar data
        self.load_nrel_data()

        # Clean data
        self.clean_temperature()
        self.nrel_data_df.loc[self.nrel_data_df['cloud_type'] < 0, 'cloud_type'] = \
            self.nrel_data_df.loc[self.nrel_data_df['cloud_type'] < 0, 'cloud_type'].apply(
                lambda x: random.randrange(11))

        # Bin data into buckets
        self.bin_hourly_data()
        self.bin_daily_data()

        # Create state transition matrices
        self.create_hourly_state_transition_matrices()
        self.create_daily_state_transition_matrices()

    def load_nrel_data(self):
        """
        Load pre-downloaded NREL solar data from csv files.

        inputs
            self.latitude: Site latitude in degrees
            self.longitude: Site longitude in degrees
            self.start_year: Start year for solar data download
            self.end_year: End year for solar data download

        outputs
            self.nrel_data_df: Dataframe with historical data downloaded from NREL's NRSDB

        """
        # Read in historical NREL solar data
        if not len(self.solar_data_dict):
            filedir = os.path.join(SOLAR_DATA_DIR, 'nrel',
                                '{}_{}'.format(self.latitude, self.longitude))

            # Check that files exist in directory
            if '{}_{}_{}.csv'.format(self.latitude, self.longitude,
                                    self.start_year) not in os.listdir(filedir):
                message = 'NREL data not found, check the latitude and longitude '\
                        'or run get_solar_data() first.'
                log_error(message)
                raise Exception(message)

            # Read in files and add to dataframe
            self.nrel_data_df = pd.DataFrame()
            for year in range(self.start_year, self.end_year+1):
                self.nrel_data_df = pd.concat([self.nrel_data_df, pd.read_csv(
                    os.path.join(filedir, '{}_{}_{}.csv'.format(
                        self.latitude, self.longitude, year)),
                    usecols=['Year', 'Month', 'Day', 'Hour', 'DNI', 'GHI',
                            'Clearsky DNI', 'Clearsky GHI', 'Cloud Type',
                            'Solar Zenith Angle', 'Temperature'])])
        else:
            self.nrel_data_df = pd.DataFrame()
            for year in range(self.start_year, self.end_year+1):
                self.solar_data_dict[year] = self.solar_data_dict[year][
                    ['Year', 'Month', 'Day', 'Hour', 'DNI', 'GHI', 'Clearsky DNI', 
                     'Clearsky GHI', 'Cloud Type', 'Solar Zenith Angle', 'Temperature']]
                self.nrel_data_df = pd.concat([self.nrel_data_df, self.solar_data_dict[year]])
        self.nrel_data_df['datetime'] = self.nrel_data_df.apply(
            lambda x: date_parser(x['Month'], x['Day'], x['Year'], x['Hour']), axis=1)
        self.nrel_data_df.columns = ['year', 'month', 'day',
                                     'hour', 'dni', 'ghi', 'clear_sky_dni',
                                     'clear_sky_ghi', 'cloud_type',
                                     'solar_zenith_angle', 'temperature', 'datetime']

        # Recast month, hour, year, day datatypes - this doesn't work with
        #   parse_dates
        self.nrel_data_df['year'] = self.nrel_data_df['year'].astype(np.int64)
        self.nrel_data_df['month'] = self.nrel_data_df['month'].astype(np.int64)
        self.nrel_data_df['day'] = self.nrel_data_df['day'].astype(np.int64)
        self.nrel_data_df['hour'] = self.nrel_data_df['hour'].astype(np.int64)

    def bin_hourly_data(self):
        """
        Bin hourly historical solar data into buckets based on ghi, dni, and temperature
            values.

        inputs
            self.nrel_data_df: Downloaded historical solar data
            self.num_hourly_ghi_states: Number of discrete GHI states for hourly model
            self.num_hourly_dni_states: Number of discrete DNI states for hourly model
            self.temp_bins: Bins for temperature data

        outputs
            self.hourly_states: Dataframe with binned hourly historical solar data

        """

        # Calculate states (bins) for ghi and dni
        hourly_data = self.nrel_data_df.copy(deep=True)
        hourly_data['ghi_state'] = np.round(
            (hourly_data['clear_sky_ghi'] - hourly_data['ghi']) /
            hourly_data['clear_sky_ghi'] *
            (self.num_hourly_ghi_states - 1)).fillna(math.exp(1))
        hourly_data['dni_state'] = np.round(
            (hourly_data['clear_sky_dni'] - hourly_data['dni']) /
            hourly_data['clear_sky_dni'] *
            (self.num_hourly_dni_states - 1)).fillna(math.exp(1))

        # Gets rid of any negative states arising from, e.g., dni > clear sky dni
        hourly_data.loc[hourly_data['ghi_state'] < 0, 'ghi_state'] = 0
        hourly_data.loc[hourly_data['dni_state'] < 0, 'dni_state'] = 0

        # Calculate states for temperature data
        hourly_data['temp_state'] = hourly_data['temperature'].apply(
            lambda x: np.digitize(x, self.temp_bins))
        hourly_data.loc[np.isnan(hourly_data['temperature']), 'temp_state'] = np.nan

        # Clean up columns and index
        self.hourly_states = hourly_data[['year', 'month', 'day', 'hour',
                                          'ghi_state', 'dni_state',
                                          'cloud_type', 'temp_state']]
        self.hourly_states.index = range(len(self.hourly_states))

    def bin_daily_data(self):
        """
        Bin daily historical solar data into buckets based on ghi, dni, cloud type, and
            temperature values. Bins represent the percentage (as a decimal) of the available
            radiation that was observed during the day.

       inputs
            self.nrel_data_df: Downloaded historical solar data
            self.num_daily_ghi_states: Number of discrete GHI states for daily model
            self.num_daily_dni_states: Number of discrete DNI states for daily model
            self.cloud_hours: Hours of the day (range) used to set the day's cloud state

        outputs
            self.daily_states: Dataframe with binned daily historical solar data

        """

        # Aggregate historical data to daily values
        daily_data = self.nrel_data_df.groupby(['year', 'month', 'day']).sum(numeric_only=True).reset_index()

        # Calculate states (bins) for daily ghi and dni data
        daily_data['ghi_state'] = np.round(daily_data['ghi'] /
                                           daily_data['clear_sky_ghi'] *
                                           (self.num_daily_ghi_states-1)) + 1
        daily_data['dni_state'] = np.round(daily_data['dni'] /
                                           daily_data['clear_sky_dni'] *
                                           (self.num_daily_dni_states-1)) + 1

        # Daily cloud data: get most common cloud type for each day from daylight hours
        daily_cloud_type = self.nrel_data_df.loc[
            (self.nrel_data_df['hour'] >= self.cloud_hours[0]) &
            (self.nrel_data_df['hour'] < self.cloud_hours[1])].groupby(
            ['year', 'month', 'day'])['cloud_type'].apply(
            lambda x: stats.mode(x).mode)
        daily_data = daily_data.merge(
            daily_cloud_type.reset_index(name='cloud_state'),
            left_on=['year', 'month', 'day'],
            right_on=['year', 'month', 'day'])
        self.daily_states = daily_data[['year', 'month', 'day', 'ghi_state',
                                        'dni_state', 'cloud_state']]

    def clean_temperature(self):
        """
        Replaces consecutive 0s in historical temperature data with nans, with a threshold at
            4 or more 0s in a row.

        inputs
            self.nrel_data_df

        outputs
            self.nrel_data_df

        """

        # Get number of 0s to keep
        temp = self.nrel_data_df['temperature'].copy(deep=True)
        temp.index = range(len(temp))

        # Get 0s from temperature array
        temp_zeros_index = temp[temp == 0].index
        if not len(temp_zeros_index):
            return

        # Get the start and end index of each group of consecutive 0s
        diff_index = temp_zeros_index[1:] - temp_zeros_index[:-1]
        group_end_index = np.where(diff_index > 1)[0]
        group_start_index = np.append([0], group_end_index+1)
        group_end_index = np.append(group_end_index, len(temp_zeros_index)-1)

        # Get the length of each group of 0s
        group_lengths = group_end_index - group_start_index + 1

        # Sort groups of 0s based on length
        groups_lengths_df = pd.Series(group_lengths).sort_values(ascending=False)
        groups_sorted = groups_lengths_df.values
        group_start_index = group_start_index[groups_lengths_df.index]
        group_end_index = group_end_index[groups_lengths_df.index]

        # Replace all groups of 4 or more 0s
        last_group = np.where(groups_sorted < 4)[0][0]

        # For each group to replace, set 0s to NaNs
        for group_index in range(last_group):
            temp.iloc[temp_zeros_index[group_start_index[group_index]]:
                      temp_zeros_index[group_end_index[group_index]]+1] = np.nan
        self.nrel_data_df['temperature'] = temp.values

    def create_hourly_state_transition_matrices(self):
        """
        Creates an hourly state transition matrix, based on historical solar data. The
            transition matrix specifies the probability of transition from one solar state
            (determined by GHI, DNI, cloud cover type, and temperature) to another, for each
            hour and month of the year.

        inputs:
            self.hourly_states: ghi, dni, cloud, and temperature states (binned values) for
                each hour from the historical data

        outputs:
            self.state_prob_hourly: dataframe containing every unique set of month, hour,
                from- (previous hour) states and to- (current hour) states, along with counts
                and probabilities. This is the probability of getting a certain to-state given
                the from-state.
            self.simple_prob_hourly: dataframe containing every unique set of month, hour, and
                to-states, along with the probability of getting that to-state for that
                month/hour, independent of a from-state.

        """

        # Create dataframe with from- and to-states for each hour
        from_and_to_states = self.hourly_states.iloc[1:]
        from_and_to_states.index = range(len(from_and_to_states))
        from_and_to_states = from_and_to_states.merge(
            self.hourly_states.iloc[:-1][['ghi_state', 'dni_state',
                                          'cloud_type', 'temp_state']],
            right_index=True, left_index=True, suffixes=['_to', '_from'])

        # Group by month, hour, from-, and to-states to get probabilities
        state_prob = from_and_to_states.groupby(
            ['month', 'hour', 'ghi_state_to', 'dni_state_to', 'cloud_type_to',
             'temp_state_to', 'ghi_state_from', 'dni_state_from',
             'cloud_type_from', 'temp_state_from'])['year'].count().to_frame(
            name='count').reset_index()

        # Divide by the total number of to-states for each from-state to get a probability
        sum_of_to_states_for_each_from_state = state_prob.groupby(
            ['month', 'hour', 'ghi_state_from', 'dni_state_from',
             'cloud_type_from', 'temp_state_from'])['count'].sum().reset_index()
        state_prob = state_prob.merge(
            sum_of_to_states_for_each_from_state,
            left_on=['month', 'hour', 'ghi_state_from', 'dni_state_from',
                     'cloud_type_from', 'temp_state_from'],
            right_on=['month', 'hour', 'ghi_state_from', 'dni_state_from',
                      'cloud_type_from', 'temp_state_from'],
            suffixes=['', '_sum'])
        state_prob['prob'] = state_prob['count'] / state_prob['count_sum']

        # Get a list of all to-states for each month and hour to calculate the simple
        #   probability of transitioning to a given to-state independent of the from-state
        simple_prob = from_and_to_states.groupby(
            ['month', 'hour', 'ghi_state_to', 'dni_state_to', 'cloud_type_to',
             'temp_state_to'])['year'].count().to_frame(name='count').reset_index()
        sum_of_to_states_for_each_hour_month = simple_prob.groupby(
            ['month', 'hour'])['count'].sum().reset_index()
        simple_prob = simple_prob.merge(
            sum_of_to_states_for_each_hour_month,
            left_on=['month', 'hour'], right_on=['month', 'hour'],
            suffixes=['', '_sum'])
        simple_prob['prob'] = simple_prob['count'] / simple_prob['count_sum']

        # Add a column to each probabilty dataframe indicating if it is night
        simple_prob['night_state'] = ((simple_prob['ghi_state_to'] == math.exp(1)) |
                                      (simple_prob['dni_state_to'] == math.exp(1)))
        state_prob['night_state'] = ((state_prob['ghi_state_to'] == math.exp(1)) |
                                     (state_prob['dni_state_to'] == math.exp(1)))

        # Save to class attributes
        self.simple_prob_hourly = simple_prob
        self.state_prob_hourly = state_prob

    def create_daily_state_transition_matrices(self):
        """
        Creates a daily state transition matrix, based on historical solar data. The
            transition matrix specifies the probability of transition from one solar state
            (determined by GHI, DNI, and cloud cover type) to another, for each month of the
            year.

        inputs:
            self.daily_states: ghi, dni, and cloud states (binned values) for each day from
                the historical data

        outputs:
            self.state_prob_daily: dataframe containing every unique set of month, from-
                (previous hour) states and to- (current hour) states, along with counts and
                probabilities. This is the probability of getting a certain to-state given the
                from-state.
            self.simple_prob_daily: dataframe containing every unique set of month and
                to-states, along with the probability of getting that to-state for that
                month, independent of a from-state.

        """

        # Create dataframe with from- and to-states for each day
        from_and_to_states = self.daily_states.iloc[1:]
        from_and_to_states.index = range(len(from_and_to_states))
        from_and_to_states = from_and_to_states.merge(
            self.daily_states.iloc[:-1][['ghi_state', 'dni_state', 'cloud_state']],
            right_index=True, left_index=True, suffixes=['_to', '_from'])

        # Group by month, from-, and to-states to get probabilities
        state_prob = from_and_to_states.groupby(
            ['month', 'ghi_state_to', 'dni_state_to', 'cloud_state_to',
             'ghi_state_from', 'dni_state_from', 'cloud_state_from'])[
            'year'].count().to_frame(name='count').reset_index()

        # Divide by the total number of to-states for each from-state to get a probability
        sum_of_to_states_for_each_from_state = state_prob.groupby(
            ['month', 'ghi_state_from', 'dni_state_from',
             'cloud_state_from'])['count'].sum().reset_index()
        state_prob = state_prob.merge(
            sum_of_to_states_for_each_from_state,
            left_on=['month', 'ghi_state_from', 'dni_state_from', 'cloud_state_from'],
            right_on=['month', 'ghi_state_from', 'dni_state_from', 'cloud_state_from'],
            suffixes=['', '_sum'])
        state_prob['prob'] = state_prob['count'] / state_prob['count_sum']

        # Get a list of all to-states for each month to calculate the simple probability of
        #   transitioning to a given to-state independent of the from-state
        simple_prob = from_and_to_states.groupby(
            ['month', 'ghi_state_to', 'dni_state_to', 'cloud_state_to'])[
            'year'].count().to_frame(name='count').reset_index()
        sum_of_to_states_for_each_month = simple_prob.groupby(
            ['month'])['count'].sum().reset_index()
        simple_prob = simple_prob.merge(
            sum_of_to_states_for_each_month, left_on=['month'],
            right_on=['month'], suffixes=['', '_sum'])
        simple_prob['prob'] = simple_prob['count'] / simple_prob['count_sum']

        # Save to class attributes
        self.simple_prob_daily = simple_prob
        self.state_prob_daily = state_prob

    def create_trial_data(self, start_datetimes=None, validate=True):
        """
        Simulates solar profiles based on hourly and daily transition matrices, which specify
            the probability of transitioning from a given solar state (determined by GHI, DNI,
            temperature, and cloud cover type) to another for each hour and month of the year.

        inputs:
            self.nrel_data_df: historical solar data
            self.simple_prob_hourly: dataframe holding the probabilities of transitioning to a
                given hourly state, independent of the from-state.
            self.simple_prob_daily: dataframe holding the probabilities of transitioning to a
                given daily state, independent of the from-state.
            self.state_prob_hourly: dataframe holding the probabilities of transition to a
                given hourly state, given the from-state.
            self.state_prob_daily: dataframe holding the probabilities of transition to a
                given daily state, given the from-state.
            self.num_trials: number of solar profiles to generate
            self.length_trials: length of profiles in hours

        outputs:
            self.solar_trials: dict containing a dataframe for each trial

        """

        # Validate input arguments
        if validate and start_datetimes is not None:
            # List of initialized parameters to validate
            args_dict = {'num_trials': self.num_trials,
                         'start_year': self.start_year,
                         'end_year': self.end_year,
                         'start_datetimes': start_datetimes}

            # Validate input parameters
            validate_all_parameters(args_dict)

        # Generate date ranges for each trial
        date_ranges = self.generate_trial_date_ranges(start_datetimes)

        # Group state dataframes to speed up trial creation
        self.preprocess_states()

        # Generate daily states
        input_list = []
        for date_range in date_ranges:
            daily_state = self.generate_random_state_daily(date_range)
            clear_sky_data = self.nrel_data_df.merge(
                date_range.to_frame(), left_on='datetime', right_on=0,
                how='inner').set_index('datetime')
            input_list += [(daily_state, date_range, clear_sky_data)]

        # Iterate through each day and create hourly states
        hourly_states_list = []

        # ...with parallelization
        if self.multithreading:
            # Create list to hold processes and Queue for multithreading
            procs = []
            queue = multiprocessing.Queue()

            for daily_state, date_range, clear_sky_data in input_list:
                proc = multiprocessing.Process(
                        target=self.compare_hourly_daily_states,
                        args=(daily_state, date_range, clear_sky_data, queue))
                procs.append(proc)
                proc.start()

            # Unpack results from Queue
            for proc in procs:
                hourly_states = queue.get()
                hourly_states_list.append(hourly_states)

            # Wait until all processes have finished
            for proc in procs:
                proc.join()

        # ...without parallelization
        else:
            for daily_state, date_range, clear_sky_data in input_list:
                hourly_states_list += [self.compare_hourly_daily_states(
                    daily_state, date_range, clear_sky_data)]

        # Calculate hourly ghi, dni, and temperature values from states
        for hourly_states, clear_sky_data in hourly_states_list:
            self.solar_trials += [self.calc_solar_params_from_states(
                hourly_states, clear_sky_data)]

    def calc_solar_params_from_states(self, hourly_states, clear_sky_data):
        """
        Calculate solar params (ghi, dni, temp) from hourly states and clear sky data.

        inputs:
            hourly_states: Dataframe with hourly state data for a given trial
            clear_sky_data: historical solar data for the same time period as the hourly
                states
            self.num_hourly_ghi_states: Number of discrete GHI states for hourly model
            self.num_hourly_dni_states: Number of discrete DNI states for hourly model
            self.temp_bins: Bins for temperature data

        outputs:
            hourly_states: Dataframe with hourly ghi, dni, cloud type, and temperature values
                for a given trial

        """

        # Merge hourly states with clearsky data from the same period
        hourly_states = hourly_states.merge(clear_sky_data[[
            'clear_sky_ghi', 'clear_sky_dni']], right_index=True, left_index=True)

        # Calculate solar params from clear sky data and solar states
        hourly_states['ghi'] = hourly_states['clear_sky_ghi'] * \
            (1 - hourly_states['ghi_state'] / (self.num_hourly_ghi_states - 1))
        hourly_states['dni'] = hourly_states['clear_sky_dni'] * \
            (1 - hourly_states['dni_state'] / (self.num_hourly_dni_states - 1))
        hourly_states['temp'] = np.diff(self.temp_bins)[0] * \
            (hourly_states['temp_state'] - 0.5) + self.temp_bins[0]
        return hourly_states[['ghi', 'dni', 'cloud_state', 'temp']]

    def generate_trial_date_ranges(self, start_datetimes=None):
        """
        Generate date_ranges for each trial and remove leap days. Allows for an input argument
            specifying the start datetimes to use.

        inputs
            start_datetimes: list of datetime objects. If set as None (default), these are
                generated randomly.
            self.nrel_data_df: historical solar data
            self.length_trials: number of hours in each trial
            self.num_trials: number of trials to generate

        outputs
            date_ranges: list of Pandas date_range objects

        """

        # Randomly generate start times for each trial
        if start_datetimes is None:
            start_datetimes = self.nrel_data_df.iloc[:-self.length_trials].sample(
                self.num_trials)['datetime'].values

        # Create a date range object for each start datetime
        date_ranges = [pd.date_range(start=start_date,
                                     periods=self.length_trials,
                                     freq='h')
                       for start_date in start_datetimes]

        # Remove any leap days and pad the date_range accordingly
        for i, date_range in enumerate(date_ranges):
            replace_timesteps = len(date_range[(date_range.month == 2) &
                                               (date_range.day == 29)])
            if replace_timesteps:
                # Remove 2/29
                date_range = date_range[date_range.date != dt.date(date_range[0].year, 2, 29)]

                # Add more timesteps
                date_ranges[i] = date_range.append(pd.date_range(
                    date_range[-1] + dt.timedelta(hours=1),
                    periods=replace_timesteps, freq='h'))

        return date_ranges

    def preprocess_states(self):
        """
        Group state dataframes to speed up creation of trials.

        inputs
            self.state_prob_hourly: Dataframe containing every unique set of month, hour,
                from- (previous hour) states and to- (current hour) states, along with counts
                and probabilities.
            self.simple_prob_hourly: Dataframe containing every unique set of month, hour,
                and to- (current hour) states, along with counts and probabilities.
            self.simple_prob_hourly: Dataframe containing every unique set of month, from-
                (previous hour) states and to-  (current day) states, along with counts and
                probabilities.
            self.simple_prob_daily: Dataframe containing every unique set of month and to-
                (current day) states, along with counts and probabilities.

        outputs
            (Dictionaries containing groupings of the input parameters)
            self.state_prob_hourly_grouped: Dict keys: (month, hour, night_state, ghi_state,
                dni_state, cloud_type, temp_state)
            self.simple_prob_hourly_grouped: Dict keys: (month, hour, night_state)
            self.state_prob_daily_grouped: Dict keys: (month, ghi_state, dni_state,
                cloud_type)
            self.simple_prob_daily_grouped: Dict keys: (month)

        """

        # Create pandas DataFrameGroupBy objects
        state_prob_hourly_groups = self.state_prob_hourly.groupby([
            'month', 'hour', 'night_state', 'ghi_state_from',
            'dni_state_from', 'cloud_type_from', 'temp_state_from'])
        simple_prob_hourly_groups = self.simple_prob_hourly.groupby([
            'month', 'hour', 'night_state'])
        state_prob_daily_groups = self.state_prob_daily.groupby([
            'month', 'ghi_state_from', 'dni_state_from', 'cloud_state_from'])
        simple_prob_daily_groups = self.simple_prob_daily.groupby([
            'month'])

        # Parse into dictionaries for faster access
        self.state_prob_hourly_grouped = {
            label: dataframe for label, dataframe in state_prob_hourly_groups}
        self.simple_prob_hourly_grouped = {
            label: dataframe for label, dataframe in simple_prob_hourly_groups}
        self.state_prob_daily_grouped = {
            label: dataframe for label, dataframe in state_prob_daily_groups}
        self.simple_prob_daily_grouped = {
            label: dataframe for label, dataframe in simple_prob_daily_groups}

    def generate_random_state_daily(self, date_range):
        """
        Generate random daily states for a solar trial.

        inputs:
            date_range: datetimeindex object for the trial
            self.length_trials: length of profiles in hours
            self.state_prob_daily_grouped: dataframe holding the probabilities of transition
                to a given daily state, given the from-state, grouped by month and from-state.
            self.simple_prob_daily_grouped: dataframe holding the probabilities of
                transitioning to a given daily state, independent of the from-state, grouped
                by month.

        outputs:
            dataframe with daily trial states

        """

        # Create lists to hold generated states and the current state
        current_state = [None, None, None]
        state_list = []

        # Create daily states iteratively
        sorted_dates = np.sort(list(set(date_range.date)))
        for date in sorted_dates:
            try:
                # Get the numpy values from the subset of states with the corresponding month
                #   and from-states. While the state data is stored in a pandas dataframe, the
                #   following operations are carried out on numpy arrays to speed up the
                #   calculation.
                # Grouped daily states have labels of the form:
                #   (month, ghi_state_from, dni_state_from, cloud_state_from)
                states_subset = self.state_prob_daily_grouped[
                    tuple([date.month] + list(current_state))].values

            # Except statement catches cases where state subset is empty (i.e. if the
            #   from-state does not exist in the corresponding month) or the first day in the
            #   trial, in which case it uses the simple probability dataframe which doesn't
            #   require a from-state.
            except KeyError:
                # Same as above, but for simple probabilities
                states_subset = self.simple_prob_daily_grouped[
                    tuple([date.month])].values

            # Get the probabilities for this subset of states
            probs = states_subset[:, -1].astype(float)

            # Randomly select a state based on the probabilities
            sample_index = np.random.choice(range(len(states_subset)),
                                            p=probs / np.sum(probs))

            # Assign the current state and add to list
            current_state = states_subset[sample_index, 1:4]
            state_list += [current_state]

        return pd.DataFrame(data=state_list, index=sorted_dates,
                            columns=['ghi_state', 'dni_state', 'cloud_state'])

    def compare_hourly_daily_states(self, daily_states, date_range,
                                    clear_sky_data, queue=None):
        """
        Iteratively generates hourly states, checking for consistency with the daily state.
            An hourly state is chosen if the aggregated ghi, dni, and cloud state values match
            that ofthe daily state or the maximum number of iterations is reached. If the
            latter, the generated state which matches the most daily state values is used.

        The hourly states are only checked for consistency with the daily state if the current
            day contains all of the "cloud hours", i.e. if a trial starts at 2pm, this check
            is not run for the first day.

        inputs:
            daily_states: Dataframe with daily states for a trial
            date_range: Pandas date_range object for the trial
            clear_sky_data: historical solar data for the same dates as the trial
            self.max_iter: maximum number of iterations for generating hourly states
            self.cloud_hours: Tuple containing maximum and minimum hours of the day to be used
                to set the day's cloud state
            self.num_iters: List containing the number of iterations required to generate
                hourly states for each day and each trial

        outputs:
            Dataframe with hourly states for the given trial

        """

        # Initialize list to hold hourly states
        all_hourly_states = []
        current_state = [None, None, None, None]

        # Loop through days in the trial
        for date_time, daily_state in daily_states.iterrows():

            # Set up parameters for while loop
            num_iter = 0
            best_num_matches = 0
            best_states = None

            # Filter date_range to only include this day
            one_day_date_range = date_range[date_range.date == date_time]

            # Get clearsky data from date range and get night states, which indicate if it is
            #   day or night time - this is used to filter which states are considered.
            one_day_clear_sky_data = clear_sky_data[
                clear_sky_data.index.date == date_time]
            night_states = one_day_clear_sky_data['clear_sky_ghi'] == 0

            # Keep generating hourly states until a set is found which matches the daily
            #   states or the maximum number of iterations is reached
            while num_iter < self.max_iter:
                # Generate hourly states
                hourly_states = self.generate_random_state_hourly(
                    one_day_date_range, night_states, current_state)

                # Check if this day contains all daylight hours, via the cloud hours param -
                #   the daily/hourly check should not be performed for the beginning or end of
                #   a trial with only a partial day of generated hourly states
                if not len(set(range(self.cloud_hours[0], self.cloud_hours[1])) -
                           set(one_day_date_range.hour)):

                    # Aggregate hourly states to daily
                    agg_ghi_state, agg_dni_state, cloud_modes = \
                        self.aggregate_hourly_states(one_day_clear_sky_data, hourly_states)

                    # Check if states match
                    num_matches = np.sum([
                        agg_ghi_state == daily_state['ghi_state'],
                        agg_dni_state == daily_state['dni_state'],
                        daily_state['cloud_state'] in cloud_modes])

                    # If all states match, this is an acceptable state, if not, save the state
                    #   if it exceeds the previous number of best matches
                    if num_matches == 3:
                        best_states = hourly_states
                        break

                    elif num_matches > best_num_matches:
                        best_num_matches = num_matches
                        best_states = hourly_states
                else:
                    best_states = hourly_states
                    break
                num_iter += 1

            # If the maximum number of iterations has been reached and no acceptable state has
            #   been found, just use the current state
            if best_states is None:
                best_states = hourly_states

            # Record the number of iterations required to find a match
            # Note: this only works without multi-threading
            self.num_iters += [num_iter]

            # Add to hourly states list
            all_hourly_states += list(best_states.values)
            current_state = best_states.iloc[-1].values

        if queue is None:
            # Without multithreading, return hourly states dataframe
            return pd.DataFrame(
                data=all_hourly_states, index=date_range,
                columns=['ghi_state', 'dni_state', 'cloud_state',
                         'temp_state']), clear_sky_data
        else:
            # With multithreading, add dataframe to Queue
            queue.put([pd.DataFrame(
                data=all_hourly_states, index=date_range,
                columns=['ghi_state', 'dni_state', 'cloud_state',
                         'temp_state']), clear_sky_data])

    def generate_random_state_hourly(self, hourly_date_range, night_states,
                                     current_state):
        """
        Generate random hourly states for a solar trial.

        inputs:
            hourly_date_range: date_range object with the hours for the current day to be
                generated
            night_states: list containing a Boolean indicating if it is night-time for each
                hour in the day
            current_state: array with hourly state from previous hour
            self.length_trials: length of profiles in hours
            self.nrel_data_df: historical solar data
            self.state_prob_hourly_grouped: dataframe holding the probabilities of transition
                to a given daily state, given the from-state, grouped by month, hour,
                night_state and from-states.
            self.simple_prob_hourly_grouped: dataframe holding the probabilities of
                transitioning to a given daily state, independent of the from-state, grouped
                by month, hour, and night_state.

        outputs:
            dataframe with hourly trial states

        """

        # Create list to hold generated states
        state_list = []

        # Create hourly states iteratively
        for date_time, night_state in zip(hourly_date_range, night_states):
            try:
                # Get the numpy values from the subset of states with the corresponding month,
                #   hour, and from-states. While the state data is stored in a pandas
                #   dataframe, the following operations are carried out on numpy arrays to
                #   speed up the calculation.
                # Grouped hourly states have labels of the form:
                #   (month, hour, night_state, ghi_state_from, dni_state_from,
                #   cloud_type_from, temp_state_from)
                states_subset = self.state_prob_hourly_grouped[
                    tuple([date_time.month, date_time.hour, night_state] +
                          list(current_state))].values
            # Except statement catches cases where state subset is empty (i.e. if the
            #   from-state does not exist in the corresponding month) or the first day in the
            #   trial, in which case it uses the simple probability dataframe which doesn't
            #   require a from-state.
            except KeyError:
                # Same as above, but for simple probabilities
                # Grouped hourly simple states have labels of the form:
                #   (month, hour, night_state)
                states_subset = self.simple_prob_hourly_grouped[
                    (date_time.month, date_time.hour, night_state)].values

            # Get the probabilities for this subset of states
            probs = states_subset[:, -2].astype(float)

            # Randomly select a state based on the probabilities
            sample_index = np.random.choice(range(len(states_subset)),
                                            p=probs / np.sum(probs))

            # Assign the current state and add to list
            current_state = states_subset[sample_index, 2:6]
            state_list += [current_state]

        # return state_list
        return pd.DataFrame(data=state_list, index=hourly_date_range,
                            columns=['ghi_state', 'dni_state', 'cloud_state', 'temp_state'])
             
    def aggregate_hourly_states(self, one_day_clear_sky_data, hourly_states):
        """
        Aggregate hourly states to a daily state.

        inputs
            one_day_clear_sky_data: historical solar data for a single day
            hourly_states: DataFrame with hourly states for a single day
            self.num_hourly_ghi_states: Number of discrete GHI states for hourly model
            self.num_hourly_dni_states: Number of discrete DNI states for hourly model
            self.num_daily_ghi_states: Number of discrete GHI states for daily model
            self.num_daily_dni_states: Number of discrete DNI states for daily model
            self.cloud_hours: Tuple containing maximum and minimum hours of the day to be used
                to set the day's cloud state

        outputs
            agg_ghi_state: Hourly GHI states aggregated to a daily value
            agg_dni_state: Hourly DNI states aggregated to a daily value
            cloud_modes: Modes of hourly cloud types

        """

        # Calculate the percent of available clearsky radiation that was observed for this day
        ghi_percent = (one_day_clear_sky_data['clear_sky_ghi'] *
                       (1 - hourly_states['ghi_state'] /
                       (self.num_hourly_ghi_states - 1))).sum() / \
            one_day_clear_sky_data['clear_sky_ghi'].sum()
        dni_percent = (one_day_clear_sky_data['clear_sky_dni'] *
                       (1 - hourly_states['dni_state'] /
                       (self.num_hourly_dni_states - 1))).sum() / \
            one_day_clear_sky_data['clear_sky_dni'].sum()

        # Calculate aggregated daily state based on hourly states
        agg_ghi_state = round(ghi_percent * (self.num_daily_ghi_states - 1)) + 1
        agg_dni_state = round(dni_percent * (self.num_daily_dni_states - 1)) + 1

        # Calculate the modes of the hourly cloud states
        cloud_value_counts = hourly_states.loc[hourly_states.index.hour.isin(
            list(range(self.cloud_hours[0], self.cloud_hours[1]))),
            'cloud_state'].value_counts()
        cloud_modes = list(cloud_value_counts[
                               cloud_value_counts == cloud_value_counts.iloc[0]].index)

        return agg_ghi_state, agg_dni_state, cloud_modes


def date_parser(month, day, year, hour):
    """ Function to parse datetimes for historical solar data. """
    return pd.to_datetime('{}/{}/{} {}:00'.format(int(month), int(day), int(year), int(hour)))


if __name__ == "__main__":
    # Used for testing
    t = dt.datetime.utcnow()
    asp = AlternativeSolarProfiles(latitude=46.34, longitude=-119.28,
                                   num_trials=10, length_trials=14,
                                   max_iter=200, multithreading=False)
    asp.create_state_transition_matrices()
    print('time for setup: {}'.format(dt.datetime.utcnow()-t))
    t = dt.datetime.utcnow()
    asp.create_trial_data()
    print('time for trial creation: {}'.format(dt.datetime.utcnow() - t))
