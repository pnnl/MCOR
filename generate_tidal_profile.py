# -*- coding: utf-8 -*-
"""

File contents:
    Classes:
        TidalProfileGenerator

    Standalone functions:
        get_tidal_data_from_upload
        calc_tidal_prod
"""

import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from utide import solve, reconstruct
from validation import validate_all_parameters, log_error, strings_warnings
from config import TIDAL_DATA_DIR, ROOT_DIR

TIDAL_DEFAULTS = {'rated_power': 50,
                  'rotor_length': 5,
                  'turbine_number': 1,
                  'maximum_cp': 0.42,
                  'cut_in_velocity': 0.7,
                  'cut_out_velocity': 2.6,
                  'inverter_efficiency': 0.9,
                  'turbine_losses': 10}

class TidalProfileGenerator:
    """   
    Class to upload tidal_current tidal data, extract tidal constituents,
    extrapolate to tidal epoch, create tidal profiles, and calculate power profiles.
    
    Parameters
    ----------

        longitude: Site longitude in degrees    
    
        latitude: Site latitude in degrees
    
        timezone: US timezone, options:
            US/Alaska, US/Aleutian, US/Arizona, US/Central, US/East-Indiana, US/Eastern,
            US/Hawaii, US/Indiana-Starke, US/Michigan, US/Mountain, US/Pacific,
            US/Pacific-New, US/Samoa

        num_trials: Number of tidal profiles to create
    
        length_trials: Length of tidal profiles in hours

        advanced_inputs: Dictionary specifying advanced tidal system inputs.
            These could include:
                rated power, rotor length, turbine number, maximum cp,
                cut in velocity, cut out velocity, inverter efficiency, turbine_losses
                
    Methods
    ----------

        get_tidal_data_from_upload: Uploads one year of tidal data

        extract_tidal_constituents: Extracts tidal constituents from tidal current data

        extrapolate_tidal_epoch: Creates tidal epoch of current data from tidal constituents

        generate_tidal_profiles: Generates tidal profiles from tidal epoch

        calc_tidal_prod: Calculates tidal production for each profile

        tidal_checks: Creates several plots to verify that the tidal power calculation went OK

        get_dc_to_ac: Returns the DC to AC ratio

        get_losses: Returns system power losses
        
    Calculated Attributes
    ----------

        tidal_profiles: list of Pandas dataframes with tidal profiles
        
        power_profiles: list of Pandas series' with tidal power profiles for a 1kW system

    """

    def __init__(self, latitude, longitude, timezone, num_trials,
                 length_trials, validate=True,advanced_inputs={}):

        # Assign parameters
        self.latitude = latitude
        self.longitude = longitude
        self.timezone = timezone
        self.num_trials = num_trials
        self.length_trials = length_trials
        self.advanced_inputs = advanced_inputs
        self.tidal_profiles = []
        self.power_profiles = []

        # Add TIDAL_DEFAULTS to advanced inputs if not already included
        for key in TIDAL_DEFAULTS:
            if key not in self.advanced_inputs:
                self.advanced_inputs[key] = TIDAL_DEFAULTS[key]

        # TODO: add new params here and add validation checks
        if validate:
            # List of initialized parameters to validate
            args_dict = {'latitude': self.latitude,
                         'longitude': self.longitude,
                         'timezone': self.timezone,
                         'num_trials': self.num_trials,
                         'length_trials': self.length_trials,
                         'tpg_advanced_inputs': self.advanced_inputs}

            # Validate input parameters
            validate_all_parameters(args_dict)

    def get_tidal_data_from_upload(self):
        """Load tidal_current-specified tidal data"""

        # TODO: edit example csv to include u and v columns
        # filedir = os.path.join(TIDAL_DATA_DIR, 'tidal_current')
        # file = os.listdir(filedir)
        # # read first (and only) file in directory
        # self.tidal_current = pd.read_csv(os.path.join(filedir, file[0]), header=0) # assuming one file

        # TODO: remove once actual data is available. This code generates fake tide with u and v components
        def fake_tide(t, M2amp, M2phase):

            return M2amp * np.sin(2 * np.pi * t / 12.42 - M2phase)

        t = pd.date_range(
            start='1/1/2017', end='1/1/2018', freq='H')[:-1]
        # Signal + some noise.
        u = fake_tide(np.arange(8760), M2amp=2, M2phase=0) + np.random.randn(8760)
        v = fake_tide(np.arange(8760), M2amp=1, M2phase=np.pi) + np.random.randn(8760)
        self.tidal_current = pd.DataFrame()
        self.tidal_current['u'] = u
        self.tidal_current['v'] = v

    def extrapolate_tidal_epoch(self):
        """Extract tidal constituents from 8760 of tidal current data and extrapolate 19-year tidal epoch"""

        # TODO: modify this in the future to extrapolate any amount of data. Also remove adding index, since user data should have a datetime index
        self.tidal_current.index = pd.date_range(
            start='1/1/2017', end='1/1/2018', freq='H')[:-1]
        coef = solve(t = self.tidal_current.index,u = self.tidal_current['u'],v = self.tidal_current['v'] , lat=self.latitude, method="ols", conf_int="linear",verbose=False)

        # TODO: match the range of solar data availability from NSRDB
        epoch_index = pd.date_range(
            start='1/1/2017', end='1/1/2036', freq='H')[:-1]
        tide = reconstruct(epoch_index, coef, verbose=False)
        self.tidal_epoch = pd.DataFrame()
        self.tidal_epoch['v'] = (tide.u**2 + tide.v**2)**(0.5)
        self.tidal_epoch.index = epoch_index

    def generate_tidal_profiles(self):
        """Generate tidal profiles from tidal epoch data"""

        # Randomly create start dates
        start_datetimes = self.tidal_epoch.iloc[:-self.length_trials].sample(
            int(self.num_trials)).index.values

        date_ranges = [pd.date_range(start=start_date,
                                     periods=self.length_trials,
                                     freq='H')
                       for start_date in start_datetimes]

        for i, date_range in enumerate(date_ranges):
            replace_timesteps = len(date_range[(date_range.month == 2) &
                                               (date_range.day == 29)])
            if replace_timesteps:
                # Remove 2/29
                date_range = date_range[date_range.date != datetime.date(date_range[0].year, 2, 29)]

                # Add more timesteps
                date_ranges[i] = date_range.append(pd.date_range(
                    date_range[-1] + datetime.timedelta(hours=1),
                    periods=replace_timesteps, freq='H'))

            # Create 20-year annual profile to allow for profiles with year-end overlap
            twentyyear_profile = pd.concat([self.tidal_epoch, self.tidal_epoch.head(8760)])
            twentyyear_profile.index = pd.date_range(
                start='1/1/2017', end='1/1/2037', freq='H')[:-25]

        # Loop over each date range and sample profile data
        for date_range in date_ranges:
            self.tidal_profiles += [twentyyear_profile.loc[date_range]]

        # Create directory to hold data
        if '{}_{}_{}d_{}t'.format(
                self.latitude, self.longitude, int(self.length_trials / 24),
                int(self.num_trials)) not in \
                os.listdir(os.path.join(TIDAL_DATA_DIR, 'tidal_profiles')):
            os.mkdir(os.path.join(
                TIDAL_DATA_DIR, 'tidal_profiles', '{}_{}_{}d_{}t'.format(
                    self.latitude, self.longitude, int(self.length_trials / 24),
                    int(self.num_trials))))

        for i, tidal_profile in enumerate(self.tidal_profiles):
            tidal_profile.to_csv(os.path.join(
            TIDAL_DATA_DIR, 'tidal_profiles', '{}_{}_{}d_{}t'.format(
                self.latitude, self.longitude, int(self.length_trials / 24),
                int(self.num_trials)),
            '{}_{}_tidal_trial_{}.csv'.format(self.latitude,
                                              self.longitude, i)))

    def add_storm_factors(self):
        """Add storm factors to tidal profiles, correlate with solar profiles"""

        # TODO: implement

    def get_power_profiles(self):
        """ 
        Calculate the output AC power for a 1kW system for each tidal profile.
       
        If read_from_file is True, reads the tidal data from csv,allowing
            for faster lookup rather than re-running get_tidal_data and get_tidal_profiles.
            
        """

        # For each tidal profile, calculate production
        # Load the tidal data from csv
        for i in range(int(self.num_trials)):
            try:
                tidal = pd.read_csv(os.path.join(
                    TIDAL_DATA_DIR, 'tidal_profiles', '{}_{}_{}d_{}t'.format(
                        self.latitude, self.longitude, int(self.length_trials/24),
                        int(self.num_trials)),
                    '{}_{}_tidal_trial_{}.csv'.format(self.latitude, self.longitude, i)),
                    index_col=0, parse_dates=[0])

            except FileNotFoundError:
                message = 'Tidal profile csvs not found. Please check that you have entered' \
                          ' the longitude, latitude, number, and length of trials for a ' \
                          'site with previously generated tidal profiles.'
                log_error(message)
                raise Exception(message)

            # Localize timezone
            tidal.index = tidal.index.tz_localize(self.timezone)

            self.tidal_profiles += [tidal]

        # Calculate production for each tidal profile
        for tidal in self.tidal_profiles:
            self.power_profiles += [calc_tidal_prod(
                tidal,
                self.latitude,
                self.longitude,
                self.advanced_inputs['rated_power'],
                self.advanced_inputs['rotor_length'],
                self.advanced_inputs['turbine_number'],
                self.advanced_inputs['inverter_efficiency'],
                self.advanced_inputs['maximum_cp'],
                self.advanced_inputs['turbine_losses'],
                self.advanced_inputs['cut_in_velocity'],
                self.advanced_inputs['cut_out_velocity'])]

        # Validate parameters
        if validate:
            args_dict = {'num_seconds': num_seconds}
            validate_all_parameters(args_dict)

        # For each profile in tidal_profiles, power_profiles, temp_profiles and
        #   night_profiles, crop to the specified number of seconds, rounding down to the
        #   nearest timestep
        for i in range(len(self.tidal_profiles)):
            self.tidal_profiles[i] = \
                self.tidal_profiles[i].loc[
                :self.tidal_profiles[i].index[0] + datetime.timedelta(
                    seconds=num_seconds - 1)]

        for i in range(len(self.power_profiles)):
            self.power_profiles[i] = \
                self.power_profiles[i].loc[
                :self.power_profiles[i].index[0] + datetime.timedelta(
                    seconds=num_seconds - 1)]

    def tidal_checks(self):
        """ Several checks to  make sure the tidal profiles look OK. """

        # Get the profiles with the min and max energy
        total_energy = [prof.sum() for prof in self.power_profiles]
        max_profile_num = np.where(total_energy == max(total_energy))[0][0]
        min_profile_num = np.where(total_energy == min(total_energy))[0][0]

        # Plot the profiles with min and max energy
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        self.power_profiles[max_profile_num].plot(
            ax=ax1, title='Profile with max energy generation')
        ax1.set_ylabel('Power (kW)')
        ax2 = fig.add_subplot(122)
        self.power_profiles[min_profile_num].plot(
            ax=ax2, title='Profile with min energy generation')
        ax2.set_ylabel('Power (kW)')

def calc_tidal_prod(tidal_profile, latitude, longitude,
                    rated_power, rotor_length, turbine_number,
                    inverter_efficiency, maximum_cp, turbine_losses,
                    cut_in_velocity, cut_out_velocity, validate=False):
    """ Calculates the production from a tidal profile. """

    if validate:
        # Put arguments in a dict
        args_dict = {'tidal_profile': tidal_profile,
                     'latitude': latitude,
                     'longitude': longitude,
                     'rated_power': rated_power,
                     'rotor_length': rotor_length,
                     'turbine_number': turbine_number,
                     'inverter_efficiency': inverter_efficiency,
                     'maximum_cp': maximum_cp,
                     'turbine_losses': turbine_losses,
                     'cut_in_velocity': 0.7,
                     'cut_out_velocity': 2.6}

        # Validate all parameters
        validate_all_parameters(args_dict)

    # Calculate DC power (normalized to turbine size. i.e. per 1kW of tidal)
    dc_power = pd.DataFrame()
    for index, row in tidal_profile.iterrows():
        u = row['v']
        if u >= cut_in_velocity and u <= cut_out_velocity:
            dc_power.at[
                index, 'power'] = 0.5 * maximum_cp * u ** 3 * np.pi * rotor_length ** 2 * turbine_number / (rated_power * turbine_number)
        elif u < cut_in_velocity:
            dc_power.at[index, 'power'] = 0
        elif u > cut_out_velocity:
            dc_power.at[index, 'power'] = rated_power * turbine_number / (rated_power * turbine_number)
        else:
            dc_power.at[index, 'power'] = np.nan

    # Calculate losses
    dc_power['power'] = dc_power['power'] * (1 - turbine_losses / 100)

    # Calculate AC power
    ac_power = dc_power['power'] * inverter_efficiency

    # Force values less than 0 to 0
    ac_power[ac_power < 0] = 0

    return ac_power

if __name__ == "__main__":
    # Used for testing
    # Create a TidalProfileGenerator object
    latitude = 46.34
    longitude = -119.28
    timezone = 'US/Pacific'
    tpg = TidalProfileGenerator(latitude, longitude, timezone, num_trials= 5, length_trials= 14, validate=False)

    tpg.get_tidal_data_from_upload()
    print('uploaded data')

    tpg.extrapolate_tidal_epoch()
    print('extrapolated tidal epoch')

    tpg.generate_tidal_profiles()
    print('generated tidal profiles')

    tpg.get_power_profiles()
    print('calculated power')

    tpg.tidal_checks()
