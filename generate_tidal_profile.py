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

TIDAL_DEFAULTS = {'tidal_turbine_rated_power': 550,
                  'depth': 10,
                  'tidal_rotor_radius': 10,
                  'tidal_rotor_number': 2,
                  'maximum_cp': 0.42,
                  'tidal_cut_in_velocity': 0.5,
                  'tidal_cut_out_velocity': 3,
                  'tidal_inverter_efficiency': 0.9,
                  'tidal_turbine_losses': 10}

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
                rated power, rotor length, tidal_rotor_number, turbine number, maximum cp,
                cut in velocity, cut out velocity, inverter efficiency, tidal_turbine_losses
                
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

    def __init__(self, marine_data_filename, latitude, longitude, timezone, num_trials, length_trials, tidal_turbine_rated_power, depth,
                 tidal_rotor_radius, tidal_rotor_number, maximum_cp,tidal_cut_in_velocity, tidal_cut_out_velocity, tidal_inverter_efficiency,
                 tidal_turbine_losses, start_year=None, end_year=None, validate=True):

        # Assign parameters
        self.marine_data_filename = marine_data_filename
        self.latitude = latitude
        self.longitude = longitude
        self.timezone = timezone
        self.num_trials = num_trials
        self.length_trials = length_trials
        self.tidal_turbine_rated_power = tidal_turbine_rated_power
        self.depth = depth
        self.tidal_rotor_radius = tidal_rotor_radius
        self.tidal_rotor_number = tidal_rotor_number
        self.maximum_cp = maximum_cp
        self.tidal_cut_in_velocity = tidal_cut_in_velocity
        self.tidal_cut_out_velocity = tidal_cut_out_velocity
        self.tidal_inverter_efficiency = tidal_inverter_efficiency
        self.tidal_turbine_losses = tidal_turbine_losses
        self.start_year = start_year
        self.end_year = end_year
        self.tidal_profiles = []
        self.power_profiles = []
        self.tmy_tidal = None

        # # Add TIDAL_DEFAULTS to advanced inputs if not already included
        # for key in TIDAL_DEFAULTS:
        #     if key not in self.advanced_inputs:
        #         self.advanced_inputs[key] = TIDAL_DEFAULTS[key]

        if validate:
            # List of initialized parameters to validate
            args_dict = {'latitude': self.latitude,
                         'longitude': self.longitude,
                         'timezone': self.timezone,
                         'num_trials': self.num_trials,
                         'length_trials': self.length_trials}
            if start_year is not None:
                args_dict['start_year'] = start_year
            if end_year is not None:
                args_dict['end_year'] = end_year

            # Validate input parameters
            validate_all_parameters(args_dict)

        # Set default start and end years is not specified
        if self.start_year is None:
            self.start_year = 2017
        if self.end_year is None:
            self.end_year = self.start_year + 19

    def get_tidal_data_from_upload(self):
        """Load tidal_current-specified tidal data"""

        file = os.path.join(TIDAL_DATA_DIR, 'tidal_current', self.marine_data_filename)
        self.tidal_current = pd.read_csv(file, header=0)

        # Find the closest depth
        def find_closest_depth(df, depth):
            depth_columns = [col for col in df.columns if col.startswith('u_') or col.startswith('v_')]
            depth_values = sorted(set(float(col.split('_')[1]) for col in depth_columns))

            if depth < min(depth_values):
                raise ValueError('Specified depth is smaller than any available depth')
            elif depth > max(depth_values):
                raise ValueError('Specified depth is larger than any available depth')

            closest_depth = min(depth_values, key=lambda x: abs(x - depth))
            return closest_depth

        # Extract depth values
        def extract_columns(df,depth):
            closest_depth = find_closest_depth(df, depth)
            depth_str = str(closest_depth)
            u_col = f'u_{depth_str}'
            v_col = f'v_{depth_str}'

            if u_col in df.columns and v_col in df.columns:
                extracted_df = df[['time', u_col, v_col]]
                extracted_df.columns = ['time', 'u', 'v']
                return extracted_df

        self.tidal_current = extract_columns(self.tidal_current, self.depth)
        self.tidal_current.set_index('time', inplace=True)
        self.tidal_current.index = pd.to_datetime(self.tidal_current.index)
        self.tidal_current = self.tidal_current.resample('H').mean()

    def extrapolate_tidal_epoch(self, validate=True):
        """Extract tidal constituents from 8760 of tidal current data and extrapolate 19-year tidal epoch"""

        # TODO: modify this in the future to extrapolate any amount of data
        coef = solve(t = self.tidal_current.index,u = self.tidal_current['u'],v = self.tidal_current['v'] , lat=self.latitude, method="ols", conf_int="linear",verbose=False)

        epoch_index = pd.date_range(
            start=f'1/1/{self.start_year}', end=f'1/1/{self.end_year+1}', freq='H', tz=self.timezone)[:-1]
        tide = reconstruct(epoch_index, coef, verbose=False)
        self.tidal_epoch = pd.DataFrame()
        self.tidal_epoch['v_mag'] = (tide.u**2 + tide.v**2)**(0.5)
        self.tidal_epoch.index = epoch_index

    def generate_tidal_profiles(self, start_datetimes=None, validate=True):
        """Generate tidal profiles from tidal epoch data"""

        # Validate input arguments
        if validate and start_datetimes is not None:
            # List of initialized parameters to validate
            args_dict = {'num_trials': self.num_trials,
                         'start_year': self.start_year,
                         'end_year': self.end_year,
                         'start_datetimes': start_datetimes}
            
            # Validate input parameters
            validate_all_parameters(args_dict)

        # Randomly create start dates
        if start_datetimes is None:
            start_datetimes = self.tidal_epoch.iloc[:-int(self.length_trials)].sample(
                int(self.num_trials)).index.values

        date_ranges = [pd.date_range(start=start_date,
                                     periods=self.length_trials,
                                     freq='H', tz=self.timezone)
                       for start_date in start_datetimes]
        
        # Add an extra year to the annual profile to allow for profiles with year-end overlap
        twentyyear_profile = pd.concat([self.tidal_epoch, self.tidal_epoch.head(8760)])
                
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
        # Load the tidal data from csv if not already in the self.tidal_profiles list
        if not len(self.tidal_profiles):
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

                self.tidal_profiles += [tidal]

        # Calculate production for each tidal profile
        for tidal in self.tidal_profiles:
            self.power_profiles += [calc_tidal_prod(
                tidal,
                self.timezone,
                self.tidal_turbine_rated_power,
                self.tidal_rotor_radius,
                self.tidal_rotor_number,
                self.tidal_inverter_efficiency,
                self.maximum_cp,
                self.tidal_turbine_losses,
                self.tidal_cut_in_velocity,
                self.tidal_cut_out_velocity)]
            
        # Calculate power production for initial 1-year profile
        self.tidal_current['v_mag'] = self.tidal_current.apply(
            lambda x: (x['u']**2 + x['v']**2)**(0.5), axis=1)
        self.tmy_tidal = calc_tidal_prod(self.tidal_current, 
                self.timezone,
                self.tidal_turbine_rated_power,
                self.tidal_rotor_radius,
                self.tidal_rotor_number,
                self.tidal_inverter_efficiency,
                self.maximum_cp,
                self.tidal_turbine_losses,
                self.tidal_cut_in_velocity,
                self.tidal_cut_out_velocity)

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

def calc_tidal_prod(tidal_profile, timezone,
                    tidal_turbine_rated_power, tidal_rotor_radius, tidal_rotor_number,
                    tidal_inverter_efficiency, maximum_cp, tidal_turbine_losses,
                    tidal_cut_in_velocity, tidal_cut_out_velocity, validate=False):
    """ Calculates the production from a tidal profile. """

    if validate:
        # Put arguments in a dict
        args_dict = {'tidal_profile': tidal_profile,
                     'tidal_turbine_rated_power': tidal_turbine_rated_power,
                     'tidal_rotor_radius': tidal_rotor_radius,
                     'tidal_rotor_number': tidal_rotor_number,
                     'tidal_inverter_efficiency': tidal_inverter_efficiency,
                     'maximum_cp': maximum_cp,
                     'tidal_turbine_losses': tidal_turbine_losses,
                     'tidal_cut_in_velocity': tidal_cut_in_velocity,
                     'tidal_cut_out_velocity': tidal_cut_out_velocity}

        # Validate all parameters
        validate_all_parameters(args_dict)

    # Calculate DC power
    dc_power = pd.DataFrame()
    for index, row in tidal_profile.iterrows():
        u = row['v_mag']
        if u >= tidal_cut_in_velocity and u <= tidal_cut_out_velocity:
            dc_power.at[
                index, 'power'] = np.min([0.5 * maximum_cp * 1.03 * u ** 3 * np.pi
                                           * tidal_rotor_radius ** 2 * tidal_rotor_number, 
                                          tidal_turbine_rated_power])
        elif u < tidal_cut_in_velocity:
            dc_power.at[index, 'power'] = 0
        elif u > tidal_cut_out_velocity:
            dc_power.at[index, 'power'] = 0 
        else:
            dc_power.at[index, 'power'] = np.nan

    # Fix timezone
    try:
        dc_power.index = dc_power.index.tz_convert(timezone)
    except TypeError:
        dc_power.index = pd.to_datetime(dc_power.index, utc=True).tz_convert(timezone)

    # Normalize DC power generation to turbine size. i.e. per 1kW of tidal
    dc_power['power'] = dc_power['power'] / (tidal_turbine_rated_power)

    # Calculate turbine losses
    dc_power['power'] = dc_power['power'] * (1 - tidal_turbine_losses / 100)

    # Calculate AC power
    ac_power = dc_power['power'] * tidal_inverter_efficiency

    # Force values less than 0 to 0
    ac_power[ac_power < 0] = 0

    return ac_power

if __name__ == "__main__":
    # Used for testing
    # Create a TidalProfileGenerator object
    marine_data_filename = 'PortAngeles_2015_alldepths.csv'
    latitude = 46.34
    longitude = -119.28
    timezone = 'US/Pacific'
    tpg = TidalProfileGenerator(marine_data_filename, latitude, longitude, timezone, tidal_turbine_rated_power = 550,
                  depth = 10, tidal_rotor_radius = 10, tidal_rotor_number = 2, maximum_cp = 0.42, tidal_cut_in_velocity = 0.5,
                  tidal_cut_out_velocity = 3, tidal_inverter_efficiency = 0.9, tidal_turbine_losses = 10, num_trials= 5,
                  length_trials= 14, validate=True)

    tpg.get_tidal_data_from_upload()
    print('uploaded data')

    tpg.extrapolate_tidal_epoch()
    print('extrapolated tidal epoch')

    tpg.generate_tidal_profiles()
    print('generated tidal profiles')

    tpg.get_power_profiles()
    print('calculated power')

    tpg.tidal_checks()
