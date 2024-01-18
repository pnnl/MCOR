# -*- coding: utf-8 -*-
"""
Runs the Alternative Solar Profiles (ASP) algorithm used for solar forecasting and uses
pvlib-python to calculate AC power for each profile.
 

    
File contents:
    Classes:
        TidalProfileGenerator

    Standalone functions:
        get_tidal_data_from_upload
        calc_pv_prod
"""

import os
import pandas as pd
import numpy as np
import math
import datetime
import pytz
import matplotlib.pyplot as plt
import io
import requests
import json
import warnings
import yaml

from pvlib.temperature import sapm_cell, TEMPERATURE_MODEL_PARAMETERS
from pvlib import solarposition, irradiance, atmosphere, pvsystem, tracking
from validation import validate_all_parameters, log_error, strings_warnings
from alternative_solar_profiles import AlternativeSolarProfiles
from config import TIDAL_DATA_DIR, ROOT_DIR

TIDAL_DEFAULTS = {'turbine_output': 100,
                  'rotor_diameter': 1,
                  'rotor_number' : 1,
                  'PTO_efficiency' : .95,
                  'maximum_cp' : 0.42,
                  'cut_in_velocity' : 0,
                  'cut_out_velocity' : 3}


class TidalProfileGenerator:
    """   
    Class to upload user tidal data, create tidal profiles, and
        calculate power profiles.
    
    Parameters
    ----------
    
        longitude: Site longitude in degrees    
    
        latitude: Site latitude in degrees
    
        timezone: US timezone, options:
            US/Alaska, US/Aleutian, US/Arizona, US/Central, US/East-Indiana, US/Eastern,
            US/Hawaii, US/Indiana-Starke, US/Michigan, US/Mountain, US/Pacific,
            US/Pacific-New, US/Samoa
        
        depth: depth bin in meters

        num_trials: Number of solar profiles to create
    
        length_trials: Length of solar profiles in hours


        max_iter: The maximum number of iterations allowed for trying to match hourly and
            daily states in the ASP code.

        multithreading: Whether to use multithreading to speed up the ASP calculation. This is
            set to True by default, but should be set to False for debugging

        advanced_inputs: Dictionary specifying advanced pv system inputs.
            These could include:
                albedo, racking, module, inverter, strings, soiling, shading, snow, mismatch,
                wiring, connections, lid, nameplate_rating, age, availability

        suppress_warnings: Boolean specifying whether or not warnings should be printed to the
            console with relevant plots.
                
    Methods
    ----------


        get_tidal_data_from_upload: Uploads one year of tidal data

        extract_tidal_constituents: Extracts tidal constituents from tidal current data

        extrapolate_tidal_epoch: Creates tidal epoch of current data from tidal constituents

        generate_tidal_profiles: Generates tidal profiles from tidal epoch

        calc_tidal_prod: Calculates tidal production for each profile
            
        crop_timeline: Used to crop the profiles to a period of less than 1 day

        tidal_checks: Creates several plots to verify that the tidal power calculation went OK

        get_dc_to_ac: Returns the DC to AC ratio

        get_losses: Returns system power losses
        
    Calculated Attributes
    ----------

        tidal_profiles: list of Pandas dataframes with tidal profiles
        
        power_profiles: list of Pandas series' with PV power profiles for a 1kW system

        constraints: Pandas DataFrame holding constraints for input parameters

    """

    def __init__(self, latitude, longitude, timezone, depth, num_trials,
                 length_trials, max_iter=200, multithreading=True,
                 advanced_inputs={}, validate=True, suppress_warnings=False):

        # Assign parameters
        self.latitude = latitude
        self.longitude = longitude
        self.timezone = timezone
        self.depth = depth
        self.num_trials = num_trials
        self.length_trials = length_trials
        self.max_iter = max_iter
        self.multithreading = multithreading
        self.advanced_inputs = advanced_inputs
        self.suppress_warnings = suppress_warnings
        self.tidal_profiles = []
        self.power_profiles = []


        # Add TIDAL_DEFAULTS to advanced inputs if not already included
        for key in TIDAL_DEFAULTS:
            if key not in self.advanced_inputs:
                self.advanced_inputs[key] = TIDAL_DEFAULTS[key]

        if validate:
            # List of initialized parameters to validate
            args_dict = {'latitude': self.latitude,
                         'longitude': self.longitude,
                         'timezone': self.timezone,
                         'depth': self.depth,
                         'num_trials': self.num_trials,
                         'length_trials': self.length_trials,
                         'max_iter': self.max_iter,
                         'multithreading': self.multithreading,
                         'tpg_advanced_inputs': self.advanced_inputs}

            # Validate input parameters
            validate_all_parameters(args_dict)


    def get_tidal_data_from_upload(self):
        """Load user-specified tidal data"""
        filedir = os.path.join(TIDAL_DATA_DIR, 'user')
        files = os.listdir(filedir)
        for file in files:
            # skip files that aren't csv files
            if not file.split('.')[-1] == '.csv':
                continue
            tidal_current = pd.read_csv(os.path.join(filedir, file)) # just one depth??

    def extract_tidal_constituents(self, tidal_current):
        """Extract tidal constituents from 8760 of tidal current data"""

        tidal_constituents = tidal_current
        return tidal_constituents

    def extrapolate_tidal_epoch(self, tidal_constituents):
        """Extrapolate 19-year tidal epoch from tidal constituents"""
        tidal_epoch = tidal_constituents
        return tidal_epoch

    def generate_tidal_profiles(self, tidal_epoch):
        """Generate tidal profiles from tidal epoch data"""

        # Randomly create start dates
        tidal_epoch.index = pd.date_range(
            start='1/1/2017', end='1/1/2035', freq='H')[:-1]
        start_datetimes = tidal_epoch.iloc[:-self.length_trials].sample(
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
            twentyyear_profile = tidal_epoch.append(tidal_epoch.head(8760))
            twentyyear_profile.index = pd.date_range(
                start='1/1/2017', end='1/1/2036', freq='H')[:-1]


        # Loop over each date range and sample profile data
        for date_range in date_ranges:
            self.power_profiles += [twentyyear_profile.loc[date_range]]

        for i, tidal_profile in self.power_profiles:
            tidal_profile.to_csv(os.path.join(
            TIDAL_DATA_DIR, 'tidal_profiles', '{}_{}_{}d_{}t'.format(
                self.latitude, self.longitude, int(self.length_trials / 24),
                int(self.num_trials)),
            '{}_{}_tidal_trial_{}.csv'.format(self.latitude,
                                              self.longitude, i)))



    def get_power_profiles(self):
        """ 
        Calculate the output AC power for a 1kW system for each tidal profile. Or do we do it for the full size of the specified turbine??
       
        If read_from_file is True, reads the solar and temperature data  from csv, allowing
            for faster lookup rather than re-running get_solar_data and get_solar_profiles.
            
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

            # Fix timezone
            try:
                tidal.index = tidal.index.tz_convert(self.timezone)
            except AttributeError:
                # Deal with pandas issue creating datetime index from timeseries including
                #   daylight savings time shift
                tidal.index = pd.to_datetime(tidal.index, utc=True).tz_convert(self.timezone)



            self.tidal_profiles += [tidal]



        # Calculate PV production for each tidal profile
        for tidal in self.tidal_profiles:
            self.power_profiles += [calc_pv_prod(
                tidal, temp, self.latitude, self.longitude, self.depth,
                suppress_warnings=self.suppress_warnings,
                advanced_inputs=self.advanced_inputs)]


    def crop_timeline(self, num_seconds, validate=True):
        """ Used to crop the profiles to a period of less than 1 day 
        
            num_seconds is the number of seconds of the new outage
                period
        """

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

    def get_dc_to_ac(self):
        """ Returns the dc to ac ratio. """

        # Get total DC power per inverter
        dc = self.advanced_inputs['module']['capacity'] * \
             self.advanced_inputs['strings']['mods_per_string'] * \
             self.advanced_inputs['strings']['strings_per_inv'] * 1000

        # Get inverter AC power
        inverter_list = pvsystem.retrieve_sam(
            self.advanced_inputs['inverter']['database'])
        inverter = inverter_list[self.advanced_inputs['inverter']['model']]
        ac = inverter['Paco']

        return dc / ac

    def get_losses(self):
        """ Returns the system losses. """

        # Calculate losses
        params = {key: val for key, val in self.advanced_inputs.items()
                  if key in ['soiling', 'shading', 'snow', 'mismatch',
                             'wiring', 'connections', 'lid',
                             'nameplate_rating', 'age', 'availability']}
        losses = pvsystem.pvwatts_losses(**params)

        return losses




def calc_pv_prod(tidal_profile, latitude, longitude, depth, validate=True,
                 suppress_warnings=False, advanced_inputs={}):
    """ Calculates the production from a tidal profile. """

    if validate:
        # Put arguments in a dict
        args_dict = {'tidal_profile': tidal_profile,
                     'latitude': latitude,
                     'longitude': longitude, 'altitude': altitude}

        # Validate all parameters
        validate_all_parameters(args_dict)


    # Calculate DC power
        dc_power = pvsystem.singlediode(photocurrent, saturation_current, resistance_series,
                                        resistance_shunt, nNsVth)

    # If dc power exceeds the rated module power by more than 110% print out a warning
    if len(dc_power[dc_power['p_mp'] > module_name['capacity'] * 1000 * 1.1]):
        message = 'Warning: DC power exceeds module rating by more than ' \
                  '110% for the following timesteps.'
        log_error(message)
        if not suppress_warnings:
            print(message)
            print((dc_power.loc[dc_power['p_mp'] > module_name['capacity']
                                * 1000 * 1.1, 'p_mp'] /
                   module_name['capacity'] / 1000 * 100).to_frame().rename(
                columns={'p_mp': 'percentage of module power'}))

    # Calculate losses
    params = {key: val for key, val in advanced_inputs.items()
              if key in ['soiling', 'shading', 'snow', 'mismatch', 'wiring',
                         'connections', 'lid', 'nameplate_rating', 'age', 'availability']}
    losses = pvsystem.pvwatts_losses(**params)
    dc_power['p_mp'] = dc_power['p_mp'] * (1 - losses / 100)

    # If dhi was < 0 at any point, plot power to make sure it looks reasonable
    if plot_power:
        plt.figure()
        dc_power['p_mp'].plot(title='DC Power for {:.0f}W system'.format(
            module_name['capacity'] * 1000))

    # Create strings of modules for conversion to AC power
    dc_power['v_mp'] *= strings['mods_per_string']
    dc_power['p_mp'] *= strings['mods_per_string'] * strings['strings_per_inv']

    # Calculate AC power
    ac_power = pvsystem.inverter.sandia(dc_power.v_mp, dc_power.p_mp, inverter)

    # Force values less than 0 to 0
    ac_power[ac_power < 0] = 0

    # Divide by the total DC power to get the power (in kW) per 1kW of solar
    return ac_power / (module_name['capacity'] * 1000 *
                       strings['mods_per_string'] * strings['strings_per_inv'])


if __name__ == "__main__":
    # Used for testing
    # Create a TidalProfileGenerator object
    latitude = 46.34
    longitude = -119.28
    timezone = 'US/Pacific'
    tpg = TidalProfileGenerator(latitude, longitude, timezone, 0, 5, 14, validate=True)
    print('generation successful')

    tpg.get_tidal_data_from_upload()
    print('uploaded data')

    tpg.extract_tidal_constituents()
    print('extracted tidal constituents')

    tpg.extrapolate_tidal_epoch()
    print('extrapolated tidal epoch')

    tpg.generate_tidal_profiles()
    print('generated tidal profiles')

    tpg.get_power_profiles()
    print('calculated power')

    spg.tidal_checks()
