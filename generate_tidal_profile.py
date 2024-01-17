# -*- coding: utf-8 -*-
"""
Runs the Alternative Solar Profiles (ASP) algorithm used for solar forecasting and uses
pvlib-python to calculate AC power for each profile.
 

    
File contents:
    Classes:
        TidalProfileGenerator
        
    Standalone functions:
        get_load_tidal_from_upload
        extract_tidal_constituents
        extrapolate_tidal_epoch
        generate_tidal_profiles
        calc_tidal_prod
                   
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

TIDAL_DEFAULTS = {'turbine_length': 1,
               'turbine_type' : 'radial'}


class TidalProfileGenerator:
    """   
    Class to download NREL solar data, create solar profiles using the ASP model, and
        calculate power profiles using pvlib.
    
    Parameters
    ----------
    
        longitude: Site longitude in degrees    
    
        latitude: Site latitude in degrees
    
        timezone: US timezone, options:
            US/Alaska, US/Aleutian, US/Arizona, US/Central, US/East-Indiana, US/Eastern,
            US/Hawaii, US/Indiana-Starke, US/Michigan, US/Mountain, US/Pacific,
            US/Pacific-New, US/Samoa
        
        altitude: Site altitude in meters
    
        tilt: Panel tilt in degrees
    
        azimuth: Panel azimuth in degrees
    
        num_trials: Number of solar profiles to create
    
        length_trials: Length of solar profiles in hours

        pv_racking: Type of pv racking (options: [roof, ground, carport])
            Default = ground

        pv_tracking: Type of tracking (options: [fixed, single_axis])
            Default = fixed

        max_track_angle: Maximum rotation angle (in degrees) for single-axis tracking

        backtrack: Whether or not backtracking is allowed for single-axis tracking
    
        start_year: Start year for solar data download
    
        end_year: End year for solar data download

        solar_source: The source of the solar data to download. The available options are:
            nsrdb: NREL's NSRDB -- CONUS, Central America, and parts of South America and
                Canada
            himawari: Himawari -- Pacific Island and East Asia locations
            Default = nsrdb
    
        num_ghi_states: Number of discrete GHI states for hourly model
    
        num_dni_states: Number of discrete DNI states for hourly model
    
        cld_hours: Hours of the day (range) used to set the day's cloud state
        
        temp_bins: Temperature bins
    
        num_daily_ghi_states: Number of discrete GHI states for daily model
    
        num_daily_dni_states: Number of discrete DNI states for daily model

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
    
        get_solar_data: Downloads solar data from NREL
        
        get_wind_speed: Gets average wind speed from TMY data
        
        get_solar_profiles: Calculates simulated solar profiles based on a solar state
            probability matrix
            
        get_power_profiles: Calculates the output AC power for a 1kW system for each solar and
            temperature profile.

        get_power_profiles_from_upload: Creates power profiles from uploaded 8760 production
            data.
            
        get_night_duration: For each power profile, gets the hours that are at night, and the
            duration of each night
            
        get_pv_params: Gets the module capacity and area for calculating array size
            
        crop_timeline: Used to crop the profiles to a period of less than 1 day

        pv_checks: Creates several plots to verify that the pv power calculation went OK

        get_dc_to_ac: Returns the DC to AC ratio

        get_losses: Returns system power losses
        
    Calculated Attributes
    ----------
    
        wind_speed: median wind speed in m/s

        solar_profiles: list of Pandas dataframes with solar profiles
        
        temp_profiles: list of Pandas dataframes with temperature profiles
        
        power_profiles: list of Pandas series' with PV power profiles for a 1kW system
            
        night_profiles: list of Pandas dataframes with info on whether it is night
            
        tmy_solar_profile: TMY solar profile
        
        tmy_power_profile: TMY power profile for a 1kW system

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
            tidal_current = pd.read_csv(os.path.join(filedir, file))

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
        start_datetimes = tidal_epoch.sample(
            int(self.num_trials)).index.values

        # Create a date range object for each start datetime
        date_ranges = [pd.date_range(start=start_date,
                                     periods=self.length_trials,
                                     freq='H')
                       for start_date in start_datetimes]

        # Create 20-year annual profile to allow for profiles with year-end overlap
        oneyear_profile = tidal_epoch.head(8760)
        twentyyear_profile = tidal_epoch.append(oneyear_profile)
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
        Calculate the output AC power for a 1kW system for each solar and temperature profile.
       
        If read_from_file is True, reads the solar and temperature data  from csv, allowing
            for faster lookup rather than re-running get_solar_data and get_solar_profiles.
            
        """

        # For each solar profile, calculate PV production
        # Load the solar and temperature data from csv
        for i in range(int(self.num_trials)):
            try:
                solar = pd.read_csv(os.path.join(
                    SOLAR_DATA_DIR, 'solar_profiles', '{}_{}_{}d_{}t'.format(
                        self.latitude, self.longitude, int(self.length_trials/24),
                        int(self.num_trials)),
                    '{}_{}_solar_trial_{}.csv'.format(self.latitude, self.longitude, i)),
                    index_col=0, parse_dates=[0])

                # Allow for backward compatibility with solar profiles generated before ASP
                #   code was converted to Python
                if 'temp' not in solar.columns:
                    solar['temp'] = pd.read_csv(os.path.join(
                        SOLAR_DATA_DIR, 'solar_profiles', '{}_{}_{}d_{}t'.format(
                            self.latitude, self.longitude,
                            int(self.length_trials/24),
                            int(self.num_trials)),
                        '{}_{}_temp_trial_{}.csv'.format(self.latitude, self.longitude, i)),
                        index_col=0, parse_dates=[0]).values

            except FileNotFoundError:
                message = 'Solar profile csvs not found. Please check that you have entered' \
                          ' the longitude, latitude, number, and length of trials for a ' \
                          'site with previously generated solar profiles.'
                log_error(message)
                raise Exception(message)

            # Fix timezone
            try:
                solar.index = solar.index.tz_convert(self.timezone)
            except AttributeError:
                # Deal with pandas issue creating datetime index from timeseries including
                #   daylight savings time shift
                solar.index = pd.to_datetime(solar.index, utc=True).tz_convert(self.timezone)

            # Check that the solar data and index are not misaligned (e.g. the sun is up
            #   during the day)
            # Get median solar start hour
            solar['date'] = solar.index.date
            median_hour = solar.groupby('date').apply(
                lambda x: x[x['ghi'] > 0].iloc[0].name.hour
                if len(x[x['ghi'] > 0]) else 0).median()
            if not 4 < median_hour < 12:
                message = 'The solar profiles are mismatched with the index (e.g. the sun ' \
                          'is not shining at the expected times). Make sure your solar ' \
                          'profiles are valid.'
                log_error(message)
                raise Exception(message)

            self.solar_profiles += [solar]
            self.temp_profiles += [solar['temp'].to_frame(name='temp_celcius')]

        # Read raw TMY file
        self.tmy_solar_profile = pd.read_csv(
            os.path.join(SOLAR_DATA_DIR,
                         'nrel',
                         '{}_{}'.format(self.latitude, self.longitude),
                         '{}_{}_tmy.csv'.format(self.latitude, self.longitude)))

        # Get average wind speed
        self.get_wind_speed()

        # Calculate PV production for each solar profile
        for solar, temp in zip(self.solar_profiles, self.temp_profiles):
            self.power_profiles += [calc_pv_prod(
                solar, temp, self.latitude, self.longitude, self.altitude,
                self.tilt, self.azimuth, self.wind_speed,
                self.advanced_inputs['albedo'],
                self.pv_racking,
                self.advanced_inputs['module'],
                self.advanced_inputs['inverter'],
                self.advanced_inputs['strings'],
                pv_tracking=self.pv_tracking,
                max_track_angle=self.max_track_angle,
                backtrack=self.backtrack, validate=False,
                suppress_warnings=self.suppress_warnings,
                advanced_inputs=self.advanced_inputs)]

        # Get TMY solar PV power
        # Parse index
        tmy_solar = self.tmy_solar_profile
        tmy_solar.index = pd.to_datetime(tmy_solar[['Year', 'Month', 'Day',
                                                    'Hour', 'Minute']])

        # Add timezone
        # If using the himawari dataset, get the timezone for the closest station
        if self.solar_source == 'himawari':
            tmy_meta_name = f'{self.latitude}_{self.longitude}_tmy.json'
            tmy_meta_path = os.path.join(
                SOLAR_DATA_DIR, 'nrel', '{}_{}'.format(self.latitude, self.longitude),
                tmy_meta_name)
            with open(tmy_meta_path, 'r') as f:
                tmy_meta = json.load(f)
            if tmy_meta['tz'] < 0:
                tmy_tz = f'Etc/GMT{int(tmy_meta["tz"])}'
            else:
                tmy_tz = f'Etc/GMT+{int(tmy_meta["tz"])}'
            tmy_solar.index = tmy_solar.index.tz_localize('UTC').tz_convert(tmy_tz)
        else:
            tmy_solar.index = tmy_solar.index.tz_localize('UTC').tz_convert(self.timezone)

        # If TMY data is listed as on the hour, add 30 minutes. This is due to a bug in the
        #   NREL NSRDB api that was corrected sometime in mid-2019, so data downloaded before
        #   that has to be corrected.
        if tmy_solar.index[0].minute == 0:
            tmy_solar.index = tmy_solar.index + datetime.timedelta(minutes=30)

        # Un-shift from timezone conversion so it starts at the beginning of the year
        if tmy_solar.index[0].month == 1:
            tmy_solar.index = tmy_solar.index - \
                              datetime.timedelta(hours=tmy_solar.index[0].hour)
        else:
            tmy_solar.index = tmy_solar.index + \
                              datetime.timedelta(hours=24 - tmy_solar.index[0].hour)

        # Rename columns
        tmy_solar.rename(columns={'DHI': 'dhi', 'DNI': 'dni', 'GHI': 'ghi'}, inplace=True)

        # Get power profile
        self.tmy_power_profile = calc_pv_prod(
            tmy_solar, tmy_solar['Temperature'].to_frame(name='temp_celcius'),
            self.latitude, self.longitude, self.altitude, self.tilt,
            self.azimuth, self.wind_speed, self.advanced_inputs['albedo'],
            self.pv_racking, self.advanced_inputs['module'],
            self.advanced_inputs['inverter'], self.advanced_inputs['strings'],
            pv_tracking=self.pv_tracking, max_track_angle=self.max_track_angle,
            backtrack=self.backtrack, validate=False,
            suppress_warnings=self.suppress_warnings,
            advanced_inputs=self.advanced_inputs)



    def get_pv_params(self):
        """ Get the module capacity and area for calculating array size. """

        return self.advanced_inputs['module']

    def crop_timeline(self, num_seconds, validate=True):
        """ Used to crop the profiles to a period of less than 1 day 
        
            num_seconds is the number of seconds of the new outage
                period
        """

        # Validate parameters
        if validate:
            args_dict = {'num_seconds': num_seconds}
            validate_all_parameters(args_dict)

        # For each profile in solar_profiles, power_profiles, temp_profiles and
        #   night_profiles, crop to the specified number of seconds, rounding down to the
        #   nearest timestep
        for i in range(len(self.solar_profiles)):
            self.solar_profiles[i] = \
                self.solar_profiles[i].loc[
                :self.solar_profiles[i].index[0] + datetime.timedelta(
                    seconds=num_seconds - 1)]

        for i in range(len(self.power_profiles)):
            self.power_profiles[i] = \
                self.power_profiles[i].loc[
                :self.power_profiles[i].index[0] + datetime.timedelta(
                    seconds=num_seconds - 1)]

        for i in range(len(self.temp_profiles)):
            self.temp_profiles[i] = \
                self.temp_profiles[i].loc[
                :self.temp_profiles[i].index[0] + datetime.timedelta(
                    seconds=num_seconds - 1)]

        for i in range(len(self.night_profiles)):
            self.night_profiles[i] = \
                self.night_profiles[i].loc[
                :self.night_profiles[i].index[0] + datetime.timedelta(
                    seconds=num_seconds - 1)]

    def pv_checks(self):
        """ Several checks to  make sure the pv profiles look OK. """

        # Get the profiles with the min and max PV energy
        total_energy = [prof.sum() for prof in self.power_profiles]
        max_profile_num = np.where(total_energy == max(total_energy))[0][0]
        min_profile_num = np.where(total_energy == min(total_energy))[0][0]

        # Plot the profiles with min and max pv energy
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        self.power_profiles[max_profile_num].plot(
            ax=ax1, title='Profile with max energy generation')
        ax1.set_ylabel('Power (kW)')
        ax2 = fig.add_subplot(122)
        self.power_profiles[min_profile_num].plot(
            ax=ax2, title='Profile with min energy generation')
        ax2.set_ylabel('Power (kW)')

        # Plot the TMY profile
        fig = plt.figure()
        temp_profile = self.tmy_power_profile.copy(deep=True)
        temp_profile.index = temp_profile.index.map(
            lambda x: x.replace(year=2017))
        temp_profile.plot(title='TMY power profile')
        plt.ylabel('Power (kW)')

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





def calc_pv_prod(solar_profile, temp_profile, latitude, longitude, altitude, tilt, azimuth,
                 wind_speed, albedo, pv_racking, module_name, inverter_name, strings,
                 pv_tracking='fixed', max_track_angle=90, backtrack=True, validate=True,
                 suppress_warnings=False, advanced_inputs={}):
    """ Calculates the PV production from a solar profile using pvlib. """

    if validate:
        # Put arguments in a dict
        args_dict = {'solar_profile': solar_profile,
                     'temp_profile': temp_profile, 'latitude': latitude,
                     'longitude': longitude, 'altitude': altitude,
                     'tilt': tilt, 'azimuth': azimuth,
                     'wind_speed': wind_speed, 'albedo': albedo,
                     'pv_racking': pv_racking, 'module': module_name,
                     'inverter': inverter_name, 'strings': strings}

        # Validate all parameters
        validate_all_parameters(args_dict)

    # Get solar position
    # Try using numba if installed to speed up calculation
    # Catch UserWarnings from pvlib/numba
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        try:
            solpos = solarposition.get_solarposition(solar_profile.index, latitude,
                                                     longitude, altitude, method='nrel_numba')
        except:
            solpos = solarposition.get_solarposition(solar_profile.index, latitude, longitude,
                                                     altitude)

    # Calculate extraterrestrial irradiance
    # try/except block to allow for old or new versions of pvlib-python
    try:  # new version
        dni_extra = irradiance.get_extra_radiation(solar_profile.index)
    except:  # old version
        dni_extra = irradiance.extraradiation(solar_profile.index)

    # Calculate dhi or ghi if not included (not calculated by ASP code)
    zenith_angle = np.array([elem if elem < 90 else 90
                             for elem in solpos['apparent_zenith'].values])
    if 'dhi' not in solar_profile:
        solar_profile['dhi'] = solar_profile['ghi'] - \
                               (solar_profile['dni'] * np.cos(zenith_angle * math.pi / 180))
    if 'ghi' not in solar_profile:
        solar_profile['ghi'] = solar_profile['dhi'] + \
                               (solar_profile['dni'] * np.cos(zenith_angle * math.pi / 180))

    # Make sure dhi not negative
    plot_power = False
    if len(solar_profile[solar_profile['dhi'] < 0]):
        message = 'Warning: dhi value is negative at the following times:{}, check to make ' \
                  'sure the ghi and dni values are consistent with the timestamp.'.format(
            solar_profile[solar_profile['dhi'] < 0].index)
        log_error(message)
        if not suppress_warnings:
            plot_power = True
            print(message)
        solar_profile.loc[solar_profile['dhi'] < 0, 'dhi'] = 0

    # Deal with issue of poa_sky_diffuse blowing up if solar_zenith too close to 90
    solpos.loc[(solpos['apparent_zenith'] > 87) &
               (solpos['apparent_zenith'] <= 90),
               'apparent_zenith'] = 87.
    solpos.loc[(solpos['apparent_zenith'] > 90) &
               (solpos['apparent_zenith'] < 93),
               'apparent_zenith'] = 93.

    # If single-axis tracking is used, calculate new tilt, azimuth, and aoi
    if pv_tracking == 'single_axis':
        tracker_data = tracking.singleaxis(solpos['apparent_zenith'],
                                           solpos['azimuth'], axis_tilt=tilt,
                                           axis_azimuth=azimuth,
                                           max_angle=max_track_angle,
                                           backtrack=backtrack)
        tracker_data.fillna(0, inplace=True)
        surface_tilt = tracker_data['surface_tilt']
        surface_azimuth = tracker_data['surface_azimuth']
        aoi = tracker_data['aoi']
    else:
        surface_tilt = tilt
        surface_azimuth = azimuth
        aoi = irradiance.aoi(tilt, azimuth, solpos['apparent_zenith'], solpos['azimuth'])

    # Calculate plane of array diffuse sky radiation using the Hay Davies model
    poa_sky_diffuse = irradiance.haydavies(surface_tilt, surface_azimuth,
                                           solar_profile['dhi'],
                                           solar_profile['dni'], dni_extra,
                                           solpos['apparent_zenith'],
                                           solpos['azimuth'])

    # try/except block to allow for old or new versions of pvlib-python
    try:  # new
        # Calculate ground diffuse
        poa_ground_diffuse = irradiance.get_ground_diffuse(
            surface_tilt, solar_profile['ghi'], albedo=albedo)

        # Calculate POA total
        poa_irrad = irradiance.poa_components(aoi, solar_profile['dni'],
                                              poa_sky_diffuse,
                                              poa_ground_diffuse)
    except:  # old
        # Calculate ground diffuse
        poa_ground_diffuse = irradiance.grounddiffuse(surface_tilt,
                                                      solar_profile['ghi'],
                                                      albedo=albedo)

        # Calculate POA total
        poa_irrad = irradiance.globalinplane(aoi, solar_profile['dni'],
                                             poa_sky_diffuse,
                                             poa_ground_diffuse)

    # Get pvsystem racking type based on racking parameter
    racking_dict = {'ground': 'open_rack_glass_glass',
                    'roof': 'close_mount_glass_glass',
                    'carport': 'open_rack_glass_glass'}

    # Model parameters from pvlib.temperature
    temp_model_params = TEMPERATURE_MODEL_PARAMETERS['sapm'][racking_dict[pv_racking]]

    # Calculate cell and module temperature
    pvtemps = sapm_cell(poa_irrad['poa_global'], wind_speed=wind_speed,
                        temp_air=temp_profile['temp_celcius'],
                        **temp_model_params)

    # Select module and inverter
    try:
        module_list = pvsystem.retrieve_sam(module_name['database'])
        module = module_list[module_name['model']]
    except (ValueError, KeyError):
        message = 'PV module not in database. Please check pvlib module options.'
        log_error(message)
        raise Exception(message)
    try:
        inverter_list = pvsystem.retrieve_sam(inverter_name['database'])
        inverter = inverter_list[inverter_name['model']]
    except (ValueError, KeyError):
        message = 'Inverter not in database. Please check pvlib module options.'
        log_error(message)
        raise Exception(message)

    # Calculate DC power
    # try/except block to account for pvlib-python update which modified function arguments
    try:
        photocurrent, saturation_current, resistance_series, \
        resistance_shunt, nNsVth = pvsystem.calcparams_desoto(
            poa_irrad['poa_global'], temp_cell=pvtemps['temp_cell'],
            alpha_isc=module['alpha_sc'], module_parameters=module,
            EgRef=1.121, dEgdT=-0.0002677)
    except:
        photocurrent, saturation_current, resistance_series, \
        resistance_shunt, nNsVth = pvsystem.calcparams_desoto(
            poa_irrad['poa_global'], pvtemps,
            module['alpha_sc'], module['a_ref'], module['I_L_ref'],
            module['I_o_ref'],
            module['R_sh_ref'], module['R_s'])
    # Properly catch RuntimeWarnings not caught in pvlib
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
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


def calc_night_duration(power_profile, percent_at_night=0, validate=True):
    """ 
    For each timestep in a solar or power profile, determine if it is night (or there is no PV
        generation), if it is the first timestep of the night and the total night duration.
        
    If percent_at_night is specified, it is considered nighttime when the pv power is at
        max(power_profile) * percent_at_night, allowing a buffer before sundown. It is allowed
        to range from 0 to 1.

    """

    if validate:
        # Put arguments in a dict
        args_dict = {'percent_at_night': percent_at_night}

        # Validate all parameters
        validate_all_parameters(args_dict)

    # Create dataframe to hold night info for each timestep
    night_df = pd.DataFrame(index=power_profile.index)

    # Determine which timesteps are during the nighttime, add to dataframe
    night_df['is_night'] = power_profile.values <= power_profile.max() * percent_at_night

    # Calculate duration for each night (or 0 PV period)
    temp1 = night_df['is_night'].cumsum().value_counts()
    temp2 = np.array(temp1[temp1 > 1].sort_index().index)
    if night_df.iloc[-1, -1]:
        temp2 = np.append(temp2, np.array(temp1.sort_index().index[-1]))
    temp2 = temp2[np.where(temp2 > 0)]
    night_lengths = [temp2[0]] + list(temp2[1:] - temp2[:-1])

    # Find the first hour of each night (or 0 PV period)
    night_df['is_first_hour_of_night'] = False
    night_df['night_duration'] = np.nan
    for i, _ in enumerate(night_df.iterrows()):
        if (i == 0 and night_df.iloc[0, -3]) or \
                (i > 0 and night_df.iloc[i, -3]
                 and not night_df.iloc[i - 1, -3]):
            night_df.iloc[i, -2] = True
            night_df.iloc[i, -1] = night_lengths.pop(0)
    night_df['night_duration'].ffill(inplace=True)
    night_df.loc[night_df['is_night'] == False, 'night_duration'] = 0

    return night_df


if __name__ == "__main__":
    # Used for testing
    # Create a SolarProfileGenerator object
    latitude = 46.34
    longitude = -119.28
    timezone = 'US/Pacific'
    spg = SolarProfileGenerator(latitude, longitude, timezone, 0, 0, 0, 5, 14,
                                pv_tracking='fixed', validate=True,
                                start_year=1998, end_year=2020,
                                solar_source='nsrdb')
    print('generation successful')

    # Download NREL profiles
    spg.get_solar_data()
    print('downloaded data')

    # Get solar profiles from ASP code
    spg.get_solar_profiles()
    print('generated profiles')

    # Get power profiles using pvlib-python
    spg.get_power_profiles()
    print('calculated power')

    spg.pv_checks()
