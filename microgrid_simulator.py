# -*- coding: utf-8 -*-
"""

Microgrid simulator class. Includes the core of the system dispatch algorithm.

File Contents:
    Classes:
        Simulator
        PVBattGenSimulator (inherits from Simulator)
        
    Standalone functions:
        calculate_load_duration
        
"""

import numpy as np
import pandas as pd
from copy import deepcopy

from microgrid_system import Generator
from microgrid_system import PV, SimpleLiIonBattery, SimpleMicrogridSystem
from generate_solar_profile import SolarProfileGenerator
from validation import validate_all_parameters, log_error

# Suppress pandas warnings
pd.options.mode.chained_assignment = None


class Simulator:
    """ 
    Runs the core algorithm of the simulation given one system
        configuration and one solar/temp/load profile.
        
    Parameters
    ----------

        load_profile: Pandas series with the load profile for a given 
            simulation period      
            
        system: MicrogridSystem object
        
        location: dictionary with the following keys and value
            datatypes:
            {'longitude': float, 'latitude': float, 'timezone': string, 
            'altitude': float}
            
        name: unique simulator name
        
    Methods
    ----------
    
        get_name: return simulator name
            
    """
    
    def __init__(self, name, load_profile, system, location, validate=True):
        self.name = name
        self.load_profile = load_profile
        self.system = system
        self.location = location

        # Validate input parameters
        if validate:
            args_dict = {'load_profile': load_profile, 'system': system,
                         'location': location}
            validate_all_parameters(args_dict)
        
    def get_name(self):
        return self.name


class PVBattGenSimulator(Simulator):
    """ 
    Simulates a system with PV, a battery and a backup generator 
    
    Parameters
    ----------

        name: unique simulator name
        
        base_power_profile: Pandas series with a PV power profile for a 1kW system
        
        temp_profile: Pandas dataframe with temperature profile
        
        load_profile: Pandas series with the load profile for a given simulation period
            
        night_profile: Pandas dataframe with info on whether it is night
                 
        system: MicrogridSystem object
        
        location: dictionary with the following keys and value
            datatypes:
            {'longitude': float, 'latitude': float, 'timezone': string, 
            'altitude': float}
            
        duration: Timestep duration in seconds

        dispatch_strategy: determines the battery dispatch strategy.
            Options include:
                night_const_batt (constant discharge at night)
                night_dynamic_batt (updates the discharge rate based on remaining available
                    capacity)
        
        generator_buffer: Buffer between chosen generator size and
            maximum required power. E.g. a buffer of 1.1 means a
            generator has to be sized 10% larger than the maximum power.
            Default = 1.1
        
    Methods
    ----------

        scale_power_profile: Scale power profile by capacity of PV system
    
        calc_dispatch: Runs the battery dispatch algorithm
        
        calc_timestep_dispatch: Calculates battery dispatch for an individual timestep

        size_single_generator: Size the generator(s) based on load not met by PV and
            batteries, given several different generator models

        calc_existing_generator_dispatch: Determines how an existing generator meets the load
            and consumes fuel. An additional generator may be added if the existing one cannot
            meet the load at all timesteps.
        
        get_load_breakdown: Returns load_breakdown attribute
        
        get_storage_recovery_percent: Returns storage_recovery_percent attribute
        
        get_fuel_used: Returns fuel_used_gal attribute
        
        get_generator_power: Returns generator_power_kW attribute
        
        get_load_duration_df: Returns load_duration+df attribute

        get_pv_avg: Returns mean of self.dispatch_df['pv_power'] attribute (excluding 0's)

        get_pv_peak: Returns max of self.dispatch_df['pv_power'] attribute

        get_gen_avg: Returns mean of self.dispatch_df['load_not_met'] attribute (excluding 0's)

        get_gen_peak: Returns max of self.dispatch_df['load_not_met'] attribute

        get_batt_avg: Returns mean of self.dispatch_df['delta_battery_power'] attribute (excluding 0's and negative values)

        get_batt_peak: Returns max of self.dispatch_df['delta_battery_power'] attribute

    Calculated Attributes
    ----------

        scaled_power_profile: PV power profile scaled to PV system capacity
            
        soc_at_initial_hour_of_night: Tracks the SOC at the beginning of each night to
            determine the nightly battery discharge rate
            
        load_breakdown: The fraction of load met by each component

        storage_recovery_percent: The percentage of unused PV that is recovered by the battery
            
        fuel_used_gal: The total fuel used by the generator(s) in gallons
            
        generator_power_kW: The rated power of the chosen generator used to calculate fuel
            consumption
            
        generator_obj: The generator object for the chosen generator
            
        dispatch_df: Pandas dataframe containing dispatch info for each timestep.
            Includes the columns:
            ['load', 'pv_power', 'battery_soc', 'delta_battery_power', 
             'excess_PV', 'load_not_met']
            
        load_duration_df: Pandas dataframe containing load duration curve, with columns:
            [load_bin, num_hours, num_hours_at_or_below] 
    
    """
    
    def __init__(self, name, base_power_profile, temp_profile, load_profile,  night_profile,
                 system, location, duration, dispatch_strategy, generator_buffer=1.1,
                 validate=True):
        self.name = name
        self.base_power_profile = base_power_profile
        self.scaled_power_profile = None
        self.temp_profile = temp_profile
        self.load_profile = load_profile
        self.night_profile = night_profile
        self.system = system
        self.location = location
        self.duration = duration
        self.dispatch_strategy = dispatch_strategy
        self.soc_at_initial_hour_of_night = 0
        self.generator_buffer = generator_buffer
        self.load_breakdown = {}
        self.storage_recovery_percent = None
        self.fuel_used_gal = None
        self.generator_power_kW = None
        self.generator_obj = None
        self.dispatch_df = None
        self.load_duration_df = None
        self.night_hours_left = 0

        # Validate input parameters
        if validate:
            args_dict = {'base_power_profile': base_power_profile,
                         'temp_profile': temp_profile,
                         'load_profile': load_profile,
                         'night_profile': night_profile, 'system': system,
                         'location': location, 'duration': duration,
                         'generator_buffer': generator_buffer,
                         'dispatch_strategy': dispatch_strategy}
            validate_all_parameters(args_dict)

            # Check that all profiles have the same index (although the year will differ)
            if ((self.base_power_profile.index[0].month,
                 self.base_power_profile.index[0].day,
                 self.base_power_profile.index[0].hour) !=
                (self.load_profile.index[0].month,
                 self.load_profile.index[0].day,
                 self.load_profile.index[0].hour)) or \
               ((self.base_power_profile.index[0].month,
                 self.base_power_profile.index[0].day,
                 self.base_power_profile.index[0].hour) !=
                (self.temp_profile.index[0].month,
                 self.temp_profile.index[0].day,
                 self.temp_profile.index[0].hour)) or \
               ((self.base_power_profile.index[0].month,
                 self.base_power_profile.index[0].day,
                 self.base_power_profile.index[0].hour) !=
                (self.night_profile.index[0].month,
                 self.night_profile.index[0].day,
                 self.night_profile.index[0].hour)):

                message = 'The pv power, load, temperature, and night ' \
                          'profiles must all have the same index.'
                log_error(message)
                raise Exception(message)

    def scale_power_profile(self):
        """ Scale power profile by capacity of PV system """
        
        self.scaled_power_profile = self.base_power_profile \
            * self.system.components['pv'].pv_capacity
        
    def calc_dispatch(self):
        """ 
        Runs dispatch algorithm
        
        The dataframe dispatch_df holds the information on the system for each timestep, with
            the following columns:
            - load: load in kW
            - pv_power: AC power produced by PV minus efficiency and inverter losses
            - battery_soc: the battery state of charge at the end of the timestep (as a
                fraction)
            - delta_battery_power: the amount of power charged or discharged from the battery
                minus efficiency and inverter losses
                
        """
        
        # Create dataframe to hold dispatch info for each timestep
        self.dispatch_df = pd.DataFrame(index=self.scaled_power_profile.index,
                                        columns=['load', 'pv_power'])
        self.dispatch_df['load'] = self.load_profile.values
        self.dispatch_df['pv_power'] = self.scaled_power_profile
        self.dispatch_df = pd.concat([self.dispatch_df, self.night_profile], axis=1)
        
        # Calculate battery SOC and power change at each timestep      
        battery_state_df = pd.DataFrame(list(self.dispatch_df.apply(
            lambda x: self.calc_timestep_dispatch(
                x['load'], x['pv_power'], None, self.duration, x['is_night'],
                x['is_first_hour_of_night'], x['night_duration']),
            axis=1).values), columns=['battery_soc', 'delta_battery_power'],
            index=self.dispatch_df.index)
        self.dispatch_df = pd.concat([self.dispatch_df, battery_state_df], axis=1)

        # Calculate battery change in power, soc, and excess PV at each timestep
        self.dispatch_df['delta_battery_power'] = \
            self.dispatch_df['delta_battery_power'].astype('float')
        self.dispatch_df['battery_soc'] = \
            self.dispatch_df['battery_soc'].astype('float')
        self.dispatch_df['excess_PV'] = \
            self.dispatch_df['pv_power'] - self.dispatch_df['load']

        # Calculate load not met
        self.dispatch_df['load_not_met'] = \
            self.dispatch_df['load'] - self.dispatch_df['pv_power'] \
            - self.dispatch_df['delta_battery_power']
        self.dispatch_df.loc[self.dispatch_df['load_not_met'] < 0, 'load_not_met'] = 0

        # Calculate load breakdown by each component
        self.load_breakdown['pv'] = 1 + self.dispatch_df.loc[
            self.dispatch_df['excess_PV'] <= 0, 'excess_PV'].sum() \
            / self.dispatch_df['load'].sum()
        self.load_breakdown['battery'] = self.dispatch_df.loc[
            self.dispatch_df['delta_battery_power'] >= 0,
            'delta_battery_power'].sum() / self.dispatch_df['load'].sum()
        self.load_breakdown['generator'] = \
            self.dispatch_df['load_not_met'].sum() / self.dispatch_df['load'].sum()
        
        # Calculate ES recovery percent
        # If there is no PV, this will cause a RuntimeWarning, so set to 0 (try/except won't
        #   catch Warnings)
        if len(self.dispatch_df.loc[self.dispatch_df['excess_PV'] > 0]):
            self.storage_recovery_percent = \
                np.abs(self.dispatch_df.loc[
                    self.dispatch_df['delta_battery_power'] < 0,
                    'delta_battery_power'].sum()
                       / self.dispatch_df.loc[
                           self.dispatch_df['excess_PV'] > 0,
                           'excess_PV'].sum() * 100)
        else:
            self.storage_recovery_percent = 0
            
    def calc_timestep_dispatch(self, load, pv_power, temperature, duration,
                               is_night, is_first_hour_of_night,
                               night_duration):
        """ Calculates dispatch for an individual timestep. """
        
        # Get current battery state
        initial_soc, voltage, cycles, time_since_last_used = \
            self.system.components['battery'].get_state()
                
        # Get net load after PV applied
        net_load = load - pv_power
        
        # If first hour of night, update soc_at_initial_hour_of_night
        if is_first_hour_of_night:
            self.soc_at_initial_hour_of_night = deepcopy(initial_soc)
            self.night_hours_left = night_duration
        elif self.night_hours_left > 0:
            self.night_hours_left -= 1
        elif is_night and self.night_hours_left <= 0:
            message = 'Night-time with no hours left at night: ' \
                      'night_hours_left {}.'.format(self.night_hours_left)
            log_error(message)
            raise Exception(message)

        # Call battery update model
        delta_power = self.system.components['battery'].update_state(
            net_load, duration, temperature, is_night, night_duration,
            self.night_hours_left, self.soc_at_initial_hour_of_night,
            self.dispatch_strategy)

        # Check for errors
        if delta_power is None:
            print("Error message: net load: {}, initial soc: {}"
                  "".format(net_load, initial_soc))
            
        # Return initial SOC and power charged or discharged
        return initial_soc, delta_power
        
    def size_single_generator(self, generator_options, validate=True):
        """ 
        Size the generator(s) based on load not met by PV and batteries
            and several different generator models.
            
        """

        # Validate input parameters
        if validate:
            args_dict = {'generator_costs': generator_options}
            validate_all_parameters(args_dict)

        # Calculate generator usage and fuel required to meet load not met
        # Total rated power (including multiple units together) based on max unmet power
        max_power = self.dispatch_df['load_not_met'].max()
        
        # Find the smallest generator(s) with sufficient rated power, assumes generators are
        #   sorted from smallest to largest. If no single generator is large enough, try
        #   multiple gensets of the same size
        gen = None
        num_gen = 1
        while gen is None:
            # Find an appropriately sized generator
            best_gen = generator_options[generator_options.index
                                         * num_gen >= max_power
                                         * self.generator_buffer]

            # If no single generator is large enough, increase the number of generators
            if not len(best_gen):
                num_gen += 1
            else:
                # Create generator object
                best_gen = best_gen.iloc[0]
                self.generator_power_kW = best_gen.name*num_gen
                gen = Generator(existing=False,
                                rated_power=best_gen.name,
                                num_units=num_gen,
                                fuel_curve_model=best_gen[
                                    ['1/4 Load (gal/hr)', '1/2 Load (gal/hr)',
                                     '3/4 Load (gal/hr)', 'Full Load (gal/hr)']].to_dict(),
                                capital_cost=best_gen['Cost (USD)'],
                                validate=False)
                self.generator_obj = gen

        # Calculate the load duration and total fuel used
        grouped_load, self.fuel_used_gal = gen.calculate_fuel_consumption(
            self.dispatch_df[['load_not_met']], self.duration, validate=False)
        self.load_duration_df = calculate_load_duration(grouped_load, validate=False)
            
    def calc_existing_generator_dispatch(self, generator_options, validate=True):
        """ 
        If there is an existing generator, determine how it meets the load and consumes fuel.
            
        If add_additional_generator is set to True, an additional generator may be dispatched
            to meet any unmet load, and the load duration curve for that additional generator
            is returned along with the total fuel used. If it is set to False an empty load
            duration curve is returned.

        Note: this function is currently not used

        """

        # Validate input parameters
        if validate:
            args_dict = {'generator_costs': generator_options}
            validate_all_parameters(args_dict)

        # Get info from existing generator
        gen = self.system.components['generator']

        self.generator_power_kW = gen.rated_power
        
        # Create a temporary dataframe to hold load not met cropped at the existing generator
        #   rated power to calculate fuel used by existing generator
        temp_df = self.dispatch_df.copy()
        temp_df.loc[temp_df['load_not_met'] > gen.rated_power,
                    'load_not_met'] = gen.rated_power
        grouped_load, existing_gen_fuel_used = gen.calculate_fuel_consumption(
            temp_df[['load_not_met']], self.duration, validate=False)
        temp_load_duration_curve = calculate_load_duration(grouped_load, validate=False)

        # Determine if unmet load can be supplied by existing generator
        self.dispatch_df['load_not_met_by_generator'] = \
            self.dispatch_df['load_not_met'] - gen.rated_power
        self.dispatch_df.loc[self.dispatch_df['load_not_met_by_generator'] < 0,
                             'load_not_met_by_generator'] = 0


        # If the existing generator can meet load, use empty load duration curve and existing
        #   fuel used

        self.load_duration_df = pd.DataFrame(
            0, index=temp_load_duration_curve.index,
            columns=temp_load_duration_curve.columns)
        self.fuel_used_gal = existing_gen_fuel_used
        
    def get_load_breakdown(self):
        return self.load_breakdown
        
    def get_storage_recovery_percent(self):
        return self.storage_recovery_percent
        
    def get_fuel_used(self):
        return self.fuel_used_gal
        
    def get_generator_power(self):
        return self.generator_power_kW
        
    def get_load_duration_df(self):
        return self.load_duration_df

    def get_pv_avg(self):
        non_zero_pv_power = self.dispatch_df[self.dispatch_df['pv_power'] != 0]
        if len(non_zero_pv_power) > 0:
            return non_zero_pv_power['pv_power'].mean()
        else:
            return 0

    def get_pv_peak(self):
        return self.dispatch_df['pv_power'].max()

    def get_gen_avg(self):
        non_zero_gen_power = self.dispatch_df[self.dispatch_df['load_not_met'] != 0]
        if len(non_zero_gen_power) > 0:
            return non_zero_gen_power['load_not_met'].mean()
        else:
            return 0

    def get_gen_peak(self):
        return self.dispatch_df['load_not_met'].max()

    def get_batt_avg(self):
        greater_than_zero_batt_power = self.dispatch_df[self.dispatch_df['delta_battery_power'] > 0]
        if len(greater_than_zero_batt_power) > 0:
            return greater_than_zero_batt_power['delta_battery_power'].mean()
        else:
            return 0

    def get_batt_peak(self):
        return self.dispatch_df['delta_battery_power'].max()


def calculate_load_duration(grouped_load, validate=True):
    """
    Create a load duration curve for a single generator.

    Inputs:
        grouped_load: dataframe with columns [binned_load, num_hours]

    Outputs:
        load_duration_df: dataframe with columns
            [load_bin, num_hours, num_hours_at_or_below,
            num_hours_above, energy_not_met_at_load_level,
            energy_not_met_above_load_level, max_power_not_met]

    """

    # Validate input parameters
    if validate:
        args_dict = {'grouped_load': grouped_load}
        validate_all_parameters(args_dict)

    # Set index as binned_load and fill in missing bins
    grouped_load = grouped_load.set_index('binned_load')[['num_hours']]
    grouped_load = grouped_load.merge(pd.DataFrame(
        index=range(0, grouped_load.index[-1]+1)), left_index=True,
        right_index=True, how='right').fillna(0)

    # Calculate cumulative hours at or below each load bin
    grouped_load['num_hours_at_or_below'] = grouped_load['num_hours'].cumsum()

    # Calculate cumulative hours above each load bin (hours not met)
    grouped_load['num_hours_above'] = \
        grouped_load['num_hours_at_or_below'].max() \
        - grouped_load['num_hours_at_or_below']

    # Calculate energy not met at each load bin
    grouped_load['energy_not_met_at_load_level'] = grouped_load.index \
        * grouped_load['num_hours']
    grouped_load['energy_not_met_above_load_level'] = \
        grouped_load['energy_not_met_at_load_level'].sum() \
        - grouped_load['energy_not_met_at_load_level'].cumsum() \
        - grouped_load.index * grouped_load['num_hours_above']

    # Calculate the maximum power not met at each load bin
    grouped_load['max_power_not_met'] = -grouped_load.index + grouped_load.index[-1]

    # Divide by the load bin to get max % not met (compared to load bin)
    grouped_load['max_percent_not_met'] = grouped_load['max_power_not_met'] \
        / grouped_load.index * 100

    return grouped_load


if __name__ == "__main__":
    # Used for testing    
    # Get solar and power profiles
    # System level
    import os
    spg = SolarProfileGenerator(46.34, -119.28, 'US/Pacific', 0, 0, 0, 200, 14,
                                validate=False)
    spg.get_power_profiles()
    spg.get_night_duration(percent_at_night=0.2, validate=False)

    # Sample generator options
    generator_options = pd.read_excel('data/MCOR Prices.xlsx', sheet_name='generator_costs',
                                      index_col=0)

    # Create a sample system
    batt = SimpleLiIonBattery(False, 2000, 6000, validate=False)
    pv = PV(False, 4000, 0, 0, 0.360, 3, 2, validate=False, pv_tracking=False,
            pv_racking='ground')
    gen = Generator(True, 50, 1, {'1/4 Load (gal/hr)': 1.8, '1/2 Load (gal/hr)': 2.9,
                                  '3/4 Load (gal/hr)': 3.8, 'Full Load (gal/hr)': 4.8},
                    5000, validate=False)
    system = SimpleMicrogridSystem('pv_359.4_batt_0kW_0kWh')
    system.add_component(batt, validate=False)
    system.add_component(pv, validate=False)
    system.add_component(gen, validate=False)

    # Create a simulation object
    load_profile = pd.read_csv(os.path.join('data', 'sample_load_profile.csv'),
                               index_col=0)['Load']
    load_profile.index = pd.date_range(start='1/1/2017', end='1/1/2018',
                                       freq='{}S'.format(int(3600)))[:-1]
    load_profile = load_profile.iloc[4951:4951+336]
    sim = PVBattGenSimulator('pv_359.4_batt_0kW_0kWh_profile_0',
                             spg.power_profiles[95],
                             spg.temp_profiles[95], load_profile,
                             spg.night_profiles[95], system,
                             {'longitude': -119.28, 'latitude': 46.34,
                              'timezone': 'US/Pacific', 'altitude': 0}, 3600,
                             'night_const_batt', validate=False)

    # Run the simulation
    sim.scale_power_profile()
    sim.calc_dispatch()
    # sim.size_single_generator(generator_options, validate=False)
    # sim.calc_existing_generator_dispatch(generator_options, validate=False)

    # Plot dispatch
    sim.dispatch_df[['load', 'pv_power', 'delta_battery_power', 'load_not_met']].plot()
