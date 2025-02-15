# -*- coding: utf-8 -*-
"""

Microgrid simulator class. Includes the core of the system dispatch algorithm.

File Contents:
    Classes:
        Simulator
        REBattGenSimulator (inhertis from Simulator)
        
    Standalone functions:
        calculate_load_duration
        
"""

import copy
import numpy as np
import pandas as pd
from copy import deepcopy
from microgrid_system import Generator
from microgrid_system import PV, Tidal, Wave, SimpleLiIonBattery, SimpleMicrogridSystem
from generate_solar_profile import SolarProfileGenerator
from generate_tidal_profile import TidalProfileGenerator
from validation import validate_all_parameters, log_error

# Suppress pandas warnings
pd.options.mode.chained_assignment = None


class Simulator:
    """ 
    Runs the core algorithm of the simulation given one system
        configuration and one RE/load profile.
        
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


class REBattGenSimulator(Simulator):
    """ 
    Simulates a system with one or more renewable resources, a battery and a backup generator 
    
    Parameters
    ----------

        name: unique simulator name

        renewable_resources: List of renewable resources in order that they will be dispatched
        
        base_power_profiles: Dictionaries of Pandas series' with a renewable power profile for a 1kW system,
            the keys should be the same as the resources from 'renewable_resources', also includes Pandas 
            series for night profiles if PV is included in the resources
        
        load_profile: Pandas series with the load profile for a given simulation period
                 
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
                available_capacity
        
        generator_buffer: Buffer between chosen generator size and
            maximum required power. E.g. a buffer of 1.1 means a
            generator has to be sized 10% larger than the maximum power.
            Default = 1.1
        
    Methods
    ----------

        scale_power_profiles: Scale power profiles by capacity of each renewable system
    
        calc_dispatch: Runs the battery dispatch algorithm
        
        calc_timestep_dispatch: Calculates battery dispatch for an individual timestep

        size_single_generator: Size the generator(s) based on load not met by RE and
            batteries, given several different generator models

        calc_existing_generator_dispatch: Determines how an existing generator meets the load
            and consumes fuel. An additional generator may be added if the existing one cannot
            meet the load at all timesteps.

        calc_resilience_metrics: Calculates the following resilience metrics for the system with 
        (1) RE, batt + gen, (2) RE, batt (3) gen only:
        	Number of hours before load shortfall
            Fuel use, if all load is met, otherwise -1
            Percent load not met
            Total load not met
            Peak load not met
        
        get_load_breakdown: Returns load_breakdown attribute
        
        get_storage_recovery_percent: Returns storage_recovery_percent attribute
        
        get_fuel_used: Returns fuel_used_gal attribute
        
        get_generator_power: Returns generator_power_kW attribute
        
        get_load_duration_df: Returns load_duration+df attribute

        get_renewable_avg: Returns mean power of each renewable resource

        get_renewable_peak: Returns max power of each renewable resource

        get_gen_avg: Returns mean power generator production

        get_gen_peak: Returns max power generator production

        get_gen_total: Returns total generator production

        get_outage_load: Returns total load from the outage period

        get_hours_before_gen: Returns the number of hours the microgrid can supply power before the generator is used

        get_batt_avg: Returns mean power of battery power supplied to system (excluding 0's and negative values)

        get_batt_peak: Returns max power of battery power supplied to system

    Calculated Attributes
    ----------

        scaled_power_profiles: Dictionary of renewable power profile scaled to system capacities
            
        soc_at_initial_hour_of_night: Tracks the SOC at the beginning of each night to
            determine the nightly battery discharge rate
            
        load_breakdown: The fraction of load met by each component

        storage_recovery_percent: The percentage of unused RE that is recovered by the battery
            
        fuel_used_gal: The total fuel used by the generator(s) in gallons
            
        generator_power_kW: The rated power of the chosen generator used to calculate fuel
            consumption
            
        generator_obj: The generator object for the chosen generator
            
        dispatch_df: Pandas dataframe containing dispatch info for each timestep.
            Includes the columns:
            ['load', '<re_type>_power', 'battery_soc', 'delta_battery_power', 
             'excess_<re_type>', 'gen_power', 'load_not_met', 'load_not_met_by_RE']
            
        load_duration_df: Pandas dataframe containing load duration curve, with columns:
            [load_bin, num_hours, num_hours_at_or_below] 
    
    """
    
    def __init__(self, name, renewable_resources, base_power_profiles, load_profile,
                 system, location, duration, dispatch_strategy, generator_buffer=1.1,
                 validate=True):
        self.name = name
        self.renewable_resources = renewable_resources
        self.base_power_profiles = base_power_profiles
        self.scaled_power_profiles = {}
        self.load_profile = load_profile
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
            args_dict = {'renewable_resources': renewable_resources,
                         'power_profiles': base_power_profiles,
                         'load_profile': load_profile,
                         'system': system,
                         'location': location, 'duration': duration,
                         'generator_buffer': generator_buffer,
                         'dispatch_strategy': dispatch_strategy}
            validate_all_parameters(args_dict)

            # Check that all profiles have the same index (although the year will differ)
            if any([((base_power_profile.index[0].month,
                      base_power_profile.index[0].day,
                      base_power_profile.index[0].hour) !=
                      (self.load_profile.index[0].month,
                       self.load_profile.index[0].day,
                       self.load_profile.index[0].hour)) 
                       for base_power_profile in base_power_profiles.values()]):
                message = 'The RE power, load, and night ' \
                          'profiles must all have the same index.'
                log_error(message)
                raise Exception(message)

    def scale_power_profiles(self):
        """ Scale power profile by capacity of RE system """
        
        # TODO: may want to rethink this to allow for multiple systems of the same resource type
        for re_resource in self.renewable_resources:
             self.scaled_power_profiles[re_resource] = self.base_power_profiles[re_resource] \
            * self.system.components[re_resource].capacity
        
    def calc_dispatch(self):
        """ 
        Runs dispatch algorithm
        
        The dataframe dispatch_df holds the information on the system for each timestep, with
            the following columns:
            - load: load in kW
            - <re_type>_power: AC power produced by each renewable resource in order specified by
                renewable_resources minus efficiency and inverter losses
            - battery_soc: the battery state of charge at the end of the timestep (as a
                fraction)
            - delta_battery_power: the amount of power charged or discharged from the battery
                minus efficiency and inverter losses
                
        """
        
        # Create dataframe to hold dispatch info for each timestep
        index = self.base_power_profiles[self.renewable_resources[0]].index
        self.dispatch_df = pd.DataFrame(index=index,
                                        columns = ['load'] + [f'{re_resource}_power' 
                                        for re_resource in self.renewable_resources])
        self.dispatch_df['load'] = self.load_profile.values
        for re_resource in self.renewable_resources:
                self.dispatch_df[f'{re_resource}_power'] = self.scaled_power_profiles[re_resource]

        # Include night-time params if using a night-based dispatch strategy
        if self.dispatch_strategy == 'available_capacity':
            self.dispatch_df['is_night'] = 0
            self.dispatch_df['is_first_hour_of_night'] = 0
            self.dispatch_df['night_duration'] = 0
        else:
            self.dispatch_df = pd.concat([self.dispatch_df, self.base_power_profiles['night']], axis=1)
        
        # Calculate battery SOC and power change at each timestep      
        battery_state_df = pd.DataFrame(list(self.dispatch_df.apply(
            lambda x: self.calc_timestep_dispatch(
                x['load'], self.renewable_resources,
                {re_resource: x[f'{re_resource}_power'] for re_resource in self.renewable_resources},
                self.duration, x['is_night'],
                x['is_first_hour_of_night'], x['night_duration']),
            axis=1).values), columns=['battery_soc', 'delta_battery_power'],
            index=self.dispatch_df.index)
        self.dispatch_df = pd.concat([self.dispatch_df, battery_state_df], axis=1)

        # Calculate battery change in power, soc at each timestep
        self.dispatch_df['delta_battery_power'] = \
            self.dispatch_df['delta_battery_power'].astype('float')
        self.dispatch_df['battery_soc'] = \
            self.dispatch_df['battery_soc'].astype('float')
        
        # Calculate how much each RE resource is contributing to load at each timestep
        self.dispatch_df['net_load'] = self.dispatch_df['load'].copy(deep=True)
        self.dispatch_df['total_RE'] = 0
        self.dispatch_df['excess_RE'] = 0
        for re_resource in self.renewable_resources:
            self.dispatch_df[f'{re_resource}_power_to_load'] = self.dispatch_df.apply(
                lambda x: np.min([x[f'{re_resource}_power'], x['net_load']]), axis=1)
            self.dispatch_df['net_load'] = self.dispatch_df.apply(
                lambda x: np.max([x['net_load'] - x[f'{re_resource}_power'], 0]), axis=1)
            self.dispatch_df[f'excess_{re_resource}'] = self.dispatch_df.apply(
                lambda x: np.max([x[f'{re_resource}_power'] - x[f'{re_resource}_power_to_load'], 0]),
                axis=1)
            # self.dispatch_df[f'excess_{re_resource}'] = self.dispatch_df[f'{re_resource}_power'] \
            #     - self.dispatch_df[f'{re_resource}_power_to_load']
            self.dispatch_df['total_RE'] += self.dispatch_df[f'{re_resource}_power']
            self.dispatch_df['excess_RE'] += self.dispatch_df[f'excess_{re_resource}']

        # Calculate load not met
        self.dispatch_df['load_not_met_by_RE'] = self.dispatch_df['net_load']
        self.dispatch_df.loc[self.dispatch_df['net_load'] > 0, 'load_not_met_by_RE'] = \
            self.dispatch_df.loc[self.dispatch_df['net_load'] > 0, 'net_load'] \
                - self.dispatch_df.loc[self.dispatch_df['net_load'] > 0, 'delta_battery_power']
        self.dispatch_df.loc[self.dispatch_df['load_not_met_by_RE'] < 0, 'load_not_met_by_RE'] = 0
        
        # Calculate ES recovery percent
        # If there is no RE, this will cause a RuntimeWarning, so set to 0 (try/except won't
        #   catch Warnings)
        if len(self.dispatch_df.loc[self.dispatch_df['excess_RE'] > 0]):
            self.storage_recovery_percent = \
                np.abs(self.dispatch_df.loc[
                    self.dispatch_df['delta_battery_power'] < 0,
                    'delta_battery_power'].sum()
                       / self.dispatch_df.loc[
                           self.dispatch_df['excess_RE'] > 0,
                           'excess_RE'].sum() * 100)
        else:
            self.storage_recovery_percent = 0
            
    def calc_timestep_dispatch(self, load, resource_order, re_power, duration,
                               is_night, is_first_hour_of_night,
                               night_duration):
        """ Calculates dispatch for an individual timestep. """
        
        # Get current battery state
        initial_soc, voltage, cycles, time_since_last_used = \
            self.system.components['battery'].get_state()
                
        # Get net load after each renewable resource is applied
        net_load = copy.deepcopy(load)
        for re_resource in resource_order:
            net_load = net_load - re_power[re_resource]

        # Check battery discharge method
        if self.dispatch_strategy == 'available_capacity':
            night_params = None
        else:        
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
            night_params = {
                'is_night': is_night,
                'night_duration': night_duration,
                'night_hours_left': self.night_hours_left,
                'soc_at_initial_hour_of_night': self.soc_at_initial_hour_of_night
            }

        # Call battery update model
        delta_power = self.system.components['battery'].update_state(
            net_load, duration, self.dispatch_strategy, night_params)

        # Check for errors
        if delta_power is None:
            print("Error message: net load: {}, initial soc: {}"
                  "".format(net_load, initial_soc))
            
        # Return initial SOC and power charged or discharged
        return initial_soc, delta_power
        
    def size_single_generator(self, generator_options, validate=True):
        """ 
        Size the generator(s) based on load not met by RE and batteries
            and several different generator models.
            
        """

        # Validate input parameters
        if validate:
            args_dict = {'generator_costs': generator_options}
            validate_all_parameters(args_dict)

        # Calculate generator usage and fuel required to meet load not met by RE and batteries
        # Total rated power (including multiple units together) based on max unmet power
        max_power = self.dispatch_df['load_not_met_by_RE'].max()
        
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
        self.dispatch_df['gen_power'] = self.dispatch_df['load_not_met_by_RE']
        grouped_load, self.fuel_used_gal = gen.calculate_fuel_consumption(
            self.dispatch_df[['gen_power']], self.duration, validate=False)
        # self.load_duration_df = calculate_load_duration(grouped_load, validate=False)
        self.dispatch_df['load_not_met'] = 0
            
    def calc_existing_generator_dispatch(self, validate=True):
        """ 
        If there are existing generators, determine how they meet the load and consume fuel.

        Once that is established, update and save how much load is still not met and
        calculate some of the metrics based on the how much load the existing generators meet.

        Note: This function was optimized for efficiency using the AI Incubator tool

        """

        # Get info from existing generators
        gen_group = self.system.components['generator']

        # Calculate dispatch and fuel consumption
        fuel_used_gal, gen_power, gens_rated_power = calc_generator_dispatch(
            gen_group, self.duration, self.dispatch_df['load_not_met_by_RE'].values, 
            validate=validate)
        
        # Determine if unmet load can be supplied by generators
        self.dispatch_df['gen_power'] = gen_power
        self.dispatch_df['load_not_met'] = self.dispatch_df['load_not_met_by_RE'] - \
            self.dispatch_df['gen_power']

        # Set simulation attributes
        self.fuel_used_gal = fuel_used_gal
        self.generator_power_kW = gens_rated_power

    def calc_resilience_metrics(self, system):
        '''
        Calculates the following resilience metrics for the system with (1) RE, batt + gen, 
        (2) RE, batt (3) gen only:
        	Number of hours before load shortfall
            Fuel use, if all load is met, otherwise -1
            Percent load not met
            Total load not met
            Peak load not met
        '''

        res_metrics = {
            'RE_batt_gen': {},
            'RE_batt': {},
            'gen': {}
        }

        # Store dispatch data in arrays for faster operation
        load_not_met = self.dispatch_df['load_not_met'].values
        gen_power = self.dispatch_df['gen_power'].values
        load = self.dispatch_df['load'].values
        gen_only_load_not_met = np.max([load - system.components['generator'].rated_power, 
                                        [0]*len(load)], axis=0)

        # Hours before shortfall
        res_metrics['RE_batt_gen']['hours_before_shortfall'] = np.where(load_not_met > 0)[0][0] \
            if len(np.where(load_not_met > 0)[0]) else len(load_not_met)
        res_metrics['RE_batt']['hours_before_shortfall'] = np.where(load_not_met + gen_power > 0)[0][0] \
            if len(np.where(load_not_met + gen_power > 0)[0]) else len(gen_power)
        res_metrics['gen']['hours_before_shortfall'] = np.where(gen_only_load_not_met > 0)[0][0] \
            if len(np.where(gen_only_load_not_met > 0)[0]) else len(load)
        
        # Fuel use, given load is met
        res_metrics['RE_batt_gen']['fuel_use'] = self.fuel_used_gal if np.all(load_not_met == 0) \
            else -1
        res_metrics['RE_batt']['fuel_use'] = 0.
        if res_metrics['gen']['hours_before_shortfall'] == len(load):
            # Calc fuel consumption for gen only system
            gen_group = self.system.components['generator']
            res_metrics['gen']['fuel_use'], _, _ = calc_generator_dispatch(
                gen_group, self.duration, load, False)
        else:
            res_metrics['gen']['fuel_use'] = -1

        # Total load not met
        res_metrics['RE_batt_gen']['total_load_not_met'] = load_not_met.sum()
        res_metrics['RE_batt']['total_load_not_met'] = (load_not_met.sum() + gen_power.sum())
        res_metrics['gen']['total_load_not_met'] = gen_only_load_not_met.sum()

        # Percent load not met
        res_metrics['RE_batt_gen']['percent_load_not_met'] = \
            res_metrics['RE_batt_gen']['total_load_not_met'] / load.sum() * 100
        res_metrics['RE_batt']['percent_load_not_met'] = \
            res_metrics['RE_batt']['total_load_not_met'] / load.sum() * 100
        res_metrics['gen']['percent_load_not_met'] = \
            res_metrics['gen']['total_load_not_met'] / load.sum() * 100

        # Peak load not met
        res_metrics['RE_batt_gen']['peak_load_not_met'] = load_not_met.max()
        res_metrics['RE_batt']['peak_load_not_met'] = (load_not_met + gen_power).max()
        res_metrics['gen']['peak_load_not_met'] = gen_only_load_not_met.max()

        return res_metrics

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

    def get_renewable_avg(self):
        non_zero_re_power = {}
        for re_resource in self.renewable_resources:
            non_zero_re_power[re_resource] = self.dispatch_df[self.dispatch_df[f'{re_resource}_power'] != 0]
            if len(non_zero_re_power[re_resource] > 0):
                non_zero_re_power[re_resource] = non_zero_re_power[re_resource][f'{re_resource}_power'].mean()
            else:
                non_zero_re_power[re_resource] = 0
        return non_zero_re_power

    def get_renewable_peak(self):
        re_peak = {}
        for re_resource in self.renewable_resources:
            re_peak[re_resource] = self.dispatch_df[f'{re_resource}_power'].max()
        return re_peak

    def get_gen_avg(self):
        non_zero_gen_power = self.dispatch_df[self.dispatch_df['gen_power'] != 0]
        if len(non_zero_gen_power) > 0:
            return non_zero_gen_power['gen_power'].mean()
        else:
            return 0

    def get_gen_peak(self):
        return self.dispatch_df['gen_power'].max()
    
    def get_gen_total(self):
        non_zero_gen_power = self.dispatch_df[self.dispatch_df['gen_power'] != 0]
        if len(non_zero_gen_power) > 0:
            return non_zero_gen_power['gen_power'].sum()
        else:
            return 0

    def get_load_not_met(self):
        return abs(self.dispatch_df['load_not_met'].sum())
    
    def get_outage_load(self):
        return self.dispatch_df['load'].sum()
    
    def get_hours_before_gen(self):
        df_temp = self.dispatch_df.copy(deep=True).reset_index()
        try:
            return df_temp[df_temp['gen_power'] > 0].index[0]
        except IndexError:
            return len(self.dispatch_df)

    def get_batt_avg(self):
        greater_than_zero_batt_power = self.dispatch_df[self.dispatch_df['delta_battery_power'] > 0]
        if len(greater_than_zero_batt_power) > 0:
            return greater_than_zero_batt_power['delta_battery_power'].mean()
        else:
            return 0

    def get_batt_peak(self):
        return self.dispatch_df['delta_battery_power'].max()

def calc_generator_dispatch(gen_group, duration, load_not_met_by_RE, validate=True):
    """ 
    Calculate how a generator is dispatched to meet load and required fuel.
    Note: This function was optimized for efficiency using the AI Incubator tool

    """

    # Validate input parameters
    if validate:
        args_dict = {'duration': duration}
        validate_all_parameters(args_dict)

    # Sort generators into prime and non-prime
    prime_gens = []
    non_prime_gens = []
    for gen in gen_group.generator_list:
        if gen.prime_generator:
            for _ in range(gen.num_units):
                prime_gens += [gen.copy_and_mod('num_units', 1)]
        else:
            for _ in range(gen.num_units):
                non_prime_gens += [gen.copy_and_mod('num_units', 1)]
    
    # Determine generator order and cumulative power
    gen_order = np.append(np.random.permutation(prime_gens), np.random.permutation(non_prime_gens))
    gens_power = [gen.rated_power for gen in gen_order]
    gen_cumulative_power = np.cumsum(gens_power)

    # Determine the number of generators that must be dispatched at each timestep to meet 
    #   load (if possible) and total power dispatched
    gens_available = np.searchsorted(gen_cumulative_power, load_not_met_by_RE, side='left')
    gen_cumulative_power = np.append(gen_cumulative_power, [gen_cumulative_power[-1]])
    total_power_of_gens_dispatched = gen_cumulative_power[gens_available]
    gen_power = np.minimum(load_not_met_by_RE, total_power_of_gens_dispatched)

    # Calculate the load met by each generator for fuel calculation
    gen_power_dict = {}
    gen_0_power_condition = gen_power < gen_cumulative_power[0]
    gen_power_dict['gen_0_power'] = np.where(gen_0_power_condition, gen_power, gens_power[0])
    for gen_indx in range(1, len(gen_order)):
        power_column = f'gen_{gen_indx}_power'
        gen_power_dict[power_column] = gens_power[gen_indx]
        condition = gen_power < gen_cumulative_power[gen_indx]
        gen_power_dict[power_column] = np.where(condition,
            gen_power - gen_cumulative_power[gen_indx-1],
            gen_power_dict[power_column])
        gen_power_dict[power_column] = np.maximum(gen_power_dict[power_column], 0)
    
    # Calculate fuel used by each generator
    if any(gen_power > 0):
        fuel_used_gal = sum(
            gen_order[gen_indx].calculate_fuel_consumption(
                pd.DataFrame(gen_power_dict[f'gen_{gen_indx}_power']), duration, validate=False)[1]
            for gen_indx in range(len(gens_power)))
    else:
        fuel_used_gal = 0.
    
    return fuel_used_gal, gen_power, gen_cumulative_power[-1]
        

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
    # Get solar and tidal profiles
    # System level
    import os
    latitude = 46.34
    longitude = -119.28
    timezone = 'US/Pacific'
    num_trials = 200.
    length_trials = 14. * 24
    spg = SolarProfileGenerator(latitude, longitude, timezone, 0, 0, 0, num_trials, length_trials,
                                validate=True)
    spg.get_power_profiles()
    spg.get_night_duration(percent_at_night=0.2, validate=True)
    start_datetimes = [profile.index[0] for profile in spg.power_profiles]
    tpg = TidalProfileGenerator(latitude, longitude, timezone, num_trials, length_trials, 
                                start_year=1998, end_year=2022)
    tpg.get_tidal_data_from_upload()
    tpg.extrapolate_tidal_epoch()
    tpg.generate_tidal_profiles(start_datetimes)
    tpg.get_power_profiles()

    # Sample generator options
    generator_options = pd.read_excel('data/MCOR Prices.xlsx', sheet_name='generator_costs',
                                      index_col=0)

    # Create a sample system
    batt = SimpleLiIonBattery(False, 50, 200, validate=True)
    pv = PV(False, 200, 0, 0, 0.360, 3, 2, validate=True, pv_tracking='fixed',
            pv_racking='ground')
    tidal = Tidal(False, 200, 200, validate=True)
    gen = Generator(True, 50, 1, {'1/4 Load (gal/hr)': 1.8, '1/2 Load (gal/hr)': 2.9,
                                  '3/4 Load (gal/hr)': 3.8, 'Full Load (gal/hr)': 4.8},
                    5000, validate=True)
    system = SimpleMicrogridSystem('pv_50_tidal_50_batt_50kW_200kWh')
    system.add_component(batt, validate=True)
    system.add_component(pv, validate=True)
    system.add_component(tidal, validate=True)
    system.add_component(gen, validate=True)

    # Create a simulation object
    load_profile = pd.read_csv(os.path.join('data', 'sample_load_profile.csv'),
                               index_col=0)['Load']
    load_profile = load_profile.iloc[4951:4951+336]
    renewable_resources = ['mre', 'pv']
    base_power_profiles = {'pv': spg.power_profiles[95],
                           'night': spg.night_profiles[95],
                           'mre': tpg.power_profiles[95]}
    load_profile.index = base_power_profiles['pv'].index

    sim = REBattGenSimulator('pv_50_tidal_50_batt_50kW_200kWh',
                             renewable_resources,
                             base_power_profiles,
                             load_profile,
                             system,
                             {'longitude': -119.28, 'latitude': 46.34,
                              'timezone': 'US/Pacific', 'altitude': 0}, 3600,
                             'night_const_batt', validate=True)

    # Run the simulation
    sim.scale_power_profiles()
    sim.calc_dispatch()
    sim.size_single_generator(generator_options, validate=True)

    # Plot dispatch
    sim.dispatch_df[['load', 'pv_power', 'mre_power', 'delta_battery_power', 'gen_power']].plot()
