# -*- coding: utf-8 -*-
"""

Optimization class for simulating, filtering and ranking microgrid
    systems.

File contents:
    Classes:
        Optimizer
        GridSearchOptimizer (inherits from Optimizer)
        
    Standalone functions:
        get_electricity_rate

"""

import json
import multiprocessing
import os
import urllib
import copy

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import tabulate
from geopy.geocoders import Nominatim

from generate_solar_profile import SolarProfileGenerator
from generate_tidal_profile import TidalProfileGenerator
from microgrid_simulator import REBattGenSimulator
from microgrid_system import PV, Wave, Tidal, SimpleLiIonBattery, SimpleMicrogridSystem
from validation import validate_all_parameters, log_error, annual_load_profile_warnings
from constants import system_metrics, pv_metrics, battery_metrics, mre_metrics, \
    generator_metrics, re_metrics, metric_order


class Optimizer:
    """ Parent optimization class """

    def next_system(self):
        pass

    def run_sims(self):
        pass


class GridSearchOptimizer(Optimizer):
    """
    Simulates a grid of microgrid systems, and for each system, runs all
     scenario profiles.

    Parameters
    ----------

        power_profiles: dictionary of lists of Pandas series' with RE power profiles
            for a 1kW system, also includes list of Pandas dataframes with info on whether
            it is night if a PV system is included

        annual_load_profile: Pandas series with a full year-long load
            profile. It must have a DateTimeIndex with a timezone.

        location: dictionary with the following keys and value
            datatypes:
            {'longitude': float, 'latitude': float, 'timezone': string,
            'altitude': float}

        tmy_solar: TMY solar pv production time series in kwh, only included if pv considered

        tmy_mre: TMY mre production time series in kwh, only included if mre considered 

        pv_params: dictionary with the following keys and value
            datatypes:
            {'tilt': float, 'azimuth': float, 'module_capacity': float,
             'module_area': float (in square inches),
             'pv_racking': string (options: [roof, ground, carport]),
                  Default = ground,
             'pv_tracking': string (options: [fixed, single_axis]),
             'advanced_inputs': dict (currently does nothing)}

        # TODO - update once determining num_gens in sizing function
        mre_params: dictionary with the following keys and value 
            datatypes:
            {'generator_type': str, 'num_generators': int,
            'generator_capacity': float}

        battery_params: dictionary with the following keys and value
            datatypes:
            {'battery_power_to_energy': float, 'initial_soc': float,
              'one_way_battery_efficiency': float,
              'one_way_inverter_efficiency': float,
              'soc_upper_limit': float, 'soc_lower_limit': float,
              'init_soc_lower_limit': float}

        # TODO: update with wave and tidal costs
        system_costs: dictionary containing the following Pandas
            dataframes:
            pv_costs: cost of PV per Watt, with the upper limit for pv
                sizes as the columns:
                    100, 5000, 100000
                and the pv racking and tracking type as the rows:
                    ground;fixed: 2.78, 1.64, 0.83
                    ground;single_axis: 2.97, 1.75, 0.89
                    roof;fixed: 2.65, 1.56, 0.83
                    carport;fixed: 3.15, 1.86, 0.83

            om_costs: operations and maintenance costs for the following
                components:
                Generator ($/kW-yr): scalar - 102.65, exponent - 0.669
                Battery ($/kW-yr): 2.41
                PV_ground;fixed ($/kW-yr): 15
                PV_ground;single_axis ($/kW-yr): 18
                PV_roof;fixed ($/kW-yr): 17.50
                PV_carport;fixed ($/kW-yr): 12.50

            battery_costs: costs for the battery:
                Battery System ($/Wh): 0.521
                Inverter and BOS ($/W): 0.401

            fuel_tank_costs: costs for fuel tanks based on size (gal)

            generator_costs: list of possible generators, with the
                following columns:
                Power: Generator rated power
                1/4 Load (gal/hr), 1/2 Load (gal/hr), 3/4 Load (gal/hr), Full Load (gal/hr):
                    loading levels for the generator fuel curve
                Cost (USD): cost for specific generator size

        duration: Timestep duration in seconds.
            Default: 3600 (1 hour)

        # TODO: May want to develop an alternative strategy when using tidal 
        dispatch_strategy: determines the battery dispatch strategy.
            Options include:
                night_const_batt (constant discharge at night)
                night_dynamic_batt (updates the discharge rate based on
                    remaining available capacity)
                available_capacity (does not reserve any battery during specific times)
            Default: night_dynamic_batt

        # TODO: May want to develop an alternative strategy when using tidal 
        batt_sizing_method: method for sizing the battery. Options are:
                - longest_night
                - no_RE_export
                Default = longest_night

        electricity_rate: Local electricity rate in $/kWh. If it is set
            to None, the rate is determined by looking up the average
            state rate found here:
            https://www.eia.gov/electricity/state/
            Default = None

        net_metering_rate: Rate in $/kWh used for calculating exported
            RE revenue. If it is set to None, the rate is assumed to be
            the same as the electricity rate.
            Default = None

        demand_rate: Demand charge rate in $/kW used for calculating RE
            system revenue. Can be either a number or a list of 12
            numbers, one for each month. If it is set to None, no demand
            charge savings are calculated.

        net_metering_limits: Dictionary specifying local limits for the
            net-metering policy in the form of:
                {type: ['capacity_cap' or 'percent_of_load'],
                value: [<kW value> or <percentage>]}
            Default = None

        generator_buffer: Buffer between chosen generator size and
            maximum required power. E.g. a buffer of 1.1 means a
            generator has to be sized 10% larger than the maximum power.
            Default = 1.1

        existing_components: Dictionary containing Component objects for
            equipment already on site in the form:
            {'pv': <PV Object>, 'generator': <Generator Object>}

        filter: lists any filtering criteria that have been applied to
            results.
            Default = None

        rank: lists any ranking criteria that have been applied to
            results.
            Default = None

        # TODO - may need to update when non-PV RE types 
        off_grid_load_profile: load profile to be used for off-grid
            operation. If this parameter is not set to None, the
            annual_load_profile is used to size the PV system and
            calculate annual revenue, and this profile is used to size
            the battery and generator and calculate resilience metrics.
            Default = None

    Methods
    ----------

        # TODO - may need to update when non-PV RE types 
        size_PV_for_netzero: Sizes PV system according to net zero and
            incrementally smaller sizes

        size_batt_by_longest_night: Sizes the battery system according
            to the longest night of the year.

        size_batt_for_no_RE_export: Sizes the battery such that no
            excess RE is exported to the grid during normal operation.

        create_new_system: Create a new SimpleMicrogridSystem to add to
            the system grid

        define_grid: Defines the grid of system sizes to consider

        print_grid: Prints out the sizes in a grid of system
            configurations

        get_load_profiles: For each solar profile, extracts the load
            profile from annual_load_profile for the corresponding
            dates/times

        next_system: Pops and returns the next system in the
            input_system_grid list

        run_sims: Runs the simulations for all systems and profiles

        run_sims_par: Run the simulations for all systems and profiles
            using Python's multiprocessing package

        aggregate_by_system: Runs the simulation for a given system
            configuration for multiple resource profiles and
            aggregates the results

        parse_results: Parses simulation results into a dataframe

        filter_results: Filters the results_grid dataframe by specified
            constraints

        rank_results: Ranks the results_grid dataframe by specified
            ranking criteria

        print_systems_results: Returns info about each of the systems
            from the results grid

        plot_best_system: Displays dispatch and load duration plots for
            3 systems (best in terms of ranking, best with battery,
            and system with least fuel usage)

        add_system: Add a specific system to the input list

        get_system: Return a specific system based on its name

        get_input_systems: Returns the dictionary of input systems

        get_output_systems: Returns the dictionary of output systems

        format_inputs: Formats the inputs into dicts for writing to file

        save_results_to_file: Saves inputs, assumptions, and results to
            an excel file

        save_timeseries_to_json: Saves time series data to a json file

    Calculated Attributes
    ----------

        load_profiles: List of load profiles for the corresponding
            outage periods

        input_system_grid: Dictionary of MicrogridSystem objects to
            simulate

        output_system_grid: Dictionary of already simulated
            MicrogridSystem objects

        results_grid: Pandas dataframe containing output results from
            the simulations, with one row per system

        """

    def __init__(self, renewable_resources, power_profiles, annual_load_profile, location, 
                 battery_params, system_costs, duration=3600,
                 pv_params=None, mre_params=None, 
                 tmy_solar=None, tmy_mre=None,
                 dispatch_strategy='night_dynamic_batt',
                 batt_sizing_method='longest_night', electricity_rate=None,
                 net_metering_rate=None, demand_rate=None,
                 net_metering_limits=None,
                 generator_buffer=1.1,
                 existing_components={},
                 off_grid_load_profile=None,
                 output_tmy=False, validate=True):

        self.renewable_resources = renewable_resources
        self.power_profiles = power_profiles
        self.annual_load_profile = annual_load_profile
        self.location = location
        self.tmy_solar = tmy_solar
        self.tmy_mre = tmy_mre
        self.pv_params = pv_params
        self.mre_params = mre_params
        self.battery_params = battery_params
        self.system_costs = system_costs
        self.duration = duration
        self.dispatch_strategy = dispatch_strategy
        self.batt_sizing_method = batt_sizing_method
        self.electricity_rate = electricity_rate
        self.net_metering_rate = net_metering_rate
        self.demand_rate = demand_rate
        self.net_metering_limits = net_metering_limits
        self.generator_buffer = generator_buffer
        self.existing_components = existing_components
        self.off_grid_load_profile = off_grid_load_profile
        self.output_tmy = output_tmy
        self.load_profiles = []
        self.input_system_grid = {}  # Dict of MicrogridSystem objects
        self.output_system_grid = {}
        self.results_grid = None
        self.filter = None
        self.rank = None

        if validate:
            # List of initialized parameters to validate
            args_dict = {'renewable_resources': renewable_resources,
                         'power_profiles': power_profiles,
                         'annual_load_profile': annual_load_profile,
                         'location': location, 
                         'battery_params': battery_params,
                         'duration': duration,
                         'dispatch_strategy': dispatch_strategy,
                         'batt_sizing_method': batt_sizing_method,
                         'system_costs': system_costs}
            if pv_params is not None:
                args_dict['pv_params'] = pv_params
            if tmy_solar is not None:
                args_dict['tmy_solar'] = tmy_solar
            if mre_params is not None:
                args_dict['mre_params'] = mre_params
            if tmy_mre is not None:
                args_dict['tmy_mre'] = tmy_mre                
            if electricity_rate is not None:
                args_dict['electricity_rate'] = electricity_rate
            if net_metering_rate is not None:
                args_dict['net_metering_rate'] = net_metering_rate
            if demand_rate is not None:
                args_dict['demand_rate_list'] = demand_rate
            if net_metering_limits is not None:
                args_dict['net_metering_limits'] = net_metering_limits
            if len(existing_components):
                args_dict['existing_components'] = existing_components
            if off_grid_load_profile is not None:
                args_dict['off_grid_load_profile'] = off_grid_load_profile

            # Validate input parameters
            validate_all_parameters(args_dict)

            # Ensure that pv_params, mre_params, tmy_solar, and tmy_mre are not none if the
            #   corresponding resource is included in renewable_resources
            if 'pv' in renewable_resources:
                if pv_params is None or tmy_solar is None:
                    message = 'If a pv system is included in the considered resources, then both ' \
                              'pv_params and tmy_solar must be included as non-null inputs.'
                    log_error(message)
                    raise Exception(message)
            if 'mre' in renewable_resources:
                if mre_params is None or tmy_mre is None:
                    message = 'If an mre system is included in the considered resources, then both ' \
                              'mre_params and tmy_mre must be included as non-null inputs.'
                    log_error(message)
                    raise Exception(message)

        # De-localize timezones from profiles
        for re_resource, profiles in self.power_profiles.items():
            for profile in profiles:
                profile.index = profile.index.map(lambda x: x.tz_localize(None))
        if tmy_solar is not None:
            tmy_solar.index = tmy_solar.index.map(lambda x: x.tz_localize(None))
        if tmy_mre is not None:
            tmy_mre.index = tmy_mre.index.map(lambda x: x.tz_localize(None))

        # Fix annual load profile index
        self.annual_load_profile.index = pd.date_range(
            start='1/1/2017', end='1/1/2018',
            freq='{}S'.format(int(self.duration)))[:-1]
        if self.off_grid_load_profile is not None:
            self.off_grid_load_profile.index = self.annual_load_profile.index

        if validate:
            # Check for warnings
            annual_load_profile_warnings(self.annual_load_profile)
            if self.off_grid_load_profile is not None:
                annual_load_profile_warnings(self.off_grid_load_profile)

        # Get electricity rate data if a rate is not specified
        if self.electricity_rate is None:
            self.electricity_rate = get_electricity_rate(self.location,
                                                         validate=False)
            
        # If no PV is included, ensure that dispatch strategy is set to 'available_capacity'
        if 'pv' not in self.renewable_resources:
            self.dispatch_strategy = 'available_capacity'

    # TODO - update with a more sophisticated methodology that includes more options
    def size_RE_system(self):
        """
        Sizes Renewable Energy components. If only one type of RE is included, returns the 
            net-zero capacity for that resource, if multiple types are present, then returns
            the net-zero capacity for each resource as well as a system where half of the 
            load is met by each resource (based on the TMY profiles). 

        """

        # Set up dictionary with sizes
        re_sizes = {}

        # Get the total annual load
        total_annual_load = self.annual_load_profile.sum()

        # If PV is included, get the total solar energy produced and net-zero capacity
        if self.pv_params:
            total_annual_solar = self.tmy_solar.sum()
            net_zero_pv = total_annual_load / total_annual_solar
            re_sizes['nz_pv'] = {'pv': net_zero_pv, 'mre': 0}

        # If marine renewable energy is included, get the total solar energy produced and net-zero
        #   capacity
        if self.mre_params:
            total_annual_mre = self.tmy_mre.sum()
            net_zero_mre = total_annual_load / total_annual_mre
            re_sizes['nz_mre'] = {'mre': net_zero_mre, 'pv': 0}

        # If both PV and marine resources are included, get the capacities required for each
        #   resource to supply half of the net-zero load
        if self.pv_params and self.mre_params:
            # Take MRE as baseload
            total_annual_mre = self.tmy_mre.sum()
            half_net_zero_mre = 0.5 * total_annual_load / total_annual_mre

            # Find the PV capacity required to meet the remaining load
            remaining_load = self.annual_load_profile.values - self.tmy_mre.values * half_net_zero_mre
            remaining_load[remaining_load < 0] = 0
            total_annual_solar = self.tmy_solar.sum()
            half_net_zero_pv = remaining_load.sum() / total_annual_solar
            re_sizes['nz_pv_mre'] = {'pv': half_net_zero_pv, 'mre': half_net_zero_mre}

        return re_sizes
    
    # TODO - need to update for different types of RE
    def size_PV_for_netzero(self):
        """
        Sizes PV system according to net zero and incrementally smaller
            sizes.

        The maximum PV size is determined by the net-zero case
            (annual solar production = annual load) plus solar in excess
            of load times the system RTE losses, with smaller sizes
            calculated as % of net zero size:
            - Maximum size = net-zero + excess*(1-RTE)
            - Net-zero size
            - Net-zero * 50%
            - Net-zero * 25%
            - 0 PV
        """

        # Get the total annual load and solar energy produced
        total_annual_load = self.annual_load_profile.sum()
        total_annual_solar = self.tmy_solar.sum()
        net_zero = total_annual_load / total_annual_solar

        # Calculate round-trip efficiency based on battery and inverter
        #   efficiency
        system_rte = self.battery_params['one_way_battery_efficiency'] ** 2 \
            * self.battery_params['one_way_inverter_efficiency'] ** 2

        # Calculate the amount of pv energy lost through
        #   charging/discharging batteries at the net zero capacity
        losses = pd.Series(self.tmy_solar.values * net_zero -
                           self.annual_load_profile.values)
        losses.loc[losses < 0] = 0
        total_lost = losses.sum() * (1 - system_rte)

        # Maximum (net-zero) solar size is based on scaling total annual
        #   solar to equal annual load plus losses
        max_cap = (total_annual_load + total_lost) / total_annual_solar

        # Create grid based on max, min and standard intervals
        return [max_cap, net_zero, net_zero * 0.5, net_zero * 0.25]

    # TODO - Update for different types of RE
    def size_batt_by_longest_night(self, load_profile):
        """
        Sizes the battery system according to the longest night of the
            year.

        The maximum battery capacity is determined by the load of the
            highest load night, with the power determined by a fixed
            power to energy ratio. Smaller sizes are calculated as a
            fraction of the maximum size, also with a fixed power to
            energy ratio:
            - Maximum size = capacity for longest night
            - Maximum size * 75%
            - Maximum size * 50%
            - Maximum size * 25%
            - O ES
            """

        # TODO - Update for different types of RE
        # Get nighttime load based on TMY pv power
        night_df = load_profile.to_frame(name='load')
        night_df['pv_power'] = self.tmy_solar.values

        # Set daytime load to 0
        night_df.loc[night_df['pv_power'] > 0, 'load'] = 0

        # Get date (without hour) for each timestep
        night_df['day'] = night_df.index.day
        night_df['month'] = night_df.index.month

        # Add up the energy for each night (since this is done by
        #   calendar date, these aren't technically continuous nights,
        #   but this whole calculation is an approximation anyway)
        max_nightly_energy = night_df.groupby(['month', 'day'])['load']. \
            sum().max()

        # Maximum battery capacity = max nightly load / RTE
        system_rte = self.battery_params['one_way_battery_efficiency'] ** 2 \
                     * self.battery_params['one_way_inverter_efficiency'] ** 2
        max_cap = max_nightly_energy / system_rte
        max_pow = max_cap * self.battery_params['battery_power_to_energy']

        return [(max_cap, max_pow),
                (max_cap * 0.75, max_pow * 0.75),
                (max_cap * 0.5, max_pow * 0.5),
                (max_cap * 0.25, max_pow * 0.25),
                (0, 0)]

    # TODO - update with a more sophisticated methodology that includes more options
    def size_batt_for_no_RE_export(self, re_sizes, load_profile):
        """
        Sizes the battery such that no excess renewable energy is exported to the grid during
        normal operation. 

        Args:
            re_sizes (dict): dictionary of PV and MRE capacities, with key as system name and
                values as dictionaries with keys 'pv' and/or 'mre'
            load_profile (Pandas Series): _description_
        """
        
        # Calculate excess RE production for each system
        excess_re = load_profile.to_frame(name='load')
        if self.pv_params:
            excess_re['pv_base'] = self.tmy_solar.values
        else:
            excess_re['pv_base'] = 0
        if self.mre_params:
            excess_re['mre_base'] = self.tmy_mre.values
        else:
            excess_re['mre_base'] = 0
        for system_name, sizes in re_sizes.items():
            excess_re[system_name] = excess_re['pv_base'] * sizes['pv'] \
                + excess_re['mre_base'] * sizes['mre']
            excess_re[f'{system_name}_exported'] = excess_re[system_name] \
                - excess_re['load']
        excess_re[excess_re < 0] = 0

        # Calculate battery power as the maximum exported RE power
        power = excess_re.max()

        # Calculate capacity as the maximum daily exported RE energy
        excess_re['day'] = excess_re.index.date
        cap = excess_re.groupby('day').sum().max()

        return {system_name: (round(cap[f'{system_name}_exported'] *
                                    self.battery_params['one_way_inverter_efficiency'] *
                                    self.battery_params['one_way_battery_efficiency'], 2),
                              round(power[f'{system_name}_exported'] *
                                    self.battery_params['one_way_inverter_efficiency'], 2))
                for system_name in re_sizes.keys()}

    # TODO - Update for different types of RE
    def size_batt_for_no_pv_export(self, pv_sizes, load_profile):
        """
        Sizes the battery such that no excess PV is exported to the grid
            during normal operation.
        """

        # TODO - Update for different types of RE
        # Calculate excess PV production for each PV size
        excess_pv = load_profile.to_frame(name='load')
        excess_pv['pv_base'] = self.tmy_solar.values
        for size in pv_sizes:
            excess_pv[int(size)] = excess_pv['pv_base'] * size
            excess_pv['{}_exported'.format(int(size))] = excess_pv[int(size)]\
                                                         - excess_pv['load']
        excess_pv[excess_pv < 0] = 0

        # Calculate battery power as the maximum exported PV power
        power = excess_pv.max()

        # Calculate capacity as the maximum daily exported PV energy
        excess_pv['day'] = excess_pv.index.date
        cap = excess_pv.groupby('day').sum().max()

        return [(round(cap['{}_exported'.format(int(size))] *
                       self.battery_params['one_way_inverter_efficiency'] *
                       self.battery_params['one_way_battery_efficiency'], 2),
                 round(power['{}_exported'.format(int(size))] *
                       self.battery_params['one_way_inverter_efficiency'], 2))
                for size in pv_sizes]

    # TODO - Update when MRE params known 
    def create_new_system(self, pv_size, mre_size, battery_size):
        """
        Create a new SimpleMicrogridSystem to add to the system grid.
        """

        component_list = []
        system_name_list = []
        # Create PV object
        if self.pv_params:
            pv = PV('pv' in self.existing_components, pv_size,
                    self.pv_params['tilt'], self.pv_params['azimuth'],
                    self.pv_params['module_capacity'],
                    self.pv_params['module_area'],
                    self.pv_params['spacing_buffer'],
                    self.pv_params['pv_tracking'],
                    self.pv_params['pv_racking'],
                    self.pv_params['advanced_inputs'], validate=False)
            component_list += [pv]
            system_name_list += ['pv_{:.1f}kW'.format(pv_size)]
            
        # Create MRE object
        # TODO - update when we have the final list of params 
        if self.mre_params:
            if self.mre_params['generator_type'] == 'tidal':
                mre = Tidal('mre' in self.existing_components, mre_size,
                        self.mre_params['num_generators'], 
                        self.mre_params['generator_capacity'],
                        validate=False)
                component_list += [mre]
                system_name_list += ['tidal_{:.1f}kW'.format(mre_size)]
            elif self.mre_params['generator_type'] == 'wave':
                mre = Wave('mre' in self.existing_components, mre_size,
                        self.mre_params['num_generators'], 
                        self.mre_params['generator_capacity'],
                        self.mre_params['wave_inputs'],
                        validate=False)
                component_list += [mre]
                system_name_list += ['wave_{:.1f}kW'.format(mre_size)]

        # Create Battery object
        batt = SimpleLiIonBattery(
            'battery' in self.existing_components, battery_size[1],
            battery_size[0], self.battery_params['initial_soc'],
            self.battery_params['one_way_battery_efficiency'],
            self.battery_params['one_way_inverter_efficiency'],
            self.battery_params['soc_upper_limit'],
            self.battery_params['soc_lower_limit'], validate=False)
        component_list += [batt]
        system_name_list += ['batt_{:.1f}kW_{:.1f}kWh'.format(battery_size[1], battery_size[0])]

        # Determine system name
        system_name = '_'.join(system_name_list)

        # Create system object
        system = SimpleMicrogridSystem(system_name)

        # Add components to system
        for component in component_list:
            system.add_component(component, validate=False)

        return system_name, system

    # TODO - update for new algorithm 
    def define_grid(self, include_pv=(), include_batt=(), include_mre=(), validate=True):
        """
        Defines the grid of system sizes to consider.

        Parameters:

            include_pv: list of pv sizes to be added to the grid (in kW)

            include_mre: list of mre sizes to be added to the grid (in kW)

            include_batt: list of battery sizes to be added to the grid
                in the form of a tuple:
                (batt capacity, batt power) in (kWh, kW)

        """

        # TODO - Update for different types of RE
        if validate:
            # List of initialized parameters to validate
            args_dict = {}
            if len(include_pv):
                args_dict['include_pv'] = include_pv
            if len(include_batt):
                args_dict['include_batt'] = include_batt
            if len(include_mre):
                args_dict['include_mre'] = include_mre

            if len(args_dict):
                # Validate input parameters
                validate_all_parameters(args_dict)

        # If marine renewables are included, use method that can accept different types of RE
        if self.mre_params:
            re_sizes = self.size_RE_system()
        else:
            # Size the pv system based on load and pv power
            pv_range = self.size_PV_for_netzero()

            # Add any sizes in include_pv
            for size in include_pv:
                pv_range += [size]

            # If there is an existing pv system, use this to inform ranges
            if 'pv' in self.existing_components:
                # Use the current PV size as the minimum
                min_cap = self.existing_components['pv'].capacity

                # If it is not currently in the range, add it
                if self.existing_components['pv'].capacity not in pv_range:
                    pv_range += [self.existing_components['pv'].capacity]
            else:
                min_cap = 0

            # Get rid of any pv sizes smaller than the minimum (e.g. from
            #   existing system)
            pv_range = [elem for elem in pv_range if elem >= min_cap]

        # Determine which method to use for sizing the battery
        if self.mre_params:
            # Size battery to capture all excess RE generation
            batt_range = self.size_batt_for_no_RE_export(
                re_sizes, self.annual_load_profile.copy(deep=True))
            
        elif self.batt_sizing_method == 'longest_night':
            # Determine which load profile to use for sizing the battery
            if self.off_grid_load_profile is None:
                batt_sizing_load = self.annual_load_profile.copy(deep=True)
            else:
                batt_sizing_load = self.off_grid_load_profile.copy(deep=True)

            # Determine maximum battery size based on "worst" night (night
            #   with highest load)
            batt_range = self.size_batt_by_longest_night(batt_sizing_load)

        elif self.batt_sizing_method == 'no_pv_export':
            # Size battery to capture all excess PV generation
            batt_range = self.size_batt_for_no_pv_export(
                pv_range, self.annual_load_profile.copy(deep=True))

        else:
            # Add error about wrong label
            message = 'Invalid battery sizing method'
            log_error(message)
            raise Exception(message)

        # Add any sizes in include_batt
        # Note: this will not have an effect for the no pv export battery
        #   sizing methodology
        # TODO: This will throw an error currently if mre is included
        for size in include_batt:
            batt_range += [size]

        # Create MicrogridSystem objects for each system
        if self.mre_params:
            for system_name in re_sizes:
                system_name, system = self.create_new_system(re_sizes[system_name]['pv'], 
                                                             re_sizes[system_name]['mre'],
                                                             batt_range[system_name])
                self.input_system_grid[system_name] = system
        elif self.batt_sizing_method == 'longest_night':
            for pv_size in pv_range:
                for battery_size in batt_range:
                    # Add system to input system dictionary
                    system_name, system = self.create_new_system(pv_size, 0,
                                                                 battery_size)
                    self.input_system_grid[system_name] = system
        elif self.batt_sizing_method == 'no_pv_export':
            for pv_size, battery_size in zip(pv_range, batt_range):
                system_name, system = self.create_new_system(pv_size, 0,
                                                             battery_size)
                self.input_system_grid[system_name] = system

        # Add a system with 0 RE and 0 batteries
        system_name, system = self.create_new_system(0, 0, [0, 0])
        self.input_system_grid[system_name] = system

        # If there is an existing generator, add to each system
        if 'generator' in self.existing_components:
            for system in self.input_system_grid.values():
                system.add_component(self.existing_components['generator'],
                                     validate=False)

    def print_grid(self):
        """ Print out the sizes in a grid of system configurations. """

        # Iterate through both the input and output grids and print
        #   sizes for each system
        for system in np.sort(list(self.input_system_grid.keys())
                              + list(self.output_system_grid.keys())):
            print(system)

    def get_system(self, system_name):
        """ Return a specific system based on its name. """
        if system_name in self.input_system_grid:
            return self.input_system_grid[system_name]
        elif system_name in self.output_system_grid:
            return self.output_system_grid[system_name]
        else:
            print('Please enter a valid system name')

    def get_load_profiles(self):
        """
        For each RE profile, extract the load profile from the
            corresponding date/times.

        """

        # If the user specified an off-grid load profile, use this for the
        #   resiliency simulations, otherwise use the annual load profile
        if self.off_grid_load_profile is None:
            load_profile = self.annual_load_profile
        else:
            load_profile = self.off_grid_load_profile

        # Create 2-year load profile to allow for profiles with
        #   year-end overlap
        twoyear_load_profile = pd.concat([load_profile, load_profile])
        twoyear_load_profile.index = pd.date_range(
            start='1/1/2017', end='1/1/2019',
            freq='{}S'.format(int(self.duration)))[:-1]

        # Loop over each RE generation profile
        for power_profile in self.power_profiles[self.renewable_resources[0]]:

            # Get the first timestamp for the solar profile
            start_time = power_profile.index[0]

            # If it is February 29, decrease by one day
            if start_time.day == 29 and start_time.month == 2:
                start_time = start_time.replace(day=28)

            # Change the start time to year 2017 (same as load profile)
            start_time = start_time.replace(year=2017)

            # Create a datetime index 
            temp_index = pd.date_range(start=start_time,
                                       periods=len(power_profile),
                                       freq='{}S'.format(int(self.duration)))

            # Get the load profile values at the corresponding
            #   date/times
            self.load_profiles += [twoyear_load_profile.loc[temp_index]]

    def next_system(self):
        """
        For the gridsearch optimizer, this function just returns the
            next system in the grid.

        """
        try:
            return self.input_system_grid.popitem()
        except KeyError:
            message = 'There are no more systems in the input system grid.'
            log_error(message)
            raise Exception(message)

    def run_sims(self):
        """ Runs the simulations """

        # Loop over each system using next_system()
        while len(self.input_system_grid):
            system_name, system = self.next_system()
            print('Running system: {}'.format(system_name))

            # Call aggregate_by_system to run multiple solar profiles
            #   per system
            results, _ = self.aggregate_by_system(system, validate=False)

            # Save the results in the MicrogridSystem object
            system.outputs = results

            # Calculate the annual benefits
            if self.pv_params:
                system.calc_annual_RE_benefits(self.tmy_solar,
                                            self.annual_load_profile,
                                            'pv',
                                            self.duration,
                                            self.electricity_rate,
                                            self.net_metering_rate,
                                            self.demand_rate,
                                            self.batt_sizing_method,
                                            self.battery_params[
                                                'one_way_battery_efficiency'],
                                            self.battery_params[
                                                'one_way_inverter_efficiency'],
                                            self.net_metering_limits,
                                            self.existing_components,
                                            validate=False)
            if self.mre_params:
                system.calc_annual_RE_benefits(self.tmy_mre,
                                            self.annual_load_profile,
                                            'mre',
                                            self.duration,
                                            self.electricity_rate,
                                            self.net_metering_rate,
                                            self.demand_rate,
                                            self.batt_sizing_method,
                                            self.battery_params[
                                                'one_way_battery_efficiency'],
                                            self.battery_params[
                                                'one_way_inverter_efficiency'],
                                            self.net_metering_limits,
                                            self.existing_components,
                                            validate=False)

            # Calculate the required fuel tank size and number
            system.size_fuel_tank(self.system_costs['fuel_tank_costs'],
                                  self.existing_components,
                                  max(system.outputs['fuel_used_gal']))

            # Calculate system costs and payback
            system.calc_costs(self.system_costs,
                              self.existing_components, validate=False)
            system.calc_payback()

            # Calculate PV area
            if self.pv_params:
                system.get_pv_area()

            # Add to output_system_grid list
            self.output_system_grid[system_name] = system

    def run_sims_par(self):
        """ Runs the simulations using the multiprocessing toolbox """

        # Create list of systems to pass to multiprocessing pool
        input_list = []
        while len(self.input_system_grid):
            input_list += [(self.input_system_grid.popitem()[1], False)]
        output_list = []

        # Create multiprocessing pool
        with multiprocessing.Pool(os.cpu_count()) as pool:
            # Run processes
            results = pool.starmap(self.aggregate_by_system, input_list)

            for result, system in results:
                print('Running system: {}'.format(system.get_name()))

                # Save the results in the MicrogridSystem object
                system.outputs = result

                # Add to output_system_grid list
                output_list += [system]

        for system in output_list:
            # Calculate the annual benefits
            if self.pv_params:
                system.calc_annual_RE_benefits(self.tmy_solar,
                                            self.annual_load_profile,
                                            'pv',
                                            self.duration,
                                            self.electricity_rate,
                                            self.net_metering_rate,
                                            self.demand_rate,
                                            self.batt_sizing_method,
                                            self.battery_params[
                                                'one_way_battery_efficiency'],
                                            self.battery_params[
                                                'one_way_inverter_efficiency'],
                                            self.net_metering_limits,
                                            self.existing_components,
                                            validate=False)
            if self.mre_params:
                system.calc_annual_RE_benefits(self.tmy_mre,
                                            self.annual_load_profile,
                                            'mre',
                                            self.duration,
                                            self.electricity_rate,
                                            self.net_metering_rate,
                                            self.demand_rate,
                                            self.batt_sizing_method,
                                            self.battery_params[
                                                'one_way_battery_efficiency'],
                                            self.battery_params[
                                                'one_way_inverter_efficiency'],
                                            self.net_metering_limits,
                                            self.existing_components,
                                            validate=False)

            # Calculate the required fuel tank size and number
            system.size_fuel_tank(self.system_costs['fuel_tank_costs'],
                                  self.existing_components,
                                  max(system.outputs['fuel_used_gal']))

            # Calculate system costs and payback
            system.calc_costs(self.system_costs, self.existing_components,
                              validate=False)
            system.calc_payback()

            # Calculate PV area
            if self.pv_params:
                system.get_pv_area()

            # Add to output system grid dict
            self.output_system_grid[system.get_name()] = system

    def aggregate_by_system(self, system, validate=True):
        """
        Runs the simulation for a given system configuration for
            multiple resource profiles and aggregates the results.

        """

        if validate:
            # List of initialized parameters to validate
            args_dict = {'system': system}

            # Validate input parameters
            validate_all_parameters(args_dict)

        # Create lists to hold dispatch outputs
        results_summary = {'pv_percent': [], 'mre_percent': [], 
                           're_percent': [], 'batt_percent': [],
                           'gen_percent': [], 'storage_recovery_percent': [],
                           'fuel_used_gal': [], 'generator_power_kW': [],
                           'pv_avg_load': [], 'pv_peak_load': [],
                           'mre_avg_load': [], 'mre_peak_load': [],
                           'gen_avg_load': [], 'gen_peak_load': [],
                           'batt_avg_load': [], 'batt_peak_load': []}

        # For each resource profile, create a simulator object
        for i in range(len(self.load_profiles)):
            # Reset battery state
            system.components['battery'].reset_state()

            # Set simulation name
            simulation_name = 'system_{}_profile_{}'.format(system.get_name(), i)

            # Get indexed power profile for each RE resource
            power_profiles = {re_resource: re_profiles[i] 
                              for re_resource, re_profiles in self.power_profiles.items()}

            # Create simulation object
            simulation = REBattGenSimulator(
                simulation_name, self.renewable_resources,
                power_profiles, self.load_profiles[i],
                system, self.location, self.duration,
                self.dispatch_strategy,
                generator_buffer=self.generator_buffer, validate=False)

            # Run the simulation for each object
            simulation.scale_power_profiles()
            simulation.calc_dispatch()

            # Check length to make sure there was not a merging error
            if len(simulation.dispatch_df) != len(self.load_profiles[i]):
                message = 'Error in dispatch calculation: dispatch ' \
                          'dataframe wrong size - check for merging error.'
                log_error(message)
                raise Exception(message)

            # Size and dispatch generator
            simulation.size_single_generator(
                self.system_costs['generator_costs'], validate=False)

            # Add the results to the lists
            if self.pv_params:
                results_summary['pv_percent'] += \
                    [simulation.get_load_breakdown()['pv'] * 100]
                results_summary['pv_avg_load'] += \
                    [simulation.get_renewable_avg()['pv']]
                results_summary['pv_peak_load'] += \
                    [simulation.get_renewable_peak()['pv']]
            if self.mre_params:
                results_summary['mre_percent'] += \
                    [simulation.get_load_breakdown()['mre'] * 100]
                results_summary['mre_avg_load'] += \
                    [simulation.get_renewable_avg()['mre']]
                results_summary['mre_peak_load'] += \
                    [simulation.get_renewable_peak()['mre']]
            results_summary['re_percent'] += [sum(val for key, val in 
                                                  simulation.get_load_breakdown().items()
                                                  if key in ['pv', 'mre']) * 100]
            results_summary['batt_percent'] += \
                [simulation.get_load_breakdown()['battery'] * 100]
            results_summary['gen_percent'] += \
                [simulation.get_load_breakdown()['generator'] * 100]
            results_summary['storage_recovery_percent'] += \
                [simulation.get_storage_recovery_percent()]
            results_summary['fuel_used_gal'] += \
                [simulation.get_fuel_used()]
            results_summary['generator_power_kW'] += \
                [simulation.get_generator_power()]
            results_summary['gen_avg_load'] += \
                [simulation.get_gen_avg()]
            results_summary['gen_peak_load'] += \
                [simulation.get_gen_peak()]
            results_summary['batt_avg_load'] += \
                [simulation.get_batt_avg()]
            results_summary['batt_peak_load'] += \
                [simulation.get_batt_peak()]

            # Add the simulation to the system simulations dictionary
            system.add_simulation(i, simulation, validate=False)

        # Find the simulation with the largest generator and add that
        #   generator object to the system
        max_gen_sim_num = \
            np.where(results_summary['generator_power_kW']
                     == max(results_summary['generator_power_kW']))[0][0]
        max_gen_sim = system.get_simulation(max_gen_sim_num)
        system.add_component(max_gen_sim.generator_obj, validate=False)

        return results_summary, system

    def parse_results(self):
        """ Parse simulation results into a dataframe """

        # If simulations have not been run, run them now
        if not len(self.output_system_grid):
            self.run_sims()

        # Results columns
        metrics = list(system_metrics.keys()) + list(re_metrics.keys()) \
            + list(battery_metrics.keys()) + list(generator_metrics.keys())
        if self.pv_params:
            metrics += list(pv_metrics.keys())
        if self.mre_params:
            metrics += list(mre_metrics.keys())

        # Create dataframe to hold results
        self.results_grid = pd.DataFrame(columns=metrics)

        # Iterate through each simulation and add to dataframe
        for system_name, system in self.output_system_grid.items():

            # Get system outputs
            outputs = system.outputs

            # For each metric, calculate mean, max, and standard deviation
            results_summary = {key: {'mean': np.mean(val), 'std': np.std(val), 'max':np.max(val)}
                               for key, val in outputs.items() if len(val)}

            # Save to system object
            system.set_outputs(results_summary, validate=False)

            # Turn results into a dataframe
            system_row = pd.DataFrame.from_dict(results_summary).transpose().\
                stack()
            system_row.index = [' '.join(col).strip()
                                for col in system_row.index.values]

            # Add static outputs (ones that don't vary between
            #   simulations)
            static_outputs = list(system_metrics.keys())
            if self.pv_params:
                static_outputs += ['pv_area_ft2']
            if self.mre_params:
                static_outputs += ['mre_area_ft2']
            # Add up values from dictionary-based static outputs
            static_output_vals = {key: sum(val.values()) if isinstance(val, dict) else val 
                                  for key, val in system.get_outputs(static_outputs).items()}
            system_row = pd.concat([system_row, pd.Series(static_output_vals)])

            # Get component sizes and costs
            row_dict = {'battery_capacity':
                            system.components['battery'].batt_capacity,
                        'battery_power': system.components['battery'].power,
                        'generator_power':
                            system.components['generator'].rated_power
                            * system.components['generator'].num_units,
                        'fuel_tank_size_gal':
                            system.components['fuel_tank'].tank_size
                            * system.components['fuel_tank'].num_units}
            if self.pv_params:
                row_dict['pv_capacity'] = system.components['pv'].capacity
            if self.mre_params:
                row_dict['mre_capacity'] = system.components['mre'].capacity
            system_row = pd.concat([system_row, pd.Series(row_dict)])
            system_row = pd.concat([system_row, pd.Series(system.costs_usd)])

            # Add results to dataframe
            self.results_grid.loc[system_name] = system_row

    def filter_results(self, filter_constraints, validate=True):
        """
        Filters the system results by defined constraints.
                
        The filter_constraints list contains dictionaries with the
            format:
            {parameter, type, value}
        where parameter can be any of the following:
            capital_cost_usd, pv_area_ft2, mre_area_ft2, annual_benefits_usd, 
            simple_payback_yr, fuel_used_gal, re_percent, 
            pv_percent, mre_percent, gen_percent
        and type can be [max, min]
        and value is the maximum or minimum allowable value
        
        """

        if validate:
            # List of initialized parameters to validate
            args_dict = {'filter_constraints': filter_constraints}

            # Validate input parameters
            validate_all_parameters(args_dict)

        # If simulations have not been run, run them now
        if self.results_grid is None:
            self.parse_results()

        # Iterate through each constraint
        for constraint in filter_constraints:
            # Filter out any systems that do not meet the constraint
            if constraint['type'] == 'min':
                self.results_grid = self.results_grid.loc[
                    self.results_grid[constraint['parameter']] >=
                    constraint['value']]
            else:
                self.results_grid = self.results_grid.loc[
                    self.results_grid[constraint['parameter']] <=
                    constraint['value']]

        # Set filter parameter
        self.filter = filter_constraints

    def rank_results(self, ranking_criteria, validate=True):
        """
        Ranks the system results by defined ranking criteria.
        
        The ranking_criteria list is ordered from most to least
        important, and includes dictionaries with the format:
            {parameter, order_type}
        where parameter can be any of the following:
            capital_cost_usd, annual_benefits_usd, simple_payback_yr, 
            fuel_used_gal
        and order_type can be [ascending, descending]
        
        """

        if validate:
            # List of initialized parameters to validate
            args_dict = {'ranking_criteria': ranking_criteria}

            # Validate input parameters
            validate_all_parameters(args_dict)

        if not len(ranking_criteria):
            return

        # If simulations have not been run, run them now
        if self.results_grid is None:
            self.parse_results()

        # Reformat ranking criteria into lists
        ranking_params = [criteria['parameter'] for criteria
                          in ranking_criteria]
        ranking_order = [criteria['order_type'] == 'ascending' for criteria
                         in ranking_criteria]

        # Sort the results grid
        self.results_grid.sort_values(by=ranking_params,
                                      ascending=ranking_order, inplace=True)

        # Set rank parameter
        self.rank = ranking_criteria

    def print_systems_results(self, num_systems=None, validate=True):
        """
        Returns info about each of the systems from the results grid. If 
            num_systems is specified, only return up to that many
            systems.
            
        """

        if validate and num_systems is not None:
            # List of initialized parameters to validate
            args_dict = {'num_systems': num_systems}

            # Validate input parameters
            validate_all_parameters(args_dict)

        # Only return top systems if there is a limit
        if num_systems is None:
            num_systems = len(self.results_grid)

        # Iterate through (presumably) filtered and ordered results
        #   dataframe
        for system_name, row in self.results_grid.iloc[
                                :min(num_systems,
                                     len(self.results_grid))].iterrows():
            # Print system info
            print('\n')
            print(system_name)

            # Print results info
            print(tabulate.tabulate(row.to_frame(), floatfmt='.1f'))

    def plot_best_system(self, scenario_criteria='pv', scenario_num=None, stacked_graphs=True,
                         validate=True):
        """
        Displays dispatch and load duration plots for three systems:
            (1) The top system (as ranked)
            (2) The top system (as ranked) with a battery
            (3) The system with the least fuel consumption

        Parameters
            scenario_criteria: the criteria for identifying the best and worst scenarios,
                options=['pv', 'mre', 'gen'] with 'pv' showing the scenarios with the least 
                and most solar irradiance, 'mre' showing the resources with the least and most 
                mre resources and 'gen' showing the scenarios with the least and most
                generator fuel use
            scenario_num: the number of a specific scenario to plot
        """

        if validate:
            # List of initialized parameters to validate
            args_dict = {'scenario_criteria': scenario_criteria}

            # Validate input parameters
            validate_all_parameters(args_dict)

        # Identify systems to run
        systems = {}
        try:
            systems['Most Optimal System'] = \
                self.get_system(self.results_grid.index[0])
        except IndexError:
            message = 'No systems in results grid. Check that filtering ' \
                      'constraint is not too restrictive.'
            log_error(message)
            raise Exception(message)
        systems['Most Optimal System with Batteries'] = \
            self.get_system(self.results_grid[
                                self.results_grid['battery_capacity'] > 0].
                            index[0])
        systems['System with Least Fuel Consumption'] = \
            self.get_system(self.results_grid.sort_values(
                by='fuel_used_gal max').index[0])

        # Create figures to hold plots
        for i, (system_name, system) in enumerate(systems.items()):
            # Determine which scenarios to plot
            if scenario_criteria == 'pv':
                # Find the outage periods with the max and min PV
                criteria_dict = {key: val.sum() for key, val
                          in enumerate(self.power_profiles['pv'])}
                scenario_label = 'Solar Irradiance Scenario'
            elif scenario_criteria == 'mre':
                # Find the outage periods with the max and min MRE
                criteria_dict = {key: val.sum() for key, val
                          in enumerate(self.power_profiles['mre'])}
                scenario_label = 'MRE Generation Scenario'
            else:
                # Find the outage periods with the max and min generator runtime
                criteria_dict = {key: val for key, val
                          in enumerate(system.outputs['fuel_used_gal'])}
                scenario_label = 'Fuel Consumption Scenario'
            max_scenario = max(criteria_dict, key=criteria_dict.get)
            min_scenario = min(criteria_dict, key=criteria_dict.get)
            title_string, legend_list, dispatch_list = system.get_components_for_figure()

            if scenario_num is None:
                # Create figure
                fig = plt.figure(figsize=[15, 10])

                # Plot the maximum PV outage dispatch
                ax = fig.add_subplot(121)
                if stacked_graphs:
                    system.plot_stacked_dispatch(max_scenario, ax=ax)
                else:
                    system.plot_dispatch(max_scenario, ax=ax)
                ax.set_title('{} Max {}\n {} generator {}kW'.format(
                    system_name, scenario_label, system.get_name().replace('_', ' '),
                    system.components['generator'].rated_power))

                # Plot the minimum PV outage dispatch
                ax = fig.add_subplot(122)
                if stacked_graphs:
                    system.plot_stacked_dispatch(min_scenario, ax=ax)
                else:
                    system.plot_dispatch(min_scenario, ax=ax)
                ax.set_title('{} Min {}\n {} generator {}kW'.format(
                    system_name, scenario_label, system.get_name().replace('_', ' '),
                    system.components['generator'].rated_power))
                plt.tight_layout()

            else:
                # Plot individual scenario
                fig = plt.figure(figsize=[8, 10])
                ax = fig.add_subplot(111)
                if stacked_graphs:
                    system.plot_stacked_dispatch(scenario_num, ax=ax)
                else:
                    system.plot_dispatch(scenario_num, ax=ax)
                ax.set_title('{} \n {} generator {}kW'.format(
                    system_name, system.get_name().replace('_', ' '),
                    system.components['generator'].rated_power))
                plt.tight_layout()

    def add_system(self, new_system, validate=True):
        """  Add a specific system to the input list """

        if validate:
            # List of initialized parameters to validate
            args_dict = {'system': new_system}

            # Validate input parameters
            validate_all_parameters(args_dict)

        # Check that the name is unique
        system_name = new_system.get_name()
        if system_name in self.input_system_grid.keys() or system_name \
                in self.output_system_grid.keys():
            message = 'Could not add system, name not unique'
            log_error(message)
            raise Exception(message)

        # Add the new system to the input list
        self.input_system_grid[system_name] = new_system

    def get_input_systems(self):
        return self.input_system_grid

    def get_output_systems(self):
        return self.output_system_grid

    def format_inputs(self, spg, tpg):
        """
            Formats the inputs into dicts for writing to file.

            Parameters
            ----------
            spg: SolarProfileGenerator object
            tpg: TidalProfileGenerator object

            Returns
            ----------
            inputs: dictionary holding input variables
            assumptions: dictionary holding assumption variables

            """

        # Store input variables
        inputs = {}
        inputs['Location'] = pd.DataFrame.from_dict(self.location,
                                                    orient='index')
        inputs['Location'].loc['altitude'] = \
            '{}m'.format(inputs['Location'].loc['altitude'].values[0])
        inputs['Simulation Info'] = \
            pd.DataFrame.from_dict({
                '# scenarios': int(len(self.power_profiles[self.renewable_resources[0]])),
                'scenario length': len(self.power_profiles[self.renewable_resources[0]][0]),
                'scenario filters': 'None', 'scenario ranking': 'None'},
                orient='index')
        if self.filter is not None:
            inputs['Simulation Info'].loc['scenario filters'] = \
                ', '.join(['{}: {}'.format(constraint['parameter'],
                                           constraint['value'])
                           for constraint in self.filter])
        if self.rank is not None:
            inputs['Simulation Info'].loc['scenario ranking'] = \
                ', '.join([constraint['parameter'] for constraint in self.rank])
        if self.pv_params:
            inputs['PV System'] = \
                pd.DataFrame.from_dict({
                    'tilt': spg.tilt, 'azimuth': spg.azimuth,
                    'spacing_buffer': self.pv_params['spacing_buffer'],
                    'pv_tracking': self.pv_params['pv_tracking'],
                    'pv_racking': self.pv_params['pv_racking']},
                    orient='index')
        if self.mre_params and self.mre_params['generator_type'] == 'tidal':
            inputs['MRE System'] = \
                pd.DataFrame.from_dict({
                    'tidal_turbine_rated_power': tpg.advanced_inputs['tidal_turbine_rated_power'],
                    'tidal_rotor_radius': tpg.advanced_inputs['tidal_rotor_radius'],
                    'tidal_rotor_number': tpg.advanced_inputs['tidal_rotor_number']
                }, orient='index')
        inputs['Battery System'] = pd.DataFrame.from_dict({
            key.replace('_', ' '): val for key, val in
            self.battery_params.items()}, orient='index')
        inputs['Battery System'].loc['battery sizing method'] = self.batt_sizing_method
        inputs['Existing Equipment'] = pd.DataFrame.from_dict(
            {'PV': 'None', 'Battery': 'None', 'Generator': 'None', 'FuelTank': 'None'},
            orient='index')
        if 'pv' in self.existing_components:
            inputs['Existing Equipment'].loc['PV'] = \
                '{}kW'.format(self.existing_components['pv'].capacity)
        if 'mre' in self.existing_components:
            inputs['Existing Equipment'].loc['MRE'] = \
                '{}kW'.format(self.existing_components['mre'].capacity)
        if 'generator' in self.existing_components:
            inputs['Existing Equipment'].loc['Generator'] = \
                '{} units of {}kW'.format(
                self.existing_components['generator'].num_units,
                self.existing_components['generator'].rated_power)
        if 'battery' in self.existing_components:
            inputs['Existing Equipment'].loc['Battery'] = \
                '{}kW, {}kWh'.format(
                self.existing_components['battery'].power,
                self.existing_components['battery'].batt_capacity)
        if 'fuel_tank' in self.existing_components:
            inputs['Existing Equipment'].loc['FuelTank'] = \
                '{}gal'.format(self.existing_components['fuel_tank'].tank_size)

        # Store assumptions
        assumptions = {}
        if self.pv_params:
            assumptions['PV System'] = pd.DataFrame.from_dict(
                {'albedo': spg.advanced_inputs['albedo'],
                'dc to ac ratio': spg.get_dc_to_ac(),
                'losses': spg.get_losses()},
                orient='index')
        if self.mre_params and self.mre_params['generator_type'] == 'tidal':
            assumptions['MRE System'] = \
                pd.DataFrame.from_dict({
                    'maximum_cp': tpg.advanced_inputs['maximum_cp'],
                    'tidal_cut_in_velocity': tpg.advanced_inputs['tidal_cut_in_velocity'],
                    'tidal_cut_out_velocity': tpg.advanced_inputs['tidal_cut_out_velocity'],
                    'tidal_inverter_efficiency': tpg.advanced_inputs['tidal_inverter_efficiency'],
                    'tidal_turbine_losses': tpg.advanced_inputs['tidal_turbine_losses']
                }, orient='index')

        assumptions['Cost'] = pd.DataFrame.from_dict(
            {'utility rate': '${}/kWh'.format(self.electricity_rate),
             'net-metering rate': '${}/kWh'.format(self.net_metering_rate),
             'net-metering limits': 'None'},
            orient='index')
        
        if self.net_metering_limits is not None:
            if self.net_metering_limits['type'] == 'capacity_cap':
                assumptions['Cost'].loc['net-metering limits'] = \
                    '{}kW capacity cap'.format(self.net_metering_limits['value'])
            elif self.net_metering_limits['type'] == 'percent_of_load':
                assumptions['Cost'].loc['net-metering limits'] = \
                    '{}% of annual load'.format(self.net_metering_limits['value'])

        assumptions['Generator'] = pd.DataFrame.from_dict(
            {'sizing buffer': '{:.0f}%'.format(
                self.generator_buffer*100-100)}, orient='index')

        return inputs, assumptions

    def save_results_to_file(self, spg, tpg, filename='simulation_results'):
        """
            Saves inputs, assumptions and results to an excel file.

            Parameters
            ----------
            spg: SolarProfileGenerator object
            tpg: TidalProfileGenerator object

            filename: filename for results spreadsheet, without an
                extension

        """

        # Get dictionaries of inputs and assumptions
        inputs, assumptions = self.format_inputs(spg, tpg)

        # Parse results if not already done
        if self.results_grid is None:
            self.parse_results()

        # Re-format column and index names
        format_results = self.results_grid.copy(deep=True)

        # Re-order columns
        metric_order_local = copy.deepcopy(metric_order)
        if not self.pv_params:
            metric_order_local = [elem for elem in metric_order_local if 'pv' not in elem]
        if not self.mre_params:
            metric_order_local = [elem for elem in metric_order_local if 'mre' not in elem]
        format_results = format_results[metric_order_local]

        # Add units
        merged_metric_dict = {**system_metrics, **re_metrics, **pv_metrics,  **mre_metrics, **battery_metrics,
                              **generator_metrics}   
        format_results.loc['units'] = [merged_metric_dict[col_name]['units'] for col_name in format_results.columns]
        
        # Create workbook
        writer = pd.ExcelWriter('output/{}.xlsx'.format(filename),
                                engine='xlsxwriter')
        workbook = writer.book

        # Create formatting
        bold_bottomborder = workbook.add_format({'bold': True, 'bottom': True,
                                                 'align': 'center'})
        bold = workbook.add_format({'bold': True, 'border': 0})
        index_format = workbook.add_format({'align': 'left', 'border': 0,
                                            'bold': False})
        data_formats = {}
        data_formats['dollars'] = workbook.add_format({'num_format': '$#,##0'})
        data_formats['one_fp'] = workbook.add_format({'num_format': '0.0'})
        data_formats['no_fp'] = workbook.add_format({'num_format': 0x01})
        data_formats['perc'] = workbook.add_format({'num_format': 0x01})

        # Determine format for each column
        formats = [data_formats[merged_metric_dict[col_name]['format']] for col_name in format_results.columns]

        # Rename columns
        format_results.rename(columns=
            {col_name: merged_metric_dict[col_name]['display_name'] 
             for col_name in format_results.columns},
             inplace=True)
        format_results.columns = [col.replace('_', ' ').capitalize()
                                  for col in format_results.columns]
        format_results["temp"] = range(1, len(format_results) + 1)
        format_results.loc['units', 'temp'] = 0
        format_results = format_results.sort_values("temp").drop('temp', axis=1)

        # Write results sheet
        format_results.reset_index(drop=True).to_excel(writer,
                                                       sheet_name='Results',
                                                       index=False)
        results_sheet = writer.sheets['Results']

        # Format results sheet
        results_sheet.set_row(0, None, bold)
        results_sheet.set_row(1, None, bold_bottomborder)
        for i, formatting in enumerate(formats):
            results_sheet.set_column(i, i, len(format_results.columns[i]),
                                     formatting)

        # Write load profile sheet
        if self.off_grid_load_profile is None:
            lp = self.annual_load_profile.reset_index()
            lp.columns = ['Datetime', 'Load (kW)']
            lp.to_excel(writer, sheet_name='Load Profile', index=False)
            load_sheet = writer.sheets['Load Profile']
            load_sheet.set_column(0, 1, 25, None)
        else:
            lp = self.annual_load_profile.reset_index()
            lp['off-grid'] = self.off_grid_load_profile.values
            lp.columns = ['Datetime', 'Annual Load (kW)', 'Off-Grid Load (kW)']
            lp.to_excel(writer, sheet_name='Load Profile', index=False)
            load_sheet = writer.sheets['Load Profile']
            load_sheet.set_column(0, 1, 25, None)
            load_sheet.set_column(0, 2, 25, None)

        # Write TMY solar and MRE sheets
        if self.output_tmy:
            if self.pv_params:
                sp = self.tmy_solar.reset_index()
                sp.columns = ['Datetime', 'PV Power for a 1kW array (kW)']
                sp['Datetime'] = self.annual_load_profile.index
                sp.to_excel(writer, sheet_name='TMY PV Profile', index=False)
                load_sheet = writer.sheets['TMY PV Profile']
                load_sheet.set_column(0, 1, 25, None)
            if self.mre_params:
                mp = self.tmy_mre.reset_index()
                mp.columns = ['Datetime', 'MRE Power for a 1kW array (kW)']
                mp['Datetime'] = self.annual_load_profile.index
                mp.to_excel(writer, sheet_name='TMY MRE Profile', index=False)
                load_sheet = writer.sheets['TMY MRE Profile']
                load_sheet.set_column(0, 1, 25, None)

        # Write inputs sheet
        # Location variables
        inputs['Location'].reset_index().to_excel(writer,
                                                  sheet_name='Input Variables',
                                                  index=False)
        inputs_sheet = writer.sheets['Input Variables']
        inputs_sheet.write(0, 0, 'Location', bold_bottomborder)
        inputs_sheet.write(0, 1, '', bold_bottomborder)

        # Simulation variables
        inputs['Simulation Info'].reset_index().to_excel(
            writer, sheet_name='Input Variables', startrow=6, index=False)
        inputs_sheet.write(6, 0, 'Simulation Info', bold_bottomborder)
        inputs_sheet.write(6, 1, '', bold_bottomborder)

        # PV variables
        if self.pv_params:
            inputs['PV System'].reset_index().to_excel(
                writer, sheet_name='Input Variables', startrow=12, index=False)
            inputs_sheet.write(12, 0, 'PV System', bold_bottomborder)
            inputs_sheet.write(12, 1, '', bold_bottomborder)

        # MRE variables
        if self.mre_params:
            inputs['MRE System'].reset_index().to_excel(
                writer, sheet_name='Input Variables', startrow=19, index=False)
            inputs_sheet.write(19, 0, 'MRE System', bold_bottomborder)
            inputs_sheet.write(19, 1, '', bold_bottomborder)

        # Battery variables
        inputs['Battery System'].reset_index().to_excel(
            writer, sheet_name='Input Variables', startrow=24, index=False)
        inputs_sheet.write(24, 0, 'Battery System', bold_bottomborder)
        inputs_sheet.write(24, 1, '', bold_bottomborder)

        # Existing equipment variables
        inputs['Existing Equipment'].reset_index().to_excel(
            writer, sheet_name='Input Variables', startrow=33, index=False)
        inputs_sheet.write(33, 0, 'Existing Equipment', bold_bottomborder)
        inputs_sheet.write(33, 1, '', bold_bottomborder)
        inputs_sheet.set_column(0, 1, 30, index_format)

        # Write assumptions sheet
        assumptions['Cost'].reset_index().to_excel(
            writer, sheet_name='Assumptions', index=False)
        assumptions_sheet = writer.sheets['Assumptions']
        assumptions_sheet.write(0, 0, 'Costs', bold_bottomborder)
        assumptions_sheet.write(0, 1, '', bold_bottomborder)
        if self.pv_params:
            assumptions['PV System'].reset_index().to_excel(
                writer, sheet_name='Assumptions', startrow=5, index=False)
            assumptions_sheet.write(5, 0, 'PV System', bold_bottomborder)
            assumptions_sheet.write(5, 1, '', bold_bottomborder)
        if self.mre_params:
            assumptions['MRE System'].reset_index().to_excel(
                writer, sheet_name='Assumptions', startrow=10, index=False)
            assumptions_sheet = writer.sheets['Assumptions']      
            assumptions_sheet.write(10, 0, 'MRE System', bold_bottomborder)
            assumptions_sheet.write(10, 1, '', bold_bottomborder)
        assumptions['Generator'].reset_index().to_excel(
            writer, sheet_name='Assumptions', startrow=17, index=False)
        assumptions_sheet.write(17, 0, 'Generator', bold_bottomborder)
        assumptions_sheet.write(17, 1, '', bold_bottomborder)
        assumptions_sheet.set_column(0, 1, 30, index_format)

        writer.close()

        # Also output results to json
        with open('output/{}_scalar_outputs.json'.format(filename), 'w') as f:
            json.dump(self.results_grid.to_dict(orient='index'), f, indent=2)

    def save_timeseries_to_json(self, filename='simulation_results'):
        # Parse time series outputs from dispatch dataframes
        ts_outputs = {}
        for system_name, system_obj in self.output_system_grid.items():
            ts_outputs[system_name] = []
            for sim in system_obj.simulations.values():
                df = sim.dispatch_df
                output_cols = ['load', 'battery_soc', 'delta_battery_power', 'load_not_met']
                if self.pv_params:
                    output_cols += ['pv_power']
                if self.mre_params:
                    output_cols += ['mre_power']
                df = df[output_cols]
                df.rename(columns={'load_not_met': 'gen_power'}, inplace=True)
                df = df.reset_index()
                df['index'] = df['index'].apply(lambda x: x.strftime('%Y-%m-%d %X'))
                df_dict = df.to_dict(orient='list')
                ts_outputs[system_name] += [df_dict]

        with open('output/{}_timeseries.json'.format(filename), 'w') as f:
            json.dump(ts_outputs, f, indent=2)

    def plot_compare_metrics(self, x_var='simple_payback_yr',
                             y_var='capital_cost_usd', cmap='BuGn_r'):
        """
        Compares different systems by plotting metrics against each
            other. Default x and y parameters are payback and total
            capital cost, respectively.
        Note, this only works for systems with PV. 

        """

        # Parse results if not already done
        if self.results_grid is None:
            self.parse_results()

        # Check that vars exist in results grid
        if x_var not in self.results_grid.columns or y_var not in self.results_grid.columns:
            return('ERROR: {} or {} are not valid output metrics, please '
                   'choose one of the following options: {}'.format(
                    x_var, y_var, ', '.join(self.results_grid.columns.values)))

        # Make pv and batt sizes categorical, so exact sizes are shown
        #   in the legend
        results_mod = self.results_grid.copy()
        results_mod['pv_capacity'] = results_mod['pv_capacity'].apply(
            lambda x: str(int(x))+'kW')
        pv_order = [str(elem2)+'kW' for elem2 in np.flipud(
            np.sort([int(elem[:-2]) for elem in results_mod['pv_capacity'].unique()]))]
        results_mod['battery_power'] = results_mod['battery_power'].apply(
            lambda x: str(int(x))+'kW')
        batt_order = [str(elem2)+'kW' for elem2 in np.flipud(
            np.sort([int(elem[:-2]) for elem in results_mod['battery_power'].unique()]))]

        # Create plot
        fig, ax = plt.subplots(figsize=[8, 6], subplot_kw={'position': (0.1, 0.1, 0.6, 0.75)})
        sns.scatterplot(data=results_mod, x=x_var, y=y_var,
                        size='battery_power', hue='pv_capacity', palette=cmap,
                        size_order=batt_order, hue_order=pv_order, ax=ax,
                        edgecolor='#0c2c84')

        # Adjust legend params
        plt.gca().legend(loc=7, fontsize=10, bbox_to_anchor=(1.4, 0.5), scatterpoints=1)

        # Add title
        plt.title('Comparison of {} and \n{} across system sizes'.
                  format(x_var, y_var), position=(0.5, 1.05))
        plt.tight_layout()

    def plot_compare_sizes(self, var='simple_payback_yr', cmap='BuGn_r'):
        """
        Compares different systems by plotting sizes against each other
            with color determined by an output metric.
        Default metric is payback.
        Note, this only works for systems with PV. 

        """

        # Parse results if not already done
        if self.results_grid is None:
            self.parse_results()

        # Check that var exists in results grid
        if var not in self.results_grid.columns:
            return ('ERROR: {} is not a valid output metric, please choose '
                    'one of the following options: {}'
                    .format(var, ', '.join(self.results_grid.columns.values)))

        # Convert to heatmap data structure
        fig, ax = plt.subplots(figsize=[8, 6], subplot_kw={'position': (0.1, 0.1, 0.6, 0.75)})
        results_heatmap = self.results_grid.pivot(
            index='pv_capacity', columns='battery_power', values=var)
        results_heatmap.sort_index(ascending=False, inplace=True)
        sns.heatmap(results_heatmap, cmap=cmap, annot=True, fmt='.1f',
                    ax=ax, cbar_kws={'label': var},
                    xticklabels=results_heatmap.columns.values.round(1),
                    yticklabels=results_heatmap.index.values.round(1))
        plt.title('Comparison of {} across system sizes'.format(var))


def get_electricity_rate(location, validate=True):
    """
    Get the state-averaged electricity rate based on a location with the
        format {'latitude': <latitude>, 'longitude': <longitude>}
        
    """

    if validate:
        # List of initialized parameters to validate
        args_dict = {'location': location}

        # Validate input parameters
        validate_all_parameters(args_dict)

    # Pull the state rates from EIA
    try:
        rates = pd.read_html('https://www.eia.gov/electricity/state/', flavor='html5lib')[0]
        assert len({'Name', 'Average retail price (cents/kWh)',
                    'Net summer capacity (MW)', 'Net generation (MWh)',
                    'Total retail sales (MWh)'} - set(rates.columns)) == 0
    except (ValueError, AssertionError, urllib.error.URLError):
        message = 'Warning: Could not load rates from EIA, using saved ' \
                  'rates August 2018.'
        log_error(message)
        print(message)
        rates = pd.read_csv(os.path.join('data', 'electricity_rates_08.2018.csv'))

    try:
        # Get the state via reverse geocoding
        locator = Nominatim(user_agent='mcor')
        loc = locator.reverse(f"{location['latitude']}, {location['longitude']}")
        state = loc.address.split(', ')[-3]

        # Return the electricity rate for that state in $/kWh
        return rates.set_index('Name').loc[
                   state, 'Average retail price (cents/kWh)'] / 100

    except Exception as e:
        # If there is an error, return the median electricity rate
        print('Reverse Geocoding fail, using median U.S. electricity rate')
        return rates['Average retail price (cents/kWh)'].median() / 100

if __name__ == "__main__":
    # Used for testing
    multiprocessing.freeze_support()

    # Load in costs
    system_costs = pd.read_excel('data/MCOR Prices.xlsx', sheet_name=None, index_col=0)

    # Set up solar profiles
    num_trials = 2.
    length_trials = 14. * 24
    latitude = 46.34
    longitude = -119.28
    timezone = 'US/Pacific'
    spg = SolarProfileGenerator(latitude, longitude, timezone, 265.176, 20, -180,
                                num_trials, length_trials, validate=True)
    spg.get_power_profiles()
    spg.get_night_duration(percent_at_night=0.1, validate=True)
    module_params = spg.get_pv_params()
    tmy_solar = spg.tmy_power_profile
    start_datetimes = [profile.index[0] for profile in spg.power_profiles]

    # Set up tidal profiles
    tpg = TidalProfileGenerator(latitude, longitude, timezone, num_trials, length_trials,
        start_year=1998, end_year=2022)
    tpg.get_tidal_data_from_upload()
    tpg.extrapolate_tidal_epoch()
    tpg.generate_tidal_profiles(start_datetimes)
    tpg.get_power_profiles()
    tmy_mre = tpg.tmy_tidal

    # Set up load profile
    annual_load_profile = pd.read_csv(os.path.join('data', 'sample_load_profile.csv'),
                                      index_col=0)['Load']

    # Set up parameter dictionaries
    location = {'longitude': longitude, 'latitude': latitude,
                'timezone': timezone, 'altitude': 0}
    pv_params = {'tilt': 20, 'azimuth': -180,
                 'module_capacity': module_params['capacity'],
                 'module_area': module_params['area_in2'],
                 'spacing_buffer': 2,
                 'pv_racking': 'ground',
                 'pv_tracking': 'fixed',
                 'advanced_inputs': {}}
    mre_params = {'generator_type': 'tidal',
                  'num_generators': 1,
                  'generator_capacity': 100}
    battery_params = {'battery_power_to_energy': 0.25, 'initial_soc': 1,
                      'one_way_battery_efficiency': 0.9,
                      'one_way_inverter_efficiency': 0.95,
                      'soc_upper_limit': 1, 'soc_lower_limit': 0.1}

    # Create optimization object
    renewable_resources = ['pv', 'mre']
    power_profiles = {'pv': spg.power_profiles,
                      'mre': tpg.power_profiles,
                      'night': spg.night_profiles}

    optim = GridSearchOptimizer(renewable_resources, power_profiles, annual_load_profile,
                                location, battery_params, system_costs, 
                                tmy_solar=tmy_solar, pv_params=pv_params, mre_params=mre_params,
                                tmy_mre=tmy_mre, dispatch_strategy='available_capacity',
                                electricity_rate=None, net_metering_limits=None, 
                                generator_buffer=1.1, existing_components={}, output_tmy=False,
                                validate=True)

    # Create a grid of systems
    optim.define_grid()

    # Get load profiles for the corresponding solar profile periods
    optim.get_load_profiles()
    optim.run_sims()

    # Filter and rank
    optim.parse_results()
    ranking_criteria = [{'parameter': 'simple_payback_yr',
                         'order_type': 'ascending'}]
    optim.rank_results(ranking_criteria)
