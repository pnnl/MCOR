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

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import tabulate
from geopy.geocoders import Nominatim

from generate_solar_profile import SolarProfileGenerator
from microgrid_simulator import PVBattGenSimulator
from microgrid_system import PV, SimpleLiIonBattery, SimpleMicrogridSystem
from validation import validate_all_parameters, log_error, annual_load_profile_warnings


class Optimizer:
    """ Parent optimization class """

    def next_system(self):
        pass

    def run_sims(self):
        pass


class GridSearchOptimizer(Optimizer):
    """
    Simulates a grid of microgrid systems, and for each system, runs all
    solar profiles.

    Parameters
    ----------

        power_profiles: list of Pandas series' with PV power profiles
            for a 1kW system

        temp_profiles: list of Pandas dataframes with temperature
            profiles

        night_profiles: list of Pandas dataframes with info on whether
            it is night

        annual_load_profile: Pandas series with a full year-long load
            profile. It must have a DateTimeIndex with a timezone.

        location: dictionary with the following keys and value
            datatypes:
            {'longitude': float, 'latitude': float, 'timezone': string,
            'altitude': float}

        tmy_solar: TMY solar pv production time series in kwh

        pv_params: dictionary with the following keys and value
            datatypes:
            {'tilt': float, 'azimuth': float, 'module_capacity': float,
             'module_area': float (in square inches),
             'pv_racking': string (options: [roof, ground, carport]),
                  Default = ground,
             'pv_tracking': string (options: [fixed, single_axis]),
             'advanced_inputs': dict (currently does nothing)}

        battery_params: dictionary with the following keys and value
            datatypes:
            {'battery_power_to_energy': float, 'initial_soc': float,
              'one_way_battery_efficiency': float,
              'one_way_inverter_efficiency': float,
              'soc_upper_limit': float, 'soc_lower_limit': float,
              'init_soc_lower_limit': float}

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

        dispatch_strategy: determines the battery dispatch strategy.
            Options include:
                night_const_batt (constant discharge at night)
                night_dynamic_batt (updates the discharge rate based on
                    remaining available capacity)
            Default: night_dynamic_batt

        batt_sizing_method: method for sizing the battery. Options are:
                - longest_night
                - no_pv_export
                Default = longest_night

        electricity_rate: Local electricity rate in $/kWh. If it is set
            to None, the rate is determined by looking up the average
            state rate found here:
            https://www.eia.gov/electricity/state/
            Default = None

        net_metering_rate: Rate in $/kWh used for calculating exported
            PV revenue. If it is set to None, the rate is assumed to be
            the same as the electricity rate.
            Default = None

        demand_rate: Demand charge rate in $/kW used for calculating PV
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

        gen_power_percent: tuple specifying which power levels to report
            from the generator load duration curve, as a percent of the
            maximum generator size (which includes a buffer).
            Default = ()

        existing_components: Dictionary containing Component objects for
            equipment already on site in the form:
            {'pv': <PV Object>, 'generator': <Generator Object>}

        filter: lists any filtering criteria that have been applied to
            results.
            Default = None

        rank: lists any ranking criteria that have been applied to
            results.
            Default = None

        off_grid_load_profile: load profile to be used for off-grid
            operation. If this parameter is not set to None, the
            annual_load_profile is used to size the PV system and
            calculate annual revenue, and this profile is used to size
            the battery and generator and calculate resilience metrics.
            Default = None

    Methods
    ----------

        size_PV_for_netzero: Sizes PV system according to net zero and
            incrementally smaller sizes

        size_batt_by_longest_night: Sizes the battery system according
            to the longest night of the year.

        size_batt_for_no_pv_export: Sizes the battery such that no
            excess PV is exported to the grid during normal operation.

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
            configuration for multiple solar/temp profiles and
            aggregates the results

        parse_results: Parses simulation results into a dataframe

        filter_results: Filters the results_grid dataframe by specified
            constraints

        rank_results: Ranks the results_grid dataframe by specified
            ranking criteria

        print_systems_results: Returns info about each of the systems
            from the results grid

        plot_system_dispatch: Displays dispatch plots for each of the
            systems in the results grid

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

    def __init__(self, power_profiles, temp_profiles, night_profiles,
                 annual_load_profile, location, tmy_solar, pv_params,
                 battery_params, system_costs, duration=3600,
                 dispatch_strategy='night_dynamic_batt',
                 batt_sizing_method='longest_night', electricity_rate=None,
                 net_metering_rate=None, demand_rate=None,
                 net_metering_limits=None,
                 generator_buffer=1.1,
                 gen_power_percent=(), existing_components={},
                 off_grid_load_profile=None,
                 output_tmy=False, validate=True):

        self.power_profiles = power_profiles
        self.temp_profiles = temp_profiles
        self.night_profiles = night_profiles
        self.annual_load_profile = annual_load_profile
        self.location = location
        self.tmy_solar = tmy_solar
        self.pv_params = pv_params
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
        self.gen_power_percent = gen_power_percent
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
            args_dict = {'power_profiles': power_profiles,
                         'temp_profiles': temp_profiles,
                         'night_profiles': night_profiles,
                         'annual_load_profile': annual_load_profile,
                         'location': location, 'tmy_solar': tmy_solar,
                         'pv_params': pv_params,
                         'battery_params': battery_params,
                         'duration': duration,
                         'gen_power_percent': gen_power_percent,
                         'dispatch_strategy': dispatch_strategy,
                         'batt_sizing_method': batt_sizing_method,
                         'system_costs': system_costs}
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

        # De-localize timezones from profiles
        for profile in self.power_profiles:
            profile.index = profile.index.map(lambda x: x.tz_localize(None))
        for profile in self.temp_profiles:
            profile.index = profile.index.map(lambda x: x.tz_localize(None))
        for profile in self.night_profiles:
            profile.index = profile.index.map(lambda x: x.tz_localize(None))
        tmy_solar.index = tmy_solar.index.map(lambda x: x.tz_localize(None))

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

    def size_batt_for_no_pv_export(self, pv_sizes, load_profile):
        """
        Sizes the battery such that no excess PV is exported to the grid
            during normal operation.
        """

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

    def create_new_system(self, pv_size, battery_size):
        """
        Create a new SimpleMicrogridSystem to add to the system grid.
        """

        # Create PV object
        pv = PV('pv' in self.existing_components, pv_size,
                self.pv_params['tilt'], self.pv_params['azimuth'],
                self.pv_params['module_capacity'],
                self.pv_params['module_area'],
                self.pv_params['spacing_buffer'],
                self.pv_params['pv_tracking'],
                self.pv_params['pv_racking'],
                self.pv_params['advanced_inputs'], validate=False)

        # Create Battery object
        batt = SimpleLiIonBattery(
            'battery' in self.existing_components, battery_size[1],
            battery_size[0], self.battery_params['initial_soc'],
            self.battery_params['one_way_battery_efficiency'],
            self.battery_params['one_way_inverter_efficiency'],
            self.battery_params['soc_upper_limit'],
            self.battery_params['soc_lower_limit'], validate=False)

        # Determine system name
        system_name = 'pv_{:.1f}kW_batt_{:.1f}kW_{:.1f}kWh'.format(
            pv_size, battery_size[1], battery_size[0])

        # Create system object
        system = SimpleMicrogridSystem(system_name)

        # Add components to system
        system.add_component(pv, validate=False)
        system.add_component(batt, validate=False)

        return system_name, system

    def define_grid(self, include_pv=(), include_batt=(), validate=True):
        """
        Defines the grid of system sizes to consider.

        Parameters:

            include_pv: list of pv sizes to be added to the grid (in kW)

            include_batt: list of battery sizes to be added to the grid
                in the form of a tuple:
                (batt capacity, batt power) in (kWh, kW)

        """

        if validate:
            # List of initialized parameters to validate
            args_dict = {}
            if len(include_pv):
                args_dict['include_pv'] = include_pv
            if len(include_batt):
                args_dict['include_batt'] = include_batt

            if len(args_dict):
                # Validate input parameters
                validate_all_parameters(args_dict)

        # Size the pv system based on load and pv power
        pv_range = self.size_PV_for_netzero()

        # Add any sizes in include_pv
        for size in include_pv:
            pv_range += [size]

        # If there is an existing pv system, use this to inform ranges
        if 'pv' in self.existing_components:
            # Use the current PV size as the minimum
            min_cap = self.existing_components['pv'].pv_capacity

            # If it is not currently in the range, add it
            if self.existing_components['pv'].pv_capacity not in pv_range:
                pv_range += [self.existing_components['pv'].pv_capacity]
        else:
            min_cap = 0

        # Get rid of any pv sizes smaller than the minimum (e.g. from
        #   existing system)
        pv_range = [elem for elem in pv_range if elem >= min_cap]

        # Determine which method to use for sizing the battery
        if self.batt_sizing_method == 'longest_night':
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
        for size in include_batt:
            batt_range += [size]

        # Create MicrogridSystem objects for each system
        if self.batt_sizing_method == 'longest_night':
            for pv_size in pv_range:
                for battery_size in batt_range:
                    # Add system to input system dictionary
                    system_name, system = self.create_new_system(pv_size,
                                                                 battery_size)
                    self.input_system_grid[system_name] = system
        elif self.batt_sizing_method == 'no_pv_export':
            for pv_size, battery_size in zip(pv_range, batt_range):
                system_name, system = self.create_new_system(pv_size,
                                                             battery_size)
                self.input_system_grid[system_name] = system

        # Add a system with 0 PV and 0 batteries
        system_name, system = self.create_new_system(0, [0, 0])
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
        For each solar profile, extract the load profile from the
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

        # Loop over each solar profile
        for power_profile in self.power_profiles:

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
            system.load_duration = results.pop('load_duration')
            system.outputs = results

            # Calculate the annual benefits
            system.calc_annual_pv_benefits(self.tmy_solar,
                                           self.annual_load_profile,
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
                system.load_duration = result.pop('load_duration')
                system.outputs = result

                # Add to output_system_grid list
                output_list += [system]

        for system in output_list:
            # Calculate the annual benefits
            system.calc_annual_pv_benefits(self.tmy_solar,
                                           self.annual_load_profile,
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
            system.get_pv_area()

            # Add to output system grid dict
            self.output_system_grid[system.get_name()] = system

    def aggregate_by_system(self, system, validate=True):
        """
        Runs the simulation for a given system configuration for
            multiple solar/temp profiles and aggregates the results.

        """

        if validate:
            # List of initialized parameters to validate
            args_dict = {'system': system}

            # Validate input parameters
            validate_all_parameters(args_dict)

        # Create lists to hold dispatch outputs
        results_summary = {'pv_percent': [], 'batt_percent': [],
                           'gen_percent': [], 'storage_recovery_percent': [],
                           'fuel_used_gal': [], 'generator_power_kW': [],
                           'pv_avg_load': [], 'pv_peak_load': []}


        # Create empty load duration dictionaries to hold hours, kWh, and
        #   kW not met at each generator size
        if self.off_grid_load_profile is None:
            load_profile = self.annual_load_profile
        else:
            load_profile = self.off_grid_load_profile
        hours_not_met = {}
        kWh_not_met = {}
        max_kW_not_met = {}

        # For each solar/temp profile, create a simulator object
        for i, (power_profile, temp_profile, load_profile, night_profile) in \
                enumerate(zip(self.power_profiles, self.temp_profiles,
                              self.load_profiles, self.night_profiles)):
            # Reset battery state
            system.components['battery'].reset_state()

            # Set simulation name
            simulation_name = 'system_{}_profile_{}'.format(system.get_name(), i)

            # Create simulation object
            simulation = PVBattGenSimulator(
                simulation_name, power_profile, temp_profile, load_profile,
                night_profile, system, self.location, self.duration,
                self.dispatch_strategy,
                generator_buffer=self.generator_buffer, validate=False)

            # Run the simulation for each object
            simulation.scale_power_profile()
            simulation.calc_dispatch()

            # Check length to make sure there was not a merging error
            if len(simulation.dispatch_df) != len(power_profile):
                message = 'Error in dispatch calculation: dispatch ' \
                          'dataframe wrong size - check for merging error.'
                log_error(message)
                raise Exception(message)

            # Size and dispatch generator
            simulation.size_single_generator(
                self.system_costs['generator_costs'], validate=False)

            # Add the results to the lists
            results_summary['pv_percent'] += \
                [simulation.get_load_breakdown()['pv'] * 100]
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
            results_summary['pv_avg_load'] += \
                [simulation.get_pv_avg()]
            results_summary['pv_peak_load'] += \
                [simulation.get_pv_peak()]

            # Add load duration data
            grouped_load = simulation.get_load_duration_df()
            hours_not_met['sim_{}'.format(i)] = grouped_load['num_hours_above']
            kWh_not_met['sim_{}'.format(i)] = grouped_load[
                'energy_not_met_above_load_level']
            max_kW_not_met['sim_{}'.format(i)] = grouped_load[
                'max_percent_not_met']

            # Add the simulation to the system simulations dictionary
            system.add_simulation(i, simulation, validate=False)

        # Convert load duration dictionaries to dataframes
        hours_not_met = pd.concat(hours_not_met, axis=1)
        kWh_not_met = pd.concat(kWh_not_met, axis=1)
        max_kW_not_met = pd.concat(max_kW_not_met, axis=1)

        # Get mean and max for each of the load_duration metrics
        results_summary['load_duration'] = pd.DataFrame()
        results_summary['load_duration']['hours_not_met_max'] = \
            hours_not_met.fillna(0).max(axis=1)
        results_summary['load_duration']['hours_not_met_average'] = \
            hours_not_met.fillna(0).mean(axis=1)
        results_summary['load_duration']['scenarios_not_met'] = \
            hours_not_met[hours_not_met > 0].count(axis=1)
        results_summary['load_duration']['kWh_not_met_average'] = \
            kWh_not_met.fillna(0).mean(axis=1)
        results_summary['load_duration']['kWh_not_met_max'] = \
            kWh_not_met.fillna(0).max(axis=1)
        results_summary['load_duration']['max_%_kW_not_met_average'] = \
            max_kW_not_met.fillna(0).mean(axis=1)
        results_summary['load_duration']['max_%_kW_not_met_max'] = \
            max_kW_not_met.fillna(0).max(axis=1)

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
        metrics = ['pv_capacity', 'battery_capacity', 'battery_power',
                   'generator_power', 'fuel_tank_size_gal',
                   'capital_cost_usd', 'pv_capital', 'battery_capital',
                   'generator_capital', 'fuel_tank_capital', 'pv_o&m',
                   'battery_o&m', 'generator_o&m', 'pv_area_ft2',
                   'annual_benefits_usd', 'demand_benefits_usd',
                   'simple_payback_yr',
                   'pv_avg_load mean','pv_peak_load mean', 'pv_peak_load most-conservative',
                   'pv_percent mean', 'batt_percent mean', 'gen_percent mean',
                   'generator_power_kW mean', 'generator_power_kW std',
                   'generator_power_kW most-conservative',
                   'fuel_used_gal mean', 'fuel_used_gal std',
                   'fuel_used_gal most-conservative']

        # Add columns for displaying information about smaller generator sizes
        for perc in self.gen_power_percent:
            metrics += ['{}%_smaller_gen_size'.format(perc),
                        '{}%_smaller_gen_typical_fuel_gal'.format(perc),
                        '{}%_smaller_gen_conservative_fuel_gal'.format(perc),
                        '{}%_smaller_gen_cost'.format(perc),
                        '{}%_smaller_gen_scenarios_not_met'.format(perc),
                        '{}%_smaller_gen_hours_not_met_average'.format(perc),
                        '{}%_smaller_gen_hours_not_met_max'.format(perc),
                        '{}%_smaller_gen_kWh_not_met_average'.format(perc),
                        '{}%_smaller_gen_kWh_not_met_max'.format(perc),
                        '{}%_smaller_gen_max_%_kW_not_met_average'.format(perc),
                        '{}%_smaller_gen_max_%_kW_not_met_max'.format(perc)]

        # Create dataframe to hold results
        self.results_grid = pd.DataFrame(columns=metrics)

        # Iterate through each simulation and add to dataframe
        for system_name, system in self.output_system_grid.items():

            # Get system outputs
            outputs = system.outputs

            # For each metric, calculate mean and standard deviation
            results_summary = {key: {'mean': np.mean(val), 'std': np.std(val)}
                               for key, val in outputs.items()}

            # Calculate worst-case scenario metric for peak pv load
            results_summary['pv_peak_load']['most-conservative'] = \
                max(outputs['pv_peak_load'])

            # Calculate worst-case scenario metric for generator size
            results_summary['generator_power_kW']['most-conservative'] = \
                max(outputs['generator_power_kW'])

            # Calculate worst-case scenario metric for fuel use
            results_summary['fuel_used_gal']['most-conservative'] = \
                max(outputs['fuel_used_gal'])

            # Save to system object
            system.set_outputs(results_summary, validate=False)

            # Turn results into a dataframe
            system_row = pd.DataFrame.from_dict(results_summary).transpose().\
                stack()
            system_row.index = [' '.join(col).strip()
                                for col in system_row.index.values]

            # Add load duration info for smaller generator sizes
            #   specified by gen_power_percent
            for perc in self.gen_power_percent:
                # Pad the load duration curve with 0s up to the generator size
                for row in range(system.load_duration.index[-1] + 1,
                                 int(system.generator_power_kW[
                                         'most-conservative']) + 1):
                    system.load_duration.loc[row] = [0, 0, 0, 0, 0, 0, 0]

                # Find the % smallest generator size from the worst-case
                system_row = pd.concat([system_row,
                                        system.calculate_smaller_generator_metrics(
                                            perc, self.system_costs['generator_costs'],
                                            validate=False)])

            # Add static outputs (ones that don't vary between
            #   simulations)
            system_row = pd.concat([system_row,
                                    pd.Series(system.get_outputs(
                                        ['capital_cost_usd', 'pv_area_ft2',
                                         'annual_benefits_usd', 'demand_benefits_usd',
                                         'simple_payback_yr']))])

            # Get component sizes and costs
            system_row = pd.concat([system_row,
                                    pd.Series(
                                        {'pv_capacity': system.components['pv'].pv_capacity,
                                         'battery_capacity':
                                             system.components['battery'].batt_capacity,
                                         'battery_power': system.components['battery'].power,
                                         'generator_power':
                                             system.components['generator'].rated_power
                                             * system.components['generator'].num_units,
                                         'fuel_tank_size_gal':
                                             system.components['fuel_tank'].tank_size
                                             * system.components['fuel_tank'].num_units})])
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
            capital_cost_usd, pv_area_ft2, annual_benefits_usd, 
            simple_payback_yr, fuel_used_gal,
            pv_percent, gen_percent
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

    def plot_best_system(self, scenario_criteria='pv', scenario_num=None, validate=True):
        """
        Displays dispatch and load duration plots for three systems:
            (1) The top system (as ranked)
            (2) The top system (as ranked) with a battery
            (3) The system with the least fuel consumption

        Parameters
            scenario_criteria: the criteria for identifying the best and worst scenarios,
                options=['pv', 'gen'] with 'pv' showing the scenarios with the least and most
                solar irradiance and 'gen' showing the scenarios with the least and most
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
                by='fuel_used_gal most-conservative').index[0])

        # Create figures to hold plots
        for i, (system_name, system) in enumerate(systems.items()):
            # Determine which scenarios to plot
            if scenario_criteria == 'pv':
                # Find the outage periods with the max and min PV
                criteria_dict = {key: val.sum() for key, val
                          in enumerate(self.power_profiles)}
                scenario_label = 'Solar Irradiance Scenario'
            else:
                # Find the outage periods with the max and min generator runtime
                criteria_dict = {key: val for key, val
                          in enumerate(system.outputs['fuel_used_gal'])}
                scenario_label = 'Fuel Consumption Scenario'
            max_scenario = max(criteria_dict, key=criteria_dict.get)
            min_scenario = min(criteria_dict, key=criteria_dict.get)

            if scenario_num is None:
                # Create figure
                fig = plt.figure(figsize=[15, 10])

                # Plot the maximum PV outage dispatch
                ax = fig.add_subplot(121)
                system.plot_dispatch(max_scenario, ax=ax)
                ax.legend(['Load', 'PV', 'Battery', 'Generator'], loc=1,
                          fontsize=8)
                ax.set_title('{} Max {}\n {} generator {}kW'.format(
                    system_name, scenario_label, system.get_name().replace('_', ' '),
                    system.components['generator'].rated_power))

                # Plot the minimum PV outage dispatch
                ax = fig.add_subplot(122)
                system.plot_dispatch(min_scenario, ax=ax)
                ax.legend(['Load', 'PV', 'Battery', 'Generator'], loc=1,
                          fontsize=8)
                ax.set_title('{} Min {}\n {} generator {}kW'.format(
                    system_name, scenario_label, system.get_name().replace('_', ' '),
                    system.components['generator'].rated_power))
                plt.tight_layout()

            else:
                # Plot individual scenario
                fig = plt.figure(figsize=[8, 10])
                ax = fig.add_subplot(111)
                system.plot_dispatch(scenario_num, ax=ax)
                ax.legend(['Load', 'PV', 'Battery', 'Generator'], loc=1,
                          fontsize=8)
                ax.set_title('{} \n {} generator {}kW'.format(
                    system_name, system.get_name().replace('_', ' '),
                    system.components['generator'].rated_power))
                plt.tight_layout()

            # Plot load duration curve
            fig = plt.figure(figsize=[12, 10])
            ax = fig.add_subplot(111)
            system.load_duration[['hours_not_met_max',
                                  'hours_not_met_average']].plot(ax=ax)
            ax.set_xlabel('Generator Power (kW)')
            ax.set_ylabel('Number of Hours Not Met')
            ax.legend(['Maximum of Scenarios', 'Average of Scenarios'])
            ax.set_title('{} Load Duration\n {} generator {}kW'.format(
                system_name, system.get_name().replace('_', ' '),
                system.components['generator'].rated_power))

    def plot_system_dispatch(self, num_systems=None, plot_per_fig=3,
                             validate=True):
        """
        Displays dispatch plots for each of the systems in the results
            grid. If num_systems is specified, only plots up to that
            many systems.

        Includes plots for the outage periods with the most and least
            pv.

        """

        if validate:
            # List of initialized parameters to validate
            args_dict = {'plot_per_fig': plot_per_fig}
            if num_systems is not None:
                args_dict['num_systems'] = num_systems

            # Validate input parameters
            validate_all_parameters(args_dict)

        # Set the number of systems if not specified
        if num_systems is None or num_systems > len(self.results_grid):
            num_systems = len(self.results_grid)

        # Find the outage periods with the max and min PV
        sim_pv = {key: val.sum() for key, val
                  in enumerate(self.power_profiles)}
        max_pv = max(sim_pv, key=sim_pv.get)
        min_pv = min(sim_pv, key=sim_pv.get)

        # Create figures to hold plots
        # Only include plot_per_fig systems per figure
        for fig_num in range(int(np.ceil(num_systems / plot_per_fig))):

            # Get the number of subplots in this figure
            num_plots = min(num_systems - fig_num * plot_per_fig,
                            plot_per_fig)

            # Create figure 
            fig = plt.figure(figsize=[12, 10])

            # Iterate through each system 
            for plot_num, (system_name, _) in enumerate(
                    self.results_grid.iloc[
                    fig_num * plot_per_fig:fig_num * plot_per_fig + num_plots].
                    iterrows()):
                # Plot the maximum PV outage dispatch
                ax = fig.add_subplot(num_plots, 2, plot_num * 2 + 1)
                self.get_system(system_name).plot_dispatch(max_pv, ax=ax)
                ax.legend(['Load', 'PV', 'Battery', 'Generator'], loc=1,
                          fontsize=8)
                ax.set_title('{} max PV'.format(system_name))

                # Plot the minimum PV outage dispatch
                ax = fig.add_subplot(num_plots, 2, plot_num * 2 + 2)
                self.get_system(system_name).plot_dispatch(min_pv, ax=ax)
                ax.legend(['Load', 'PV', 'Battery', 'Generator'], loc=1,
                          fontsize=8)
                ax.set_title('{} min PV'.format(system_name))

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

    def format_inputs(self, spg):
        """
            Formats the inputs into dicts for writing to file.

            Parameters
            ----------
            spg: SolarProfileGenerator object

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
                '# scenarios': int(spg.num_trials),
                'scenario length': spg.length_trials,
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
        inputs['PV System'] = \
            pd.DataFrame.from_dict({
                'tilt': spg.tilt, 'azimuth': spg.azimuth,
                'spacing_buffer': self.pv_params['spacing_buffer'],
                'pv_tracking': self.pv_params['pv_tracking'],
                'pv_racking': self.pv_params['pv_racking']},
                orient='index')
        inputs['Battery System'] = pd.DataFrame.from_dict({
            key.replace('_', ' '): val for key, val in
            self.battery_params.items()}, orient='index')
        inputs['Battery System'].loc['battery sizing method'] = self.batt_sizing_method
        inputs['Existing Equipment'] = pd.DataFrame.from_dict(
            {'PV': 'None', 'Battery': 'None', 'Generator': 'None', 'FuelTank': 'None'},
            orient='index')
        if 'pv' in self.existing_components:
            inputs['Existing Equipment'].loc['PV'] = \
                '{}kW'.format(self.existing_components['pv'].pv_capacity)
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
        assumptions['PV System'] = pd.DataFrame.from_dict(
            {'albedo': spg.advanced_inputs['albedo'],
             'dc to ac ratio': spg.get_dc_to_ac(),
             'losses': spg.get_losses(), 'net-metering limits': 'None'},
            orient='index')
        if self.net_metering_limits is not None:
            if self.net_metering_limits['type'] == 'capacity_cap':
                assumptions['PV System'].loc['net-metering limits'] = \
                    '{}kW capacity cap'.format(self.net_metering_limits['value'])
            elif self.net_metering_limits['type'] == 'percent_of_load':
                assumptions['PV System'].loc['net-metering limits'] = \
                    '{}% of annual load'.format(self.net_metering_limits['value'])

        assumptions['Cost'] = pd.DataFrame.from_dict(
            {'utility rate': '${}/kWh'.format(self.electricity_rate),
             'net-metering rate': '${}/kWh'.format(self.net_metering_rate)},
            orient='index')

        assumptions['Generator'] = pd.DataFrame.from_dict(
            {'sizing buffer': '{:.0f}%'.format(
                self.generator_buffer*100-100)}, orient='index')

        return inputs, assumptions

    def save_results_to_file(self, spg, filename='simulation_results'):
        """
            Saves inputs, assumptions and results to an excel file.

            Parameters
            ----------
            spg: SolarProfileGenerator object

            filename: filename for results spreadsheet, without an
                extension

        """

        # Get dictionaries of inputs and assumptions
        inputs, assumptions = self.format_inputs(spg)

        # Parse results if not already done
        if self.results_grid is None:
            self.parse_results()

        # Re-format column and index names
        format_results = self.results_grid.copy(deep=True)

        # Re-order columns
        format_results = format_results[
            ['pv_capacity', 'battery_capacity', 'battery_power',
             'generator_power_kW mean',
             'generator_power_kW most-conservative', 'fuel_tank_size_gal',
             'pv_area_ft2', 'capital_cost_usd',
             'pv_capital', 'battery_capital', 'generator_capital',
             'fuel_tank_capital', 'pv_o&m', 'battery_o&m', 'generator_o&m',
             'annual_benefits_usd', 'demand_benefits_usd',
             'simple_payback_yr',
             'pv_avg_load mean','pv_peak_load mean','pv_peak_load most-conservative',
             'pv_percent mean', 'batt_percent mean',
             'gen_percent mean', 'fuel_used_gal mean',
             'fuel_used_gal most-conservative'] + list(
                format_results.columns[29:])]

        # Rename columns
        format_results.rename(columns=
            {'pv_capacity': 'PV Capacity',
             'battery_capacity': 'Battery Capacity',
             'battery_power': 'Battery Power',
             'generator_power_kW mean': 'Generator Power (typical scenario)',
             'generator_power_kW most-conservative':
                 'Generator Power (conservative scenario)',
             'fuel_tank_size_gal': 'Total Fuel Tank Capacity',
             'pv_area_ft2': 'PV Area', 'capital_cost_usd': 'Capital Cost',
             'pv_capital': 'PV Capital', 'battery_capital': 'Battery Capital',
             'generator_capital': 'Generator Capital',
             'fuel_tank_capital': 'Fuel Tank Capital',
             'pv_o&m': 'PV O&M', 'battery_o&m': 'Battery O&M',
             'generator_o&m': 'Generator O&M',
             'annual_benefits_usd': 'Annual PV Net-meter Revenue',
             'demand_benefits_usd': 'Annual PV Demand Savings',
             'simple_payback_yr': 'Simple Payback',
             'pv_avg_load mean' : 'Mean PV Load Met',
             'pv_peak_load mean' : 'Average Peak PV Load Met',
             'pv_peak_load most-conservative' : 'Max Peak PV Load Met',
             'pv_percent mean': 'PV Percent',
             'batt_percent mean': 'Battery Percent',
             'gen_percent mean': 'Generator Percent',
             'fuel_used_gal mean': 'Fuel used (average scenario)',
             'fuel_used_gal most-conservative': 'Fuel used (conservative scenario)'},
                              inplace=True)

        # Add units
        units = ['kW', 'kWh', 'kW', 'kW', 'kW', 'gallons', 'ft^2', '$', '$',
                 '$', '$', '$', '$/year', '$/year', '$/year', '$/year',
                 '$/year', 'years', 'kW', 'kW', 'kW', '%', '%', '%', 'gallons', 'gallons']
        for _ in self.gen_power_percent:
            units += ['kW', 'gallons', 'gallons', '$', '', '', '', 'kWh',
                      'kWh', 'kW', 'kW']
        format_results.loc['units'] = units

        format_results.columns = [col.replace('_', ' ').capitalize()
                                  for col in format_results.columns]
        format_results["temp"] = range(1, len(format_results) + 1)
        format_results.loc['units', 'temp'] = 0
        format_results = format_results.sort_values("temp").drop('temp',
                                                                 axis=1)

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
        dollars = workbook.add_format({'num_format': '$#,##0'})
        one_fp = workbook.add_format({'num_format': '0.0'})
        no_fp = workbook.add_format({'num_format': 0x01})
        perc = workbook.add_format({'num_format': 0x01})

        # Determine format for each column
        formats = [one_fp, one_fp, one_fp, one_fp, one_fp, one_fp, no_fp,
                   dollars, dollars, dollars, dollars, dollars, dollars,
                   dollars, dollars, dollars, dollars, one_fp, perc, perc,
                   perc, one_fp, one_fp]
        for _ in self.gen_power_percent:
            formats += [one_fp, one_fp, one_fp, dollars, no_fp, one_fp, no_fp,
                        no_fp, no_fp, one_fp, one_fp]

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

        # Write TMY solar sheet
        if self.output_tmy:
            sp = self.tmy_solar.reset_index()
            sp.columns = ['Datetime', 'PV Power for a 1kW array (kW)']
            sp['Datetime'] = self.annual_load_profile.index
            sp.to_excel(writer, sheet_name='TMY PV Profile', index=False)
            load_sheet = writer.sheets['TMY PV Profile']
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
        inputs['PV System'].reset_index().to_excel(
            writer, sheet_name='Input Variables', startrow=12, index=False)
        inputs_sheet.write(12, 0, 'PV System', bold_bottomborder)
        inputs_sheet.write(12, 1, '', bold_bottomborder)

        # Battery variables
        inputs['Battery System'].reset_index().to_excel(
            writer, sheet_name='Input Variables', startrow=19, index=False)
        inputs_sheet.write(19, 0, 'Battery System', bold_bottomborder)
        inputs_sheet.write(19, 1, '', bold_bottomborder)

        # Existing equipment variables
        inputs['Existing Equipment'].reset_index().to_excel(
            writer, sheet_name='Input Variables', startrow=28, index=False)
        inputs_sheet.write(28, 0, 'Existing Equipment', bold_bottomborder)
        inputs_sheet.write(28, 1, '', bold_bottomborder)
        inputs_sheet.set_column(0, 1, 30, index_format)

        # Write assumptions sheet
        assumptions['PV System'].reset_index().to_excel(
            writer, sheet_name='Assumptions', index=False)
        assumptions_sheet = writer.sheets['Assumptions']
        assumptions_sheet.write(0, 0, 'PV System', bold_bottomborder)
        assumptions_sheet.write(0, 1, '', bold_bottomborder)
        assumptions['Cost'].reset_index().to_excel(
            writer, sheet_name='Assumptions', startrow=7, index=False)
        assumptions_sheet.write(7, 0, 'Costs', bold_bottomborder)
        assumptions_sheet.write(7, 1, '', bold_bottomborder)
        assumptions['Generator'].reset_index().to_excel(
            writer, sheet_name='Assumptions', startrow=12, index=False)
        assumptions_sheet.write(12, 0, 'Generator', bold_bottomborder)
        assumptions_sheet.write(12, 1, '', bold_bottomborder)
        assumptions_sheet.set_column(0, 1, 30, index_format)

        writer.save()

    def save_timeseries_to_json(self, spg, filename='simulation_results'):
        # Parse time series outputs from dispatch dataframes
        ts_outputs = {}
        for system_name, system_obj in self.output_system_grid.items():
            ts_outputs[system_name] = []
            for sim in system_obj.simulations.values():
                df = sim.dispatch_df
                df = df[['load', 'pv_power', 'battery_soc', 'delta_battery_power', 'load_not_met']]
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
    latitude = 46.34
    longitude = -119.28
    timezone = 'US/Pacific'
    spg = SolarProfileGenerator(latitude, longitude, timezone, 265.176, 20, -180,
                                200., 14., validate=False)
    spg.get_power_profiles()
    spg.get_night_duration(percent_at_night=0.1, validate=False)
    module_params = spg.get_pv_params()
    tmy_solar = spg.tmy_power_profile

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
    battery_params = {'battery_power_to_energy': 0.25, 'initial_soc': 1,
                      'one_way_battery_efficiency': 0.9,
                      'one_way_inverter_efficiency': 0.95,
                      'soc_upper_limit': 1, 'soc_lower_limit': 0.1}

    # Create optimization object
    optim = GridSearchOptimizer(spg.power_profiles, spg.temp_profiles,
                                spg.night_profiles, annual_load_profile,
                                location, tmy_solar, pv_params, battery_params,
                                system_costs, electricity_rate=None,
                                net_metering_limits=None, generator_buffer=1.1,
                                existing_components={}, output_tmy=False,
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
