# -*- coding: utf-8 -*-
"""

Class structure for microgrid system and its components.

File contents:
    Classes:
        Component
        PV (inherits from Component)
        MRE (inherits from Component)
        Tidal (inherits from MRE)
        Wave (inherits from MRE)
        Battery (inherits from Component)
        SimpleLiIonBattery (inherits from Battery)
        Generator (inherits from Component)
        GeneratorGroup (inherits from Component)
        FuelTank (inherits from Component)
        MicrogridSystem
        SimpleMicrogridSystem (inherits from MicrogridSystem)

"""

from copy import deepcopy
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import warnings

from validation import validate_all_parameters, log_error


# Microgrid components
class Component:
    """ 
    Component class: solar module, battery, generator, fuel_tank
    
    Parameters
    ----------
    
        category: type of equipment, choices are:
            ['pv', 'battery', 'generator', 'fuel_tank']
            
        existing: whether or not the component already exists on site
        
    Methods
    ----------
        
        get_category: returns the category

        calc_capital_cost(): calculate capital costs

        calc_om_cost(): calculate om costs


    """
    category = None
    existing = False
    capital_cost = None
    om_cost = None
    
    def __repr__(self):
        pass
    
    def get_category(self):
        return self.category

    def calc_capital_cost(self, *args):
        pass

    def calc_om_cost(self, *args):
        pass

    def copy_and_mod(self, param_name, param_value):
        self_copy = deepcopy(self)
        self_copy.__setattr__(param_name, param_value)
        return self_copy


class PV(Component):
    """ 
    PV system 
    
    Parameters
    ----------

        category: equipment type, set to 'pv'
        
        existing: whether or not the component already exists on site

        capacity: total_capacity in kW
        
        tilt: panel tilt in degrees
        
        azimuth: panel azimuth in degrees
        
        module_area: individual module_area in in^2
        
        module_capacity: individual module_capacity in kW

        spacing_buffer: buffer to account for panel spacing in area calculation

        pv_racking: mount type of panels
            (options: [roof, ground, carport])

        pv_tracking: type of tracking
            (options: [fixed, single_axis])
        
        advanced_inputs: panel advanced inputs (currently does nothing)
        
    Methods
    ----------
    
        calc_area: calculates the total area of the array in ft^2
        
        get_capacity: returns the total array capacity in kW

        calc_capital_cost: calculate capital costs

        calc_om_cost: calculate om costs
    
    """
    
    def __init__(self, existing, pv_capacity, tilt, azimuth, module_capacity, module_area,
                 spacing_buffer, pv_tracking, pv_racking, advanced_inputs={}, validate=True):
        # Assign parameters
        self.category = 'pv'
        self.existing = existing
        self.capacity = pv_capacity  # in kW
        self.tilt = tilt
        self.azimuth = azimuth
        self.module_area = module_area  # in in^2
        self.module_capacity = module_capacity  # in kW
        self.spacing_buffer = spacing_buffer
        self.pv_tracking = pv_tracking
        self.pv_racking = pv_racking
        self.advanced_inputs = advanced_inputs


        if validate:
            # List of initialized parameters to validate
            args_dict = {'existing': existing, 'pv_capacity': pv_capacity,
                         'tilt': tilt, 'azimuth': azimuth,
                         'module_capacity': module_capacity,
                         'module_area_in2': module_area,
                         'spacing_buffer': spacing_buffer,
                         'pv_tracking': pv_tracking, 'pv_racking': pv_racking}

            # Validate input parameters
            validate_all_parameters(args_dict)

    def __repr__(self):
        return 'PV: Capacity: {:.1f}kW'.format(self.capacity)
                
    def calc_area(self):
        """ Calculates the area of the array """
        
        # Calculate total array area in ft^2
        return self.capacity / self.module_capacity * \
            self.module_area / 144 * self.spacing_buffer

    def get_capacity(self):
        return self.capacity

    def calc_capital_cost(self, cost_df, existing_components):
        """ Calculates cost of PV array """

        # Adjust total pv_capacity to consider existing pv
        if 'pv' in existing_components.keys():
            adjusted_pv_capacity = max(
                self.capacity - existing_components['pv'].capacity, 0)
        else:
            adjusted_pv_capacity = self.capacity

        # Set cost to largest size in case pv exceeds max size on cost doc
        pv_cost_per_w = cost_df.loc[(self.pv_racking + ';' + self.pv_tracking)].iloc[-1]
        for col in cost_df:
            if adjusted_pv_capacity <= col:
                pv_cost_per_w = cost_df.loc[(self.pv_racking + ';' + self.pv_tracking), col]
                break

        # If utility-scale roof or carport-mount, print user warning
        if self.capacity > 5000 and self.pv_racking in ['roof', 'carport']:
            print(f'WARNING: A pv capacity of {self.capacity}kW does not make sense with '
                  f'{self.pv_racking} racking. It is recommended that you select '
                  f'ground-mounted racking.')

        return adjusted_pv_capacity * 1000 * pv_cost_per_w

    def calc_om_cost(self, cost_df, existing_components):
        """ Calculates O&M costs of a PV array """

        # Adjust total pv_capacity to consider existing pv
        if 'pv' in existing_components.keys():
            adjusted_pv_capacity = max(
                self.capacity - existing_components['pv'].capacity, 0)
        else:
            adjusted_pv_capacity = self.capacity

        pv_om_cost_per_kw = cost_df['PV_{};{}'.format(
            self.pv_racking, self.pv_tracking)].values[1]
        return adjusted_pv_capacity * pv_om_cost_per_kw


class MRE(Component):
    """
    Marine Renewable Energy system

    Parameters
    ----------

        category: equipment type, set to 'mre'

        existing: whether or not the component already exists on site

        num_generators: the number of marine energy generators/turbines

        mre_capacity: total_capacity in kW

        generator_type: type of marine energy generator: 'tidal' or 'wave'

        generator_capacity: the capacity per generator/turbine, in kW

    Methods
    ----------

        get_capacity: returns the total array capacity in kW

    """

    def __init__(self, existing, mre_capacity, generator_type,
                 generator_capacity):
        # Assign parameters
        self.category = 'mre'
        self.existing = existing
        self.capacity = mre_capacity  # in kW
        self.generator_type = generator_type
        self.generator_capacity = generator_capacity # in kW
        self.num_generators = int(mre_capacity / generator_capacity)

    def get_capacity(self):
        return self.capacity


class Tidal(MRE):
    """
    Tidal Generator Array

    Parameters
    ----------

        all parameters from parent class MRE

        generator_type: type of marine energy generator: 'tidal'

    Methods
    ----------

        all methods from parent class MRE

    """

    def __init__(self, existing, mre_capacity, generator_capacity, device_name, validate=True):
        super().__init__(existing, mre_capacity, 'tidal', generator_capacity)

        if validate:
            # List of initialized parameters to validate
            args_dict = {'existing': existing, 'mre_capacity': mre_capacity,
                         'tidal_turbine_rated_power': generator_capacity}

            # Validate input parameters
            validate_all_parameters(args_dict)

        self.device_name = device_name

    def __repr__(self):
        return 'Tidal: Capacity: {:.1f}kW'.format(self.capacity)

    def calc_capital_cost(self, cost_df, existing_components):
        """ Calculates cost of Tidal array """

        # Adjust total mre_capacity to consider existing mre
        # TODO - update this to consider # of turbines instead of capacity
        if 'mre' in existing_components.keys() and \
                existing_components['mre'].generator_type == 'tidal':
            adjusted_mre_num = max(
                self.num_generators - existing_components['mre'].num_generators, 0)
        else:
            adjusted_mre_num = self.num_generators

        # Set costs using loc to find the correct device
        tidal_cost_per_turbine = cost_df.loc[self.device_name, 'Cost (USD)']

        # Set costs
        # tidal_cost_per_turbine = cost_df.reset_index().iloc[0]['Cost (USD)']

        return adjusted_mre_num * tidal_cost_per_turbine

    def calc_om_cost(self, cost_df, existing_components):
        """ Calculates O&M costs of a Tidal array """

        # Adjust total mre_capacity to consider existing mre
        if 'mre' in existing_components.keys() and \
                existing_components['mre'].generator_type == 'tidal':
            adjusted_mre_cap = max(
                self.capacity - existing_components['mre'].capacity, 0)
        else:
            adjusted_mre_cap = self.capacity

        # Set O&M cost
        tidal_om_cost_per_kw = cost_df['Tidal'].values[1]
        return adjusted_mre_cap * tidal_om_cost_per_kw


class Wave(MRE):
    """
    Wave Generator Array

    Parameters
    ----------

        all parameters from parent class MRE

        generator_type: type of marine energy generator: 'wave'

        generator_capacity: the capacity per generator, in kW

        wave_inputs: TBD

    Methods
    ----------

        all methods from parent class MRE

    """

    def __init__(self, existing, mre_capacity, generator_capacity,
                 wave_inputs, validate=True):
        super().__init__(existing, mre_capacity, 'wave',
                         generator_capacity)
        # Assign parameters
        self.wave_inputs = wave_inputs
        # TODO

        if validate:
            # List of initialized parameters to validate
            args_dict = {'existing': existing, 'mre_capacity': mre_capacity,
                         'generator_capacity': generator_capacity,
                         'wave_inputs': wave_inputs}

            # Validate input parameters
            validate_all_parameters(args_dict)

    def __repr__(self):
        return 'Wave: Capacity: {:.1f}kW'.format(self.capacity)

    def calc_capital_cost(self, cost_df, existing_components):
        """ Calculates cost of Wave array """

        # Adjust total mre_capacity to consider existing mre
        if 'mre' in existing_components.keys() and \
                existing_components['mre'].generator_type == 'wave':
            adjusted_mre_capacity = max(
                self.capacity - existing_components['mre'].capacity, 0)
        else:
            adjusted_mre_capacity = self.capacity

        # Set costs
        # TODO

        return adjusted_mre_capacity

    def calc_om_cost(self, cost_df, existing_components):
        """ Calculates O&M costs of a Wave array """

        # Adjust total mre_capacity to consider existing pv
        if 'mre' in existing_components.keys() and \
                existing_components['mre'].generator_type == 'wave':
            adjusted_mre_capacity = max(
                self.capacity - existing_components['mre'].capacity, 0)
        else:
            adjusted_mre_capacity = self.capacity

        # Set O&M cost
        # TODO

        return adjusted_mre_capacity


class Battery(Component):
    """
    General battery class
    
    Parameters
    ----------

        category: equipment type, set to 'battery'
        
        existing: whether or not the component already exists on site
                
        power: battery power in kW
            
        batt_capacity: battery capacity in kW
            
        initial_soc: initial state of charge
            default = 1
            
        one_way_battery_efficiency: one way battery efficiency
            default = 0.9
            
        one_way_inverter_efficiency: one way inverter efficiency
            default = 0.95
            
        soc_upper_limit: state of charge upper limit
            default = 1
            
        soc_lower_limit: state of charge lower limit
            default = 0.1
            
    Methods
    ----------
    
        reset_state: resets the state of the battery to initial conditions

        update_state: only defined in child classes
        
        get_state: returns current state of charge, voltage, number of cycles, and time since
            last used
            
        get_capacity_power: returns the battery capacity and power

        calc_capital_cost: calculate capital costs

        calc_om_cost: calculate om costs
        
    Calculated Attributes
    ----------

        soc: battery state of charge, initially set as initial_soc
        
        cycles: number of used cycles, initially set at 0, currently unused
        
        voltage: battery voltage, currently unused
        
        time_since_last_used: amount of time elapsed since the battery was last charged or
            discharged, currently unused

    """
    
    def __init__(self, existing, power, batt_capacity, initial_soc=1,
                 one_way_battery_efficiency=0.9,
                 one_way_inverter_efficiency=0.95, soc_upper_limit=1,
                 soc_lower_limit=0.1, validate=True):
        # Battery parameters
        self.category = 'battery'
        self.existing = existing
        self.power = power  # kW
        self.batt_capacity = batt_capacity  # kWh
        self.initial_soc = initial_soc
        self.one_way_battery_efficiency = one_way_battery_efficiency
        self.one_way_inverter_efficiency = one_way_inverter_efficiency
        self.soc_upper_limit = soc_upper_limit
        self.soc_lower_limit = soc_lower_limit
        
        # Initialize battery state
        self.soc = deepcopy(self.initial_soc)
        self.cycles = 0
        self.voltage = 0
        self.time_since_last_used = 0

        if validate:
            # List of initialized parameters to validate
            args_dict = {
                'existing': existing, 'power': power,
                'batt_capacity': batt_capacity, 'initial_soc': initial_soc,
                'one_way_battery_efficiency': one_way_battery_efficiency,
                'one_way_inverter_efficiency': one_way_inverter_efficiency,
                'soc_upper_limit': soc_upper_limit,
                'soc_lower_limit': soc_lower_limit
            }

            # Validate input parameters
            validate_all_parameters(args_dict)

    def __repr__(self):
        return 'Battery: Capacity: {:.1f}kWh, Power: {:.1f}kW'.format(
            self.batt_capacity, self.power)
        
    def reset_state(self):
        """ Resets the state to initial conditions. This is useful since many simulations
            share one system (and therefore battery) object.
        """
        # Initialize battery state
        self.soc = deepcopy(self.initial_soc)
        self.cycles = 0
        self.voltage = 0
        self.time_since_last_used = 0
    
    def update_state(self, *args):
        pass
    
    def get_state(self):
        return self.soc, self.voltage, self.cycles, self.time_since_last_used
        
    def get_capacity_power(self):
        return self.batt_capacity, self.power

    def calc_capital_cost(self, cost_df, existing_components):
        """ Calculate battery system capital costs """

        # Adjust total capacity and power to consider existing batteries
        if 'batt' in existing_components.keys():
            adjusted_batt_capacity = max(
                self.batt_capacity - existing_components['batt'].batt_capacity, 0)
            adjusted_batt_power = max(
                self.power - existing_components['batt'].power, 0)
        else:
            adjusted_batt_capacity = self.batt_capacity
            adjusted_batt_power = self.power

        # Parse battery and inverter capital costs from costs dataframe
        batt_cost_per_wh = cost_df['Battery System'].values[1]
        inverter_cost_per_w = cost_df['Inverter'].values[1]

        return adjusted_batt_capacity * 1000 * batt_cost_per_wh \
            + adjusted_batt_power * 1000 * inverter_cost_per_w

    def calc_om_cost(self, cost_df, existing_components):
        """ Calculate battery system O&M costs """

        # Adjust total capacity and power to consider existing batteries
        if 'batt' in existing_components.keys():
            adjusted_batt_capacity = max(
                self.batt_capacity - existing_components['batt'].batt_capacity, 0)
        else:
            adjusted_batt_capacity = self.batt_capacity

        # Parse battery and inverter o&m costs from costs dataframe
        batt_om_cost_per_kwh = cost_df['Battery'].values[1]
        return adjusted_batt_capacity * batt_om_cost_per_kwh


class SimpleLiIonBattery(Battery):
    """ 
    Models a simple lithium ion battery, where the battery discharges according to three
        possible methods: 
            (1) at a constant rate every night, based on the available capacity
            (2) at a dynamic rate every night based on remaining available
                    capacity
            (3) up to the available capacity at all hours, day or night
        
    Parameters
    ----------
    
        all parameters from parent class Battery
        
    Methods
    ----------
    
        all methods from parent class Battery
        
        update_state: charges or discharges the battery for a single time step based on net
            load
                
    Calculated Attributes
    ----------

        charge_eff: battery charging efficiency
        
        discharge_eff: battery discharging efficiency

    """
    
    def __init__(self, existing, power, batt_capacity, initial_soc=1,
                 one_way_battery_efficiency=0.9,
                 one_way_inverter_efficiency=0.95, soc_upper_limit=1,
                 soc_lower_limit=0.1, validate=True):
        super().__init__(existing, power, batt_capacity, initial_soc,
                         one_way_battery_efficiency,
                         one_way_inverter_efficiency, soc_upper_limit,
                         soc_lower_limit, validate)
    
        # Set efficiencies
        self.charge_eff = self.one_way_battery_efficiency \
            * self.one_way_inverter_efficiency
        self.discharge_eff = self.one_way_battery_efficiency \
            * self.one_way_inverter_efficiency

    def update_state(self, net_load, duration, dispatch_strategy, night_params=None):
        """ 
        Charge or discharge the battery for a single time step.
        
        Parameters
        ----------

            net_load: Net load to be charged into battery (negative values) or discharged from
                battery (positive values) (in kW)

            duration: duration of timestep in seconds

            dispatch_strategy: determines the battery dispatch strategy.
                Options include:
                    night_const_batt (constant discharge at night)
                    night_dynamic_batt (updates the discharge rate based on remaining available
                        capacity)
                    available_capacity (discharged up to the available capacity)

            night_params: dictionary of night params only used if dispatch_strategy is not
                set to available_capacity with the following keys: 
                    is_night: if it is currently nighttime, boolean
                    night_duration: length of current night in number of timesteps, if is_night is
                        false, it is 0
                    night_hours_left: remaining hours in night
                    soc_at_initial_hour_of_night: the initial state of charge during the first hour of
                        the night

        """

        # Initialize delta power
        delta_power = 0
        
        # Spare capacity available for charging
        spare_capacity = (self.soc_upper_limit - self.soc) * self.batt_capacity

        # Available capacity for discharging
        available_capacity = (self.soc - self.soc_lower_limit) \
            * self.batt_capacity

        # Determine if the battery is charged: if there's extra power and room in the battery
        if net_load < 0 < spare_capacity:
            
            # Amount of energy applied at terminals to fill battery
            external_energy = spare_capacity/self.charge_eff
            
            # Charging power is minimum of net load, battery power rating and  available
            #   capacity divided by # hours per timestep
            delta_power = -min([abs(net_load), self.power, external_energy/(duration/3600)])
        
            # Update SOC 
            self.soc -= delta_power*duration/3600*self.charge_eff / self.batt_capacity
                        
        # Determine if the battery is discharged
        elif net_load > 0 and available_capacity > 0:
        
            # Amount of energy available to meet the load
            external_energy = available_capacity*self.discharge_eff
            
            # Discharging power is minimum of net load, battery power rating and available
            #   capacity divided by # hours per timestep
            delta_power = min([abs(net_load), self.power, external_energy/(duration/3600)])
            
            # If nighttime and using a night-based dispatch strategy, limit the discharge rate
            if night_params and night_params['is_night']:
                # Calculate the maximum nighttime discharge power for the whole night
                if dispatch_strategy == 'night_const_batt':
                    max_nighttime_power = \
                        (night_params['soc_at_initial_hour_of_night'] - self.soc_lower_limit) \
                        * self.batt_capacity * self.discharge_eff / night_params['night_duration']
                elif dispatch_strategy == 'night_dynamic_batt':
                    max_nighttime_power = \
                        (self.soc - self.soc_lower_limit) \
                        * self.batt_capacity * self.discharge_eff \
                        / night_params['night_hours_left']

                delta_power = min([delta_power, max_nighttime_power])

            # Update SOC 
            self.soc -= delta_power * duration / 3600 / self.discharge_eff \
                / self.batt_capacity

        # Check that the SOC is above or below limit, with a slight buffer to allow for
        #   rounding errors
        if self.soc > self.soc_upper_limit+1E-5 or self.soc < self.soc_lower_limit-1E-5:
            message = 'Battery SOC is above or below allowable limit.'
            log_error(message)
            raise Exception(message)
        
        # Return power used to charge or discharge battery
        return delta_power
            
    
class Generator(Component):
    """
    Generator class
    
    Parameters
    ----------
    
        category: equipment type, set to 'generator'
        
        existing: whether or not the component already exists on site

        rated_power: generator rated power in kW
        
        num_units: number of generator units

        fuel_curve_model: gallons/hr at 1/4, 1/2, 3/4, and full load.
            Expects a Pandas series with indices of ['1/4 Load (gal/hr)', '1/2 Load (gal/hr)',
             '3/4 Load (gal/hr)', 'Full Load (gal/hr)']

        ideal_minimum_load: fractional load level below which a warning is issued.
            Currently unused.
            Default = 0.3
            
        loading_level_to_add_unit: fractional load level above which another generator unit is
            added. Currently unused.
            Default = 0.9
            
        loading_level_to_remove_unit: fractional load level below which a generator unit is
            removed (if possible). Currently unused.
            Default = 0.3
            
        capital_cost: cost for both parts and labor in USD

        prime_generator: (boolean) indicates order of dispatch for multiple generators

    Methods
    ----------
    
        get_rated_power: returns the generator rated power in kW
        
        get_fuel_curve_model: returns the fuel curve model coefficients

        calculate_fuel_consumption: calculates dispatch and fuel consumed by generators

        calc_capital_cost: calculate capital costs

        calc_om_cost: calculate om costs
                
    """
    
    def __init__(self, existing, rated_power, num_units, fuel_curve_model, capital_cost,
                 prime_generator=True, ideal_minimum_load=0.3, loading_level_to_add_unit=0.9,
                 loading_level_to_remove_unit=0.3, validate=True):
        self.category = 'generator'
        self.existing = existing
        self.rated_power = rated_power  # kW
        self.num_units = num_units
        self.prime_generator = prime_generator
        self.fuel_curve_model = fuel_curve_model
        self.capital_cost = capital_cost
        self.ideal_minimum_load = ideal_minimum_load
        self.loading_level_to_add_unit = loading_level_to_add_unit
        self.loading_level_to_remove_unit = loading_level_to_remove_unit

        if validate:
            # List of initialized parameters to validate
            args_dict = {
                'existing': existing, 'rated_power': rated_power,
                'num_units': num_units,
                'fuel_curve_model': fuel_curve_model,
                'capital_cost': capital_cost,
                'prime_generator': prime_generator,
                'ideal_minimum_load': ideal_minimum_load,
                'loading_level_to_add_unit': loading_level_to_add_unit,
                'loading_level_to_remove_unit': loading_level_to_remove_unit
            }

            # Validate input parameters
            validate_all_parameters(args_dict)

    def __repr__(self):
        return 'Generator: Rated Power: {:.1f}kW, Number: {}'.format(
            self.rated_power, self.num_units)
        
    def get_rated_power(self):
        return self.rated_power
        
    def get_fuel_curve_model(self):
        return self.get_fuel_curve_model

    def calculate_fuel_consumption(self, gen_power, duration, validate=True):
        """
        Calculates fuel consumed by generator, and also return number of hours at each load
            bin which is used for the load duration calculation.
        Note: the current implementation assumes that all generators are dispatched together
            at the same loading level. This will have to be updated in the future.

        Inputs:
            gen_power: dataframe with one column: 'gen_power'
            duration: time step length in seconds

        Outputs:
            load_duration_df: dataframe with columns [binned_load, num_hours]
            total_fuel: total fuel consumed in L

        """

        # Validate input parameters
        if validate:
            args_dict = {'gen_power': gen_power, 'duration': duration}
            validate_all_parameters(args_dict)

        # Create load bins
        gen_power.columns = ['gen_power']
        gen_power['binned_load'] = gen_power['gen_power'].apply(round)
        grouped_load = gen_power.groupby('binned_load')['gen_power'].\
            count().to_frame(name='num_timesteps').reset_index()

        # Calculate the number of hours at each load bin
        grouped_load['num_hours'] = grouped_load['num_timesteps'] * duration \
            / 3600

        # Determine fuel consumption function
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', np.RankWarning)
            fuel_func = np.poly1d(np.polyfit([0, 0.25, 0.5, 0.75, 1],
                                   [0,
                                    self.fuel_curve_model['1/4 Load (gal/hr)'],
                                    self.fuel_curve_model['1/2 Load (gal/hr)'],
                                    self.fuel_curve_model['3/4 Load (gal/hr)'],
                                    self.fuel_curve_model['Full Load (gal/hr)']],
                                   10))

        # Calculate fuel consumption for each bin in Gallons
        grouped_load['fuel_used'] = grouped_load['binned_load'].apply(
            lambda x: fuel_func(x/(self.rated_power*self.num_units)))

        # Calculate total fuel consumption
        total_fuel = (grouped_load['fuel_used']
                      * grouped_load['num_hours']).sum()
        total_fuel = np.max([total_fuel, 0.])

        return grouped_load, total_fuel

    def calc_capital_cost(self, cost_df, existing_components):
        """ Calculate generator capital costs """

        if 'generator' in existing_components.keys():
            return 0
        else:
            return self.capital_cost * self.num_units

    def calc_om_cost(self, cost_df, existing_components):
        """ Calculate generator O&M costs """

        if 'generator' in existing_components.keys():
            return 0
        else:
            # Parse generator o&m costs from costs dataframe
            gen_om_per_kW_scalar = cost_df['Generator_scalar'].values[1]
            gen_om_exp = cost_df['Generator_exp'].values[1]
            return gen_om_per_kW_scalar * (self.rated_power * self.num_units)**gen_om_exp


class GeneratorGroup(Component):
    """
    Generator Group class
    
    Parameters
    ----------
    
        category: equipment type, set to 'generator'

        generator_list: a list of Generator objects

    Methods
    ----------

        calc_capital_cost: calculate capital costs

        calc_om_cost: calculate om costs
                
    """
    
    def __init__(self, generator_list, validate=True):
        self.category = 'generator'
        self.generator_list = generator_list
        self.rated_power = np.sum([gen.rated_power * gen.num_units for gen in self.generator_list])
        self.num_units = 1  # Used for results parsing to be consistent with Generator object

        if validate:
            # List of initialized parameters to validate
            args_dict = {
                'generator_list': generator_list
            }

            # Validate input parameters
            validate_all_parameters(args_dict)

    def __repr__(self):
        return f'GeneratorGroup: {", ".join([gen.__repr__() for gen in self.generator_list])}'

    def calc_capital_cost(self, cost_df, *args):
        """ Calculate generator group capital costs """

        return np.sum([gen.calc_capital_cost(cost_df, {}) for gen in self.generator_list])

    def calc_om_cost(self, cost_df, *args):
        """ Calculate generator O&M costs """

        return np.sum([gen.calc_om_cost(cost_df, {}) for gen in self.generator_list])


class FuelTank(Component):
    """
        Fuel Tank class

        Parameters
        ----------

            category: equipment type, set to 'fuel_tank'

            existing: whether or not the component already exists on site

            tank_size: size of fuel tank (gal)

            num_units: number of fuel tanks

            tank_cost: cost of an individual tank in USD

        Methods
        ----------

        calc_capital_cost: calculate capital costs

        calc_om_cost: calculate the o&m costs

    """

    def __init__(self, existing, tank_size, num_units, tank_cost,
                 validate=True):
        self.category = 'fuel_tank'
        self.existing = existing
        self.tank_size = tank_size
        self.num_units = num_units
        self.tank_cost = tank_cost

        if validate:
            # List of initialized parameters to validate
            args_dict = {'existing': existing, 'fuel_tank_size': tank_size,
                         'num_units': num_units, 'fuel_tank_cost': tank_cost}

            # Validate input parameters
            validate_all_parameters(args_dict)

    def __repr__(self):
        return 'Fuel Tank: Size: {:.1f}Gal, Number: {}'.format(
            self.tank_size, self.num_units)

    def calc_capital_cost(self, *args):
        """ Calculate capital cost of fuel tank(s) """

        return self.num_units * self.tank_cost

    def calc_om_cost(self, *args):
        return 0


# System class, contains components
class MicrogridSystem:
    """ 
    Microgrid system containing multiple components.
    
    Parameters
    ----------
    
        name: unique system name
        
        components: collection of Component objects
        
        costs_usd: dictionary containing different costs in USD
        
        capital_cost_usd: total captial costs in USD
        
        annual_benefits_usd: dictionary of total annual benefits from the system in USD by re_type
        
        pv_area_ft2: total pv array area in ft^2
        
        fuel_used_gal: aggregated (mean, std, and worst-case) fuel used by system from
            simulations in gallons
        
        simple_payback_yr: system payback based on capital costs and annual benefits in years
            
        pv_percent: aggregated (mean, std, and worst-case) percent of load met by pv across
            all simulations
            
        batt_percent: aggregated (mean, std, and worst-case) percent of load met by batteries
            across all simulations
            
        gen_percent: aggregated (mean, std, and worst-case) percent of load met by generators
            across all simulations

        generator_power_kW: aggregated (mean, std, and worst-case) required generator size
            across all simulations

        load_duration: Pandas dataframe containing load duration curve  aggregated across all
            simulations. Contains columns:
            [load_bin, num_hours, num_hours_at_or_below]
            
        simulations: dictionary to hold simulator objects

        outputs: dictionary of outputs from all simulations
            
    Methods
    ----------
    
        add_component: adds a component to the system
        
        get_components: returns the system components
        
        get_pv_area: returns the area of the pv area in ft^2

        size_fuel_tank: sizes fuel tanks required to meet the maximum generator fuel
            consumption
        
        calc_costs: calculates the system costs in USD
        
        calc_annual_RE_benefits: calculates the annual system benefits in USD

        calc_net_metering_revenue: helper function for calc_annual_RE_benefits. Calculates
            revenue for either the total RE system or an existing system.

        calc_payback: calculates system payback in years

        calculate_resilience_metrics: calculates various resilience metrics
        
        set_outputs: sets system outputs from aggregated simulations
        
        get_outputs: returns all attributes
        
        print_outputs: prints all attributes
        
        plot_load_duration: plots the load duration curve
        
        get_name: returns the system name
        
        add_simulation: add a simulation to the simulations dictionary
        
        print_simulation_list: prints names of simulator objects
        
        get_simulation: returns a specific simulator object given its name

        plot_dispatch: plot the dispatch graph for the selected system

        plot_generator_histograms: plots histograms of generator power and fuel consumption
            across the simulations

        plot_histogram: plots a histogram for a field in the outputs dictionary, and returns
            the axes object

    """
    
    def __init__(self, name):
        self.name = name
        self.components = None
        self.costs_usd = None
        self.capital_cost_usd = 0
        self.om_cost_usd = 0
        self.annual_benefits_usd = {}
        self.demand_benefits_usd = {}
        self.pv_area_ft2 = 0
        self.fuel_used_gal = 0
        self.simple_payback_yr = np.nan
        self.pv_percent = None
        self.batt_percent = None
        self.gen_percent = None
        self.generator_power_kW = None
        self.load_duration = None
        self.simulations = {}
        self.outputs = None

    def __repr__(self):
        return_string = ''
        for component in self.components:
            return_string += str(component)
            return_string += '\n'
        return return_string
    
    def add_component(self, *args):
        pass

    def get_components(self):
        return self.components
        
    def get_pv_area(self):
        pass

    def calc_costs(self, *args):
        pass
    
    def calc_annual_RE_benefits(self, *args):
        pass

    def calculate_resilience_metrics(self, *args):
        pass

    def calc_payback(self):
        """
            Calculate simple payback, based on capital cost and annual
            benefits.
        """
        
        self.simple_payback_yr = self.capital_cost_usd / \
                                 (np.sum(list(self.annual_benefits_usd.values())) +
                                  np.sum(list(self.demand_benefits_usd.values())) -
                                  self.om_cost_usd)
        if self.simple_payback_yr < 0:
            self.simple_payback_yr = np.nan
    
    def set_outputs(self, outputs, validate=True):
        """ Add the simulation results to the system attributes """

        # Validate input parameters
        if validate:
            args_dict = {'outputs': outputs}
            validate_all_parameters(args_dict)

        for key, val in outputs.items():
            if key in self.__dict__.keys():
                self.__setattr__(key, val)
                
    def get_outputs(self, params=None):
        """ Return simulation outputs specified by the params list """
        
        out_dict = {}
        if params is None:
            for param in self.__dict__.keys():
                out_dict[param] = self.__getattribute__(param)

        else:
            for param in params:
                try:
                    out_dict[param] = self.__getattribute__(param)
                except AttributeError:
                    out_dict[param] = None
                
        return out_dict
    
    def print_outputs(self):
        """ Print the system results """
        for key, val in self.__dict__.items():
            if type(val) in [str, dict, int, float, np.float64]:
                print(key, val)
                
    def plot_load_duration(self):
        """ Plots the load duration curve """
        
        try:
            fig = plt.figure()

            # Plot hours not met
            ax2 = fig.add_subplot(111)
            self.load_duration[['hours_not_met_max',
                                'hours_not_met_average']].plot(ax=ax2)
            ax2.set_xlabel('Generator Power (kW)')
            ax2.set_ylabel('Number of Hours Not Met')
            ax2.legend(['Maximum of Scenarios', 'Average of Scenarios'])

        except NameError:
            print('You must set the system outputs first.')
            
    def get_name(self):
        return self.name
        
    def add_simulation(self, sim_name, sim, validate):
        """ Add a simulation to the simulations dictionary. """

        # Validate input parameters
        if validate:
            args_dict = {'sim': sim}
            validate_all_parameters(args_dict)

        if sim_name not in self.simulations.keys():
            self.simulations[sim_name] = sim
        else:
            print('Could not add simulation, a simulation with that name already exists.')
        
    def print_simulation_list(self):
        print(self.simulations.keys())
                
    def get_simulation(self, sim_name):
        """
            If sim name is 'max' or 'min' return simulation with max or min pv.
        """

        if sim_name in self.simulations.keys():
            return self.simulations[sim_name]
        elif sim_name == 'max':
            sim_pv = {key: val.dispatch_df['pv_power'].sum() for key, val
                      in self.simulations.items()}
            max_pv = max(sim_pv, key=sim_pv.get)
            return self.simulations[max_pv]
        elif sim_name == 'min':
            sim_pv = {key: val.dispatch_df['pv_power'].sum() for key, val
                      in self.simulations.items()}
            min_pv = min(sim_pv, key=sim_pv.get)
            return self.simulations[min_pv]
        else:
            message = 'The simulation name you entered does not exist.'
            log_error(message)
            print(message)
            
    def get_components_for_figure(self):
        title_string = ''
        legend_list = ['Load']
        dispatch_list = ['load']
        if 'pv' in self.components:
            title_string += '{:.0f}kW PV, '.format(self.components['pv'].capacity)
            legend_list += ['PV']
            dispatch_list += ['pv_power']
        if 'mre' in self.components:
            title_string += '{:.0f}kW MRE, '.format(self.components['mre'].capacity)
            legend_list += ['MRE']
            dispatch_list += ['mre_power']
        if 'battery' in self.components:
            title_string += '{:.0f}kW/{:.0f}kWh Battery, '.format(
                self.components['battery'].power, self.components['battery'].batt_capacity)
            legend_list += ['Battery']
            dispatch_list += ['delta_battery_power']
        if 'generator' in self.components:
            title_string += '{:.0f}kW Generator System'.format(self.components['generator'].rated_power)
            legend_list += ['Generator']
            dispatch_list += ['gen_power']

        return title_string, legend_list, dispatch_list


    def plot_dispatch(self, sim_name, ax=None):
        """ 
        Plot the dispatch graph for the selected system. 
        sim_name can either be a simulation name, 'max' to plot dispatch for the simulation
            with the maximum PV or 'min' to plot dispatch for the simulation with the minimum
            PV
        """
        
        title_string, legend_list, dispatch_list = self.get_components_for_figure()

        if sim_name in self.simulations.keys():
            sim = self.simulations[sim_name]
        elif 'pv' in self.components:
            # Calculate the total PV for each simulation
            sim_pv = {key: val.dispatch_df['pv_power'].sum() for key, val
                      in self.simulations.items()}
            max_pv = max(sim_pv, key=sim_pv.get)
            min_pv = min(sim_pv, key=sim_pv.get)
            
            # Get the system with either the max or min PV
            if sim_name == 'max':
                sim = self.simulations[max_pv]
            elif sim_name == 'min':
                sim = self.simulations[min_pv]
            else:
                fig = plt.figure(figsize=[16, 10])
                ax1 = fig.add_subplot(121)
                self.plot_dispatch('max', ax=ax1)
                ax1.set_title(f'Maximum PV Scenario for {title_string}')
                ax1.legend(legend_list, fontsize=12)
                ax2 = fig.add_subplot(122)
                self.plot_dispatch('min', ax=ax2)
                ax2.set_title(f'Minimum PV Scenario for {title_string}')
                ax2.legend(legend_list, fontsize=12)
                ax2.set_ylim(ax1.get_ylim())
                return
        else:
            print('Please supply a simulation number to plot.')

        # Plot the dispatch graph
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()
        sim.dispatch_df[dispatch_list].plot(ax=ax)
        ax.set_xlabel('Time')
        ax.set_ylabel('Power (kW)')
        ax.set_title(f'{title_string}')
        ax.legend(legend_list, fontsize=12)

        # Add battery charging/discharging labels if there's room
        if 'battery' in self.components and ax.get_window_extent().transformed(
                fig.dpi_scale_trans.inverted()).height > 3 \
                and self.components['battery'].power > 0:
            ax.text(1.01, 0, '(charging)', color='red', rotation='vertical',
                    va='bottom', transform=ax.transAxes)
            ax.text(1.01, 1, '(discharging)', color='red', rotation='vertical',
                    va='top', transform=ax.transAxes)
            ax.text(1.01, 0.5, 'Battery Power (kW)', color='red',
                    rotation='vertical', va='center', transform=ax.transAxes)

    def plot_stacked_dispatch(self, sim_name, ax=None):
        """ 
        Plot the stacked dispatch graph for the selected system. 
        sim_name can either be a simulation name, 'max' to plot dispatch for the simulation
            with the maximum PV or 'min' to plot dispatch for the simulation with the minimum
            PV
        """
        
        # Define colors
        color_dict = {
            'PV': '#ff7f00',
            'MRE': '#a6cee3',
            'Battery': '#33a02c',
            'Generator': '#e31a1c',
            'Load': '#737373'
        }
        title_string, legend_list, dispatch_list = self.get_components_for_figure()
        resource_color_list = [color_dict[resource] for resource in legend_list
                               if resource != 'Load']
        
        if sim_name in self.simulations.keys():
            sim = self.simulations[sim_name]
        elif 'pv' in self.components:
            # Calculate the total PV for each simulation
            sim_pv = {key: val.dispatch_df['pv_power'].sum() for key, val
                      in self.simulations.items()}
            max_pv = max(sim_pv, key=sim_pv.get)
            min_pv = min(sim_pv, key=sim_pv.get)
            
            # Get the system with either the max or min PV
            if sim_name == 'max':
                sim = self.simulations[max_pv]
            elif sim_name == 'min':
                sim = self.simulations[min_pv]
            else:
                fig = plt.figure(figsize=[16, 10])
                ax1 = fig.add_subplot(121)
                self.plot_stacked_dispatch('max', ax=ax1)
                ax1.set_title(f'Maximum PV Scenario for {title_string}')
                ax2 = fig.add_subplot(122)
                self.plot_stacked_dispatch('min', ax=ax2)
                ax2.set_title(f'Minimum PV Scenario for {title_string}')
                return
        else:
            print('Please supply a simulation number to plot.')

        # Zero out battery when charging
        df_plot = sim.dispatch_df[dispatch_list].copy(deep=True)
        df_plot.loc[df_plot['delta_battery_power'] < 0, 'delta_battery_power'] = 0

        # Plot the dispatch graph
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()
        df_plot.rename(columns={
            'pv_power': 'PV',
            'mre_power': 'MRE',
            'delta_battery_power': 'Battery', 
            'load': 'Load',
            'gen_power': 'Generator'}, inplace=True)
        df_plot.drop(columns='Load').plot.area(ax=ax, color=resource_color_list, linewidth=0)
        df_plot['Load'].plot(ax=ax, color=color_dict['Load'], lw=2)
        plt.legend(loc=1, numpoints=1)
        ax.set_xlabel('Time')
        ax.set_ylabel('Power (kW)')
        ax.set_title(f'{title_string}')
        ax.set_xlim([df_plot.index[0], df_plot.index[-1]])

    def plot_generator_histograms(self):
        """
        Plots histograms of generator power and fuel consumption across the
            simulations.

        """

        fig = plt.figure(figsize=[8, 4])
        ax1 = fig.add_subplot(121)
        ax1 = self.plot_histogram('generator_power_kW', ax=ax1)
        ax1.set_xlabel('Generator Power (kW)')
        ax1.set_ylabel('Number of Scenarios')
        ax2 = fig.add_subplot(122)
        ax2 = self.plot_histogram('fuel_used_gal', ax=ax2)
        ax2.set_xlabel('Fuel Consumption (Gal)')
        plt.tight_layout()

    def plot_histogram(self, outputs_field, ax=None):
        """
        Plots a histogram for a field in the outputs dictionary, and returns the axes object.
        Options for outputs_field include:
            'pv_percent', 'batt_percent', 'gen_percent', 'storage_recovery_percent',
            'fuel_used_gal', 'generator_power_kW'

        """

        if outputs_field in self.outputs.keys():
            if ax is None:
                fig = plt.figure()
                ax = fig.add_subplot(111)
            ax.hist(self.outputs[outputs_field])
            return ax
        else:
            print('ERROR: {} is not a valid outputs field.'.format(outputs_field))


class SimpleMicrogridSystem(MicrogridSystem):
    """ 
    Simple system where you can only have one type of each component. 
    
    For generators there can be multiple if they have the same specs.
    
    Parameters
    ----------
    
        same as for parent class MicrogridSystem
        
        
    Methods
    ----------
    
        same as for parent class MicrogridSystem

    """
    
    def __init__(self, name):
        super().__init__(name)
        self.components = {}
        
    def __repr__(self):
        return_string = ''
        for key, val in self.components.items():
            return_string += str(val)
            return_string += '\n'
        return return_string
            
    def add_component(self, component, validate=True):
        """ Add a component (pv, battery, generator) to the system """

        # Validate input parameters
        if validate:
            args_dict = {'component': component}
            validate_all_parameters(args_dict)

        self.components[component.category] = component

    def get_pv_area(self):
        """ Return the area of the PV array """
        if 'pv' in self.components:
            self.pv_area_ft2 = self.components['pv'].calc_area()

    def size_fuel_tank(self, fuel_tank_sizes, existing_components, max_fuel_used):
        """
            Sizes fuel tanks required for the maximum generator fuel consumption.
        """

        # Discount required capacity by any existing tanks
        if 'fuel_tank' in existing_components:
            max_fuel_used = max_fuel_used - \
                       existing_components['fuel_tank'].tank_size \
                       * existing_components['fuel_tank'].num_units

        # If additional tanks are required, calculate the most economical tank size and number
        if max_fuel_used > 0:
            # Calculate the required number of tanks and total cost for each possible tanks
            #   size
            sizes_mod = fuel_tank_sizes.copy(deep=True)
            sizes_mod['num_tanks'] = sizes_mod.index.map(
                lambda x: int(np.ceil(max_fuel_used/x)))
            sizes_mod['total_cost'] = sizes_mod['Cost (USD)'] * sizes_mod['num_tanks']

            # Get the least expensive size and number
            best_tank_size = sizes_mod.sort_values(by='total_cost').iloc[0]
            fuel_tank = FuelTank(False, float(best_tank_size.name),
                                 int(best_tank_size.num_tanks), best_tank_size['Cost (USD)'])
        else:
            fuel_tank = FuelTank(False, 0, 0, 0)

        # Add the fuel tank to the system components
        self.add_component(fuel_tank)

    def calc_costs(self, system_costs, existing_components={}, validate=True):
        """ Calculate the capital and maintenance costs of the system """
        
        # Validate input parameters
        if validate:
            args_dict = {'system_costs': system_costs,
                         'existing_components': existing_components}
            validate_all_parameters(args_dict)

        # Create a dictionary to hold costs for each component
        self.costs_usd = {}

        # Calculate the costs for each component and add to the total
        for component_type, component in self.components.items():
            self.costs_usd['{}_capital'.format(component_type)] = \
                component.calc_capital_cost(
                    system_costs['{}_costs'.format(component_type)],
                existing_components)
            self.costs_usd['{}_o&m'.format(component_type)] = \
                component.calc_om_cost(system_costs['om_costs'],
                                       existing_components)
            self.capital_cost_usd += self.costs_usd[
                '{}_capital'.format(component_type)]
            self.om_cost_usd += self.costs_usd['{}_o&m'.format(component_type)]

    def calc_annual_RE_benefits(self, tmy_generation, annual_load_profile, re_type, 
                                duration, electricity_rate, net_metering_rate, demand_rate,
                                batt_sizing_method, batt_eff, inv_eff,
                                net_metering_limits=None, existing_components={},
                                validate=True):
        """ 
        Calculate the annual financial benefit of the system.
            
        tmy_generation: power time series for a 1kW array based on TMY data
        re_type: type of RE resource: 'pv' or 'mre'
        annual_load_profile: annual load time series
        duration: duration of each timestep in the time series in seconds
        net_metering_limits: any limit on the net-metering policy in the form of:
            {type: ['capacity_cap' or 'percent_of_load'], value: [<kW value> or <percentage>]}
        electricity_rate: electricity rate in $/kWh
        net_metering_rate: rate for exported energy in $/kWh
        demand_rate: rate for demand charges in $/kW. Can either be a single number or a list
            of 12 numbers.
        existing_components: dictionary of any existing microgrid components
        
        """

        # Validate input parameters
        if validate:
            args_dict = {'tmy_generation': tmy_generation,
                         're_type': re_type,
                         'annual_load_profile': annual_load_profile,
                         'duration': duration,
                         'electricity_rate': electricity_rate,
                         'net_metering_rate': net_metering_rate,
                         'demand_rate_list': demand_rate,
                         'existing_components': existing_components}
            if net_metering_limits is not None:
                args_dict['net_metering_limits'] = net_metering_limits
            validate_all_parameters(args_dict)

        # If no net-metering rate is set, use electricity rate
        if net_metering_rate is None:
            net_metering_rate = electricity_rate

        # Calculate revenue from the whole RE system
        self.annual_benefits_usd[re_type] = self.calc_net_metering_revenue(
            annual_load_profile, tmy_generation, self.components[re_type].capacity,
            net_metering_limits, electricity_rate, net_metering_rate,
            batt_sizing_method, batt_eff, inv_eff)

        # Reduce revenue generating capacity if there is an existing RE system of the same type
        if re_type in existing_components:
            existing_capacity_benefits_usd = \
                self.calc_net_metering_revenue(
                    annual_load_profile, tmy_generation,
                    existing_components[re_type].capacity, net_metering_limits,
                    electricity_rate, net_metering_rate, batt_sizing_method,
                    batt_eff, inv_eff)
            self.annual_benefits_usd[re_type] = max(
                self.annual_benefits_usd[re_type] - existing_capacity_benefits_usd, 0)

        # If a demand rate is set, calculate demand charges with and without the RE system
        if demand_rate is not None:
            # Check if existing RE, and if so, subtract generation from load
            if re_type in existing_components:
                net_load = annual_load_profile - tmy_generation.values * \
                                existing_components[re_type].capacity
            else:
                net_load = annual_load_profile

            # Calculate demand charge without new RE capacity
            base_demand = self.calc_demand_savings(net_load, demand_rate)

            # Calculate demand charge with new RE capacity
            net_load = annual_load_profile - tmy_generation.values * \
                self.components[re_type].capacity
            new_demand = self.calc_demand_savings(net_load, demand_rate)

            # Subtract the demand charges to get the savings
            self.demand_benefits_usd[re_type] = max(base_demand - new_demand, 0)

    @staticmethod
    def calc_demand_savings(load_profile, demand_rate):
        """
        Helper function for calc_annual_RE_benefits. Calculates demand charge savings from RE
            generation.
        """

        # Find peak by month
        peak_month_demand = load_profile.resample('M').max()

        # Multiply by demand charge rate
        return (peak_month_demand * demand_rate).sum()

    @staticmethod
    def calc_net_metering_revenue(load_profile, re_power_profile, capacity,
                                  net_metering_limits, electricity_rate, net_metering_rate,
                                  batt_sizing_method, batt_eff, inv_eff):
        """
        Helper function for calc_annual_RE_benefits. Calculates revenue for either the total
            RE system or an existing system.

        """

        # Calculate hourly power not-imported due to RE system and power exported
        power_df = pd.DataFrame(np.transpose([load_profile.values, re_power_profile.values]),
                                columns=['load', 'RE'], index=load_profile.index)
        power_df['RE_total'] = power_df['RE'] * capacity
        power_df['power_not_imported'] = power_df.apply(
            lambda x: min(x['load'], x['RE_total']), axis=1)
        power_df['power_exported'] = power_df['RE_total'] - power_df['power_not_imported']

        revenue = None
        if net_metering_limits is None:
            # There are no net-metering limits
            revenue = \
                power_df['power_not_imported'].sum() * electricity_rate + \
                power_df['power_exported'].sum() * net_metering_rate
            
        elif net_metering_limits['type'] == 'capacity_cap':
            # There is an instantaneous capacity cap on net-metering
            revenue = \
                power_df['power_not_imported'].sum() * electricity_rate + \
                power_df.apply(lambda x: min(x['power_exported'],
                                             net_metering_limits['value']),
                               axis=1).sum() * net_metering_rate
            
        elif net_metering_limits['type'] == 'percent_of_load':
            # There is a percent of load cap on net-metering
            revenue = \
                power_df['power_not_imported'].sum() * electricity_rate + \
                max(min(load_profile.sum() * net_metering_limits[
                    'value'] / 100 - power_df['power_not_imported'].sum(),
                    power_df['power_exported'].sum()), 0) * net_metering_rate

        elif net_metering_limits['type'] == 'no_nm_use_battery':
            # There is no net-metering, but the battery should be used to capture and use
            #   excess RE generation. This excess less losses is included in differed utility
            #   costs
            rte = (batt_eff * inv_eff)**2
            revenue = \
                (power_df['power_not_imported'].sum() +
                 power_df['power_exported'].sum() * rte) * electricity_rate

            # Check that battery sizing method is 'no_pv_export'
            if batt_sizing_method != 'no_pv_export':
                print('Warning: the "no_nm_use_battery" option should only be used with a '
                      'battery sized with the "no_pv_export" method.')

        return revenue

    def calculate_smaller_generator_metrics(self, perc, generator_options, validate=True):
        """
        Calculates metrics for a generator sized at x% of the system generator size.
        """

        # Validate input parameters
        if validate:
            args_dict = {'perc': perc, 'generator_options': generator_options}
            validate_all_parameters(args_dict)

        # Find the closest actual generator size to the x% size
        gen_size = self.components['generator'].rated_power * perc / 100
        gen_size_actual = generator_options.iloc[
            np.abs(generator_options.reset_index()['Power (kW)'] - gen_size).
            sort_values().index[0]]
        gen = Generator(existing=False, rated_power=gen_size_actual.name,
                        num_units=self.components['generator'].num_units,
                        fuel_curve_model=gen_size_actual[
                            ['1/4 Load (gal/hr)', '1/2 Load (gal/hr)',
                             '3/4 Load (gal/hr)', 'Full Load (gal/hr)']].to_dict(),
                        capital_cost=gen_size_actual['Cost (USD)'],
                        validate=False)

        # Calculate fuel consumed by new generator for each simulation
        fuel_consumed = []
        for sim_num, sim in self.simulations.items():
            fuel_consumed += [gen.calculate_fuel_consumption(
                sim.dispatch_df[['gen_power']], sim.duration,
                validate=False)[1]]

        # Aggregate across simulations
        typical_fuel_gal = np.mean(fuel_consumed)
        conservative_fuel_gal = np.max(fuel_consumed)

        # Get unmet load metrics for closest whole size
        gen_load_duration = \
            self.load_duration.loc[int(gen.rated_power * gen.num_units)].\
            rename(index={name: '{}%_smaller_gen_{}'.format(perc, name)
                          for name in self.load_duration.columns})

        # Return DataFrame with generator size, cost, average and conservative fuel, and load
        #   duration metrics
        gen_load_duration.loc['{}%_smaller_gen_size'.format(
            perc)] = gen.rated_power * gen.num_units
        gen_load_duration.loc['{}%_smaller_gen_typical_fuel_gal'.format(
            perc)] = typical_fuel_gal
        gen_load_duration.loc['{}%_smaller_gen_conservative_fuel_gal'.format(
            perc)] = conservative_fuel_gal
        gen_load_duration.loc['{}%_smaller_gen_cost'.format(
            perc)] = gen.capital_cost * gen.num_units
        return gen_load_duration
    
    def calculate_resilience_metrics(self):
        """
        Calculate aggregated resilience metrics:
            Resilience goal met (percent, cdf)
            Fuel requirements (max, cdf)
            Percent load not met (mean, cdf)
            Total load not met (mean, cdf)
            Peak load not met (mean, cdf)
        """

        res_metrics = {
            'res_goal': {},
            'fuel_req': {},
            'percent_not_met': {},
            'total_not_met': {},
            'peak_not_met': {}
        }
        
        resilience_metrics = self.outputs['resilience_metrics']
        num_sims = len(self.simulations)
        sim_length = len(self.simulations[0].load_profile)
        sim_time_range = [elem for elem in range(sim_length)]
        for system_option in ['RE_batt_gen', 'RE_batt', 'gen']:
            # Resilience goal met - percentage of simulations for which there was no load shortfall
            res_metrics['res_goal'][system_option] = {}
            res_metrics['res_goal'][system_option]['scalar'] = \
                np.sum([sim[system_option]['total_load_not_met'] == 0. for sim in resilience_metrics]) / num_sims
            # Probability that the load will be met for at least X hours
            hours = [sim[system_option]['hours_before_shortfall'] for sim in resilience_metrics]
            cdf = stats.ecdf(hours).sf.evaluate(sim_time_range)
            res_metrics['res_goal'][system_option]['cdf'] = (sim_time_range, cdf.tolist())

            # Fuel requirements
            # Maximum fuel requirements for simulatons for which all load was met
            res_metrics['fuel_req'][system_option] = {}
            fuel = np.array([sim[system_option]['fuel_use'] for sim in resilience_metrics])
            fuel = fuel[fuel >= 0]
            res_metrics['fuel_req'][system_option]['scalar'] = np.max(fuel) if len(fuel) else 0.
            # % of simulations requiring X gal or less of fuel 
            cdf = stats.ecdf(fuel).cdf
            res_metrics['fuel_req'][system_option]['cdf'] = (cdf.quantiles.tolist(), cdf.probabilities.tolist())

            # % load not met
            res_metrics['percent_not_met'][system_option] = {}
            perc_load = np.array([sim[system_option]['percent_load_not_met'] for sim in resilience_metrics])
            perc_load = perc_load[perc_load > 0]
            res_metrics['percent_not_met'][system_option]['scalar'] = np.mean(perc_load) if len(perc_load) else 0.
            # % of simulations where X % load or less is not met
            cdf = stats.ecdf(perc_load).cdf
            res_metrics['percent_not_met'][system_option]['cdf'] = (cdf.quantiles.tolist(), cdf.probabilities.tolist())

            # total load not met
            res_metrics['total_not_met'][system_option] = {}
            total_load = np.array([sim[system_option]['total_load_not_met'] for sim in resilience_metrics])
            total_load = total_load[total_load > 0]
            res_metrics['total_not_met'][system_option]['scalar'] = np.mean(total_load) if len(total_load) else 0.
            # % of simulations where X kWh of load or less is not met
            cdf = stats.ecdf(total_load).cdf
            res_metrics['total_not_met'][system_option]['cdf'] = (cdf.quantiles.tolist(), cdf.probabilities.tolist())

            # peak load not met
            res_metrics['peak_not_met'][system_option] = {}
            peak_load = np.array([sim[system_option]['peak_load_not_met'] for sim in resilience_metrics])
            peak_load = peak_load[peak_load > 0]
            res_metrics['peak_not_met'][system_option]['scalar'] = np.mean(peak_load) if len(peak_load) else 0.
            # % of simulations where the peak load not met is X kW or less
            cdf = stats.ecdf(peak_load).cdf
            res_metrics['peak_not_met'][system_option]['cdf'] = (cdf.quantiles.tolist(), cdf.probabilities.tolist())

        return res_metrics


if __name__ == "__main__":

    # For testing
    # Load in costs
    system_costs = pd.read_excel('data/MCOR Prices.xlsx', sheet_name=None, index_col=0)

    # Create a PV object
    pv = PV(existing=False, pv_capacity=500, tilt=0, azimuth=-180,
            module_capacity=0.360, module_area=3, spacing_buffer=2,
            pv_tracking='fixed', pv_racking='ground', validate=True)
    
    # Create a MRE object
    tidal = Tidal(existing=False, mre_capacity=500, generator_capacity=500, 
                  validate=True)

    # Create a battery object
    batt = SimpleLiIonBattery(existing=False, power=500, batt_capacity=2000,
                              initial_soc=1, one_way_battery_efficiency=0.9,
                              one_way_inverter_efficiency=0.95,
                              soc_upper_limit=1, soc_lower_limit=0.1, validate=True)

    gen = Generator(existing=False, rated_power=500, num_units=1,
                    fuel_curve_model={'1/4 Load (gal/hr)': 11, '1/2 Load (gal/hr)': 18.5,
                                      '3/4 Load (gal/hr)': 26.4, 'Full Load (gal/hr)': 35.7},
                    capital_cost=191000, ideal_minimum_load=0.3,
                    loading_level_to_add_unit=0.9,
                    loading_level_to_remove_unit=0.3, validate=True)

    fuel_tank = FuelTank(existing=False, tank_size=1000, num_units=2,
                         tank_cost=2000)

    # Create a microgrid system and add components
    system = SimpleMicrogridSystem('pv_500kW_batt_500kW_2000kWh')
    system.add_component(pv, validate=False)
    system.add_component(batt, validate=False)
    system.add_component(gen, validate=False)
    system.add_component(fuel_tank, validate=False)
    system.add_component(tidal, validate=False)
    #system.calc_costs(system_costs, {}, validate=False)
