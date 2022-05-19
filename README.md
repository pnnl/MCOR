# Microgrid Component Optimization for Resilience (MCOR) Tool

The goal of the MCOR tool is to provide several viable microgrid configurations that can meet 
the resilience goals of a site and maximize economic benefits. This means meeting a set of 
critical loads for a specified period of time, without any power supply from the electrical 
grid. The tool simulates power dispatch for many different system configurations, and ranks 
the systems according to the goals of the site. 

## Getting Started

### Prerequisites

* Python 3
* pvlib:
  https://github.com/pvlib/pvlib-python.git
* numba
  Install numba to speed up solar power calculation:
  http://numba.pydata.org/#installing
  
To install required packages:
``` pip install -r requirements.txt```

### Set-up
You will need to create your own credentials file with an NREL API key in order to download 
historical solar data from the NSRDB. To do so:
1. Sign up for an API key here: https://developer.nrel.gov/signup/
2. Create a file called creds.yaml in the root directory and then add your email address and 
API key to the file. 
The file should have the following format:
```yaml
nrel_api_key: <your key>
nrel_api_email: <your email>
```

### How to run MCOR
To run MCOR, update the parameter values in the first half of main.py (or your site-specific
copy of main.py) under the section titled "Define simulation parameters here". Then run 
main.py in either a terminal or your IDE of choice.
  
## Contents

### data/
Includes input data such as component costs, generator specs, and validation requirements.

### main_files/main.py
Script for running MCOR from the command line. This file must be copied and 
modified to include site parameters.

### output/
Output data from an MCOR run is saved here (Excel and pkl files).

### solar_data/
Includes downloaded NREL solar data and generated solar profile files that are created when
running MCOR.

### testing/
Unit and integration tests.

### alternative_solar_profiles.py
Alternative Solar Profiles (ASP) algorithm used for solar forecasting:
Original author in MATLAB: James Follum and Trevor Hardy

#### File contents:
Classes:
* AlternativeSolarProfiles

Standalone functions:
* date_parser

### creds.py
Includes credentials for NREL api key.

### config.py
Includes repository paths.

### generate_solar_profile.py
Calls the ASP code and calculates AC power production.

Solar power calculations carried out with pvlib-python:
    https://github.com/pvlib/pvlib-python

For an explanation of the pvlib power calculation:
    http://nbviewer.jupyter.org/github/pvlib/pvlib-python/blob/master/docs/tutorials/pvsystem.ipynb

#### File contents:
Classes:
* SolarProfileGenerator

Standalone functions:
* download_solar_data
* calc_pv_prod
* calc_night_duration
* parse_himawari_tmy

### microgrid_optimizer.py
Optimization class for simulating, filtering and ranking microgrid systems.

#### File contents:
Classes:
* Optimizer
* GridSearchOptimizer (inherits from Optimizer)

Standalone functions:
* get_electricity_rate

### microgrid_simulator.py
Microgrid simulator class. Includes the core of the system dispatch algorithm.

#### File Contents:
Classes:
* Simulator
* PVBattGenSimulator (inherits from Simulator)

Standalone functions:
* calculate_load_duration

### microgrid_system.py
Class structure for microgrid system and its components.

#### File contents:
Classes:
* Component
* PV (inherits from Component)
* Battery (inherits from Component)
* SimpleLiIonBattery (inherits from Battery)
* Generator (inherits from Component)
* FuelTank (inherits from Component)
* MicrogridSystem
* SimpleMicrogridSystem (inherits from MicrogridSystem)

### validation.py
Classes:
* Error
* ParamValidationError (inherits from Error)

Standalone functions:
* log_error
* validate_all_parameters
* validate_parameter
* ...various custom validation functions

## Contact
For questions, please reach out to sarah.newman@pnnl.gov

## Disclaimer Notice
This material was prepared as an account of work sponsored by an agency of the United States Government.  Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights.
Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.
<div align="center">
<pre style="align-text:center">
PACIFIC NORTHWEST NATIONAL LABORATORY
operated by
BATTELLE
for the
UNITED STATES DEPARTMENT OF ENERGY
under Contract DE-AC05-76RL01830
</pre>
</div>
