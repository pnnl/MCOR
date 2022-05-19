# MCOR Tests

## Running the tests
Either run the test modules directly with an IDE or run in a terminal from the project root
directory:
> python -m unittest tests/unit_tests/test*
> 
> python -m unittest tests/system_tests/*test.py

## Contents
### system_tests
There are two types of system-level tests used here, integration tests and parameter tests. 
Integration tests compare simulation results for a few test cases against pre-determined 
ground truth data. Parameter tests involve changing the value of each parameter and ensuring 
that the results change in the expected way. 

### unit_tests
Currently, unit tests only exist for the alternative_solar_profiles.py module. 
