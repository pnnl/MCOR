"""
Configuration and file paths.
"""

import os

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MAIN_DIR = os.path.join(ROOT_DIR, 'main_dir')
TEST_DIR = os.path.join(ROOT_DIR, 'tests')
SYS_TESTS_DIR = os.path.join(TEST_DIR, 'system_tests')
SOLAR_DATA_DIR = os.path.join(ROOT_DIR, 'solar_data')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
UNIT_TESTS_DIR = os.path.join(TEST_DIR, 'unit_tests')
