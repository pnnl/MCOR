# -*- coding: utf-8 -*-
"""

File contents:
    Classes:
        WaveProfileGenerator

     Author: Amy Keller
     akeller54413@gmail.com

"""
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.integrate import trapezoid
import numpy as np
import seaborn as sns
import os
from config import WAVE_DATA_DIR
#from validation import validate_all_parameters

class WaveProfileGenerator:
     """   
     Class to upload wave_data, extract samples,

     create wave profiles, and calculate power profiles.

     Parameters
     ----------
          file_dir: Should include .csv and .xlsx files only. 
               
               Input site latitude and longitude into Marine Energy Atlas, and check boxes for 
          
               "significant wave height" and "energy period" and download all years. 
               
               Insert name of file dir brackets in the main section at the bottom

               of this page.

               https://maps.nrel.gov/marine-energy-atlas/?vL=OmnidirectionalWavePowerMerged

               The code finds whichever row it should start at, so empty rows or unused values 
               
               do not matter. However, the main headers and all data afterwards should look like 
               
               the following example. They do not need to be in this exact order from left to 
               
               right, but they do need to align with each other throughout each column.

               Marine Energy Atlas should default to a proper format.
          
               Row n:  Year  Month   Day   Hour  Energy Period   Significant Wave Height 
               Row n+1:  ----    1      1     0       --.---         --.----
               Row n+2:  ----    1      1     3       --.---         --.----
               Row n+3:  ----    1      1     6       --.---         --.----
               Row n+4:  ----    1      1     9       --.---         --.----
               ............................................................

               Year is the same across all columns within excel sheet - only requirement for 
               
                    document is that one file contains one year of data. Do not combine multiple 
                    
                    years in one file. The code does not use this column, so it can be omitted.

               Month increases from 1 to 12. Month column must contain one year of data of 
               
                    whole numbers only.

               Day increases to day in the month (1-31 for january) when hours roll over. 
               
                    Day column must contain one year of data of whole numbers only.

               Hour increases by 3 hour intervals consistently throughout the excel sheet. 
               
                    0, 3, 6, 9, 12, 15, 18, 21, 0, 3...etc. Hour column must contain one year 
                    
                    of data of whole numbers only.

               Significant Wave Height column must contain a year's worth of data and all values 
               
                    are in the form of a float. Length of decimal places does not matter. 
                    
                    *Note: Significant Wave Height = SWH.

               Energy Period column must contain a year's worth of data and all values in the 
               
                    form of a float. Length of decimal places does not matter. 
                    
                    *Note: Energy Period = EP.
               
          model: input 'A', 'B', 'C', 'D' or 'a', 'b', 'c', 'd' from Small WEC Tool website 
          
               - Two Body Point Absorber. https://apps.openei.org/swec/devices

          outage_month: input 1 - 12, representing the month of outage in whole numbers.

          outage_length: represents the length of the outage in whole numberrs.

          outage_starttime: represents the starttime of the outage in whole numbers - 24-hr time.

          number_of_devices: represents the number of devices in the planned array.

          start_year: start year of data files

          end_year: end year of data files

          plot: user chooses which plots they'd like to view. Inputs are: ['power', 'energy', 'SWH', 'EP'] 
          
               written exactly like that, but in any order or quantity. Enter in the main() function 
               
               at the bottom. 


     Column labeling
     ----------
          Columns can be labeled anything, but they must be labeled, and the labels must be 
          
          listed properly in the 'get_wave_data_from_upload' variable.

          The column heading must appear exactly as they are written in the variables. 
          
          If the data sheet is slightly different, just add the label as a string to the 
          
          variable in 'get_wave_data_from_upload' method. The code just looks for one of 
          
          the associated string values to use the data from that column. Currently accepted 
          
          header names are:

          col_name_swh = ['Significant Wave Height', 'significant wave height', 
          
                         'SWH', 'swh', 'Wave Height', 'wave height']

          col_name_ep = ['Energy Period', 'energy period', 'EP', 'ep']

          col_name_month = ['Month', 'month']

          col_name_day = ['Day', 'day']
          
          col_name_hour = ['Hour', 'hour']

          )
                    
     Methods
     ----------

          get_wave_data_from_upload: Uploads all years of wave data that are provided.

          interpolate_data_points: Method to interpolate the standard 3 hour data point 

               format of wave data into an hourly format. **NOTE: this code automatically 
               
               interpolates data. If you want to put in data that is already hourly, 
               
               it will interpolate it into 20 minutes intervals. #TODO add a method

               to distinguish whether or not a data set needs interpolation.

          generate_interpolated data: Generates the proper interpolated data. 

          read_power_matrix: Chooses and uploads the appropriate power matrix based on 
          
               model selection.

          validate_inputs: Validates that all user inputs are valid inputs (excluding 
          
               excel sheets).

          generate_random_sample: Chooses from all data to generate a single random sample 
          
               based on user input.

          bin_data: Static method to return the closest value from our test SWH and EP data.

          find_binned_values: Sends through our interpolated raw data and rounds everything to the closest
          
               value that is present in our power matrix values.

          generate_sample_plots: Generates plots that represents SWH/EP/Power/Energy for the 
          
               single random sample.

          generate_all_data: Gathers and analyzes all data provided for calculations and 
          
               distribution graphs.

          generate_all_plots: Generates plots that represent the SWH/EP/Power/Energy 
          
               distribution of all data.

          generate_full_year_boxplot: Generates boxplots that represent the SWH/EP/Power 
          
               distribution throughout all of the data sheets provided to capture the 
               
               seasonal regularites.

    """
     def __init__(self, file_dir, model, outage_month, outage_length, 
                  outage_starttime, number_of_devices, start_year, end_year, plot, validate = True):
          self.file_dir = file_dir
          self.model = model
          self.outage_month = outage_month
          self.outage_length = outage_length
          self.outage_starttime = outage_starttime
          self.number_of_devices = number_of_devices
          self.start_year = start_year
          self.end_year = end_year
          self.plot = plot
          self.sig_wave_heights = []
          self.energy_periods = []
          self.all_months = []
          self.all_days = []
          self.all_hours = []
          self.all_rows = []
          self.power_matrix=[]
          self.full_swh_list=[]
          self.full_ep_list = []
          self.full_points_list = []

        # if validate:
        #      args_dict = {'model': self.model,
        #                   'outage_month': self.outage_month,
        #                   'outage_length': self.outage_length,
        #                   'outage_starttime': self.outage_starttime,
        #                   'number_of_devices': self.number_of_devices}

        #      validate_all_parameters(args_dict)


#              NOTE: These are the validations that I added to 'parameter_validation.csv'.
#              I could not get this to work, but I have my own validate definition within
#              this code, so I'm not sure if this one is needed.

#              model, str, 'a';'b';'c';'d';'A';'B';'C';'D',,,,,
#              outage_month, int, 1, 12,,,,,
#              outage_length, int, 1, 672,,,,,
#              outage_starttime, int, 0, 23,,,,,
#              number_of_devices, int, 1, 1000,,,,,
#              plot, str, 'energy'; 'power'; 'SWH'; 'EP'
         
     def get_wave_data_from_upload(self):
          """   
          Definition to upload 'wave_data'
     
          Variables
          ----------

               self.full_swh_list: contains a list of lists of SWH in this format
               
                    [[[SWH from jan 1979][swh from feb 1979]...[SWH from dec 1979]]...
                    
                     ...[[SWH from jan 2010][SWH from feb 2010]...[SWH from dec 2010]]]
               
               self.full_ep_list: contains a list of lists of energy periods in this format
               
                    [[[EP from jan 1979][EP from feb 1979]...[EP from dec 1979]]...

                     ...[[EP from jan 2079][EP from feb 2079]...[EP from dec 2079]]]

               self.full_points_list: contains a list of lists of the corresponding indexes 
               
                    in each excel sheet in this format, so we can identify where exactly 
                    
                    each SWH and EP are and in which excel sheet

          """
          col_name_swh = ['Significant Wave Height', 'significant wave height', 
                         'SWH', 'swh', 'Wave Height', 'wave height']
          col_name_ep = ['Energy Period', 'energy period', 'EP', 'ep']
          col_name_month = ['Month', 'month']
          col_name_day = ['Day', 'day']
          col_name_hour = ['Hour', 'hour']

          file_names = [elem for elem in os.listdir(self.file_dir) if 'csv' in elem or 'xlsx' in elem]
          for filename in file_names:
               file_path = os.path.join(self.file_dir, filename)
               file_extension = os.path.splitext(file_path)[-1].lower()
               if file_extension =='.csv':
                    initial_rows = pd.read_csv(file_path, nrows=10)
               elif file_extension =='.xlsx':
                    initial_rows = pd.read_excel(file_path, nrows = 10)
               else:
                    print(f"Unsupported file type: {file_extension}. See class notes.")
                    pass

               header_row = None

               #this for loop finds which row the data starts on.
               for i, row in initial_rows.iterrows():
                    row_values = [str(val).strip() for val in row.values]
                    if any(keyword in row_values for keyword in col_name_swh) and \
                         any(keyword in row_values for keyword in col_name_ep):
                              header_row=i
                              break
               if header_row is None:
                    print(f"Proper header row not found in {file_path}. See class notes. Skipping file.")
                    continue

               if file_extension == '.csv':
                    df = pd.read_csv(file_path, skiprows=header_row + 1)
               elif file_extension == '.xlsx':
                    df = pd.read_excel(file_path, skiprows=header_row + 1)

               swh = next((col for col in col_name_swh if col in df.columns), None)
               ep = next((col for col in col_name_ep if col in df.columns), None)
               month = next((col for col in col_name_month if col in df.columns), None)
               day = next((col for col in col_name_day if col in df.columns), None)
               hour= next((col for col in col_name_hour if col in df.columns), None)

               self.sig_wave_heights.append(df[swh].dropna().tolist())
               self.energy_periods.append(df[ep].dropna().tolist())
               self.all_months.append(df[month].dropna().tolist())
               self.all_days.append(df[day].dropna().tolist())
               self.all_hours.append(df[hour].dropna().tolist())
               self.all_rows.append(df.index.tolist())

          print(f"Collected {len(self.sig_wave_heights)} datasets for Significant Wave Heights" 
                 f" and {len(self.energy_periods)} datasets for Energy Periods.")

          days_in_months=[]
          months_in_years=[]
          current_month_list = []
          current_year = []
          current_swh_list = []
          current_points_list = []
          current_ep_list = []
          yearly_sig_wave_heights = []
          yearly_points_list = []
          yearly_ep_list = []
          full_sig_wave_heights = []
          full_points_list = []
          full_energy_period=[]
          prev_month = 1

          for i in range(0, len(file_names)):
               current_year = self.all_months[i]
               current_swh_year = self.sig_wave_heights[i] 
               current_ep_year = self.energy_periods[i]
               for item in range(0, len(current_year)): 
                    if current_year[item] == prev_month: 
                         current_month_list.append(current_year[item])
                         current_points_list.append(item)
                         current_swh_list.append(current_swh_year[item])
                         current_ep_list.append(current_ep_year[item])
                    elif current_year[item] != prev_month:
                         days_in_months.append(current_month_list)
                         yearly_sig_wave_heights.append(current_swh_list)
                         yearly_points_list.append(current_points_list)
                         yearly_ep_list.append(current_ep_list)
                         prev_month = current_year[item]
                         current_swh_list = []
                         current_points_list = []
                         current_ep_list = []
                         current_month_list=[]
                         current_swh_list.append(current_swh_year[item])    
                         current_points_list.append(item)   
                         current_ep_list.append(current_ep_year[item])
                         current_month_list.append(current_year[item])
               days_in_months.append(current_month_list)
               months_in_years.append(days_in_months)
               yearly_sig_wave_heights.append(current_swh_list)
               yearly_points_list.append(current_points_list)
               yearly_ep_list.append(current_ep_list)
               full_sig_wave_heights.append(yearly_sig_wave_heights)
               full_points_list.append(yearly_points_list)
               full_energy_period.append(yearly_ep_list)
               days_in_months=[]
               yearly_sig_wave_heights = []
               yearly_points_list = []
               yearly_ep_list = []
               current_month_list=[]
               current_swh_list = []
               current_points_list = []
               current_ep_list = []
               prev_month = 1   
          self.full_swh_list = full_sig_wave_heights
          self.full_ep_list = full_energy_period
          self.full_points_list = full_points_list

     def interpolate_data_points(self, data, num_interpolations =2):

        """   
          Definition to interpolate all raw SWH and EP data. This definition turns

          the default 3 hour data into hourly data. Note that if your data is already hourly

          this code will split it up into twenty minute intervales. If you are only using 
          
          MEA data, this should not be a problem

          #TODO create a method that decides whether or not the data needs to be interpolated or not.

        """
        
        def interpolate (start,end,num_points):
            return np.linspace(start, end, num = num_points+2)[1:-1]

        def interpolate_flat_list(flat_list):
            new_data = []
            for i in range(len(flat_list)-1):
                new_data.append(flat_list[i])
                interpolated_points = interpolate(flat_list[i], flat_list[i+1], num_interpolations)
                new_data.extend(interpolated_points)
            new_data.append(flat_list[-1])
            return new_data

        if isinstance(data[0], list):  # Check if the data is a nested list
            new_data = []
            for sublist in data:
                interpolated_sublist = []
                for series in sublist:
                    new_series = []
                    for i in range(len(series)-1):
                        new_series.append(series[i])
                        interpolated_points = interpolate(series[i], series[i+1], num_interpolations)
                        new_series.extend(interpolated_points)
                    new_series.append(series[-1])
                    interpolated_sublist.append(new_series)
                new_data.append(interpolated_sublist)
        else:  # Assuming data is a flat list
            new_data = interpolate_flat_list(data)

        return new_data

     def generate_interpolated_data(self):

        """   
          Definition to interpolate all raw SWH and EP data. This definition turns

          the default 3 hour data into hourly data. Note that if your data is already hourly

          this code will split it up into twenty minute intervales. If you are only using 
          
          MEA data, this should not be a problem

          #TODO create a method that decides whether or not the data needs to be interpolated or not.

        """
        self.full_swh_list = self.interpolate_data_points(self.full_swh_list)
        self.full_ep_list = self.interpolate_data_points(self.full_ep_list)
        self.full_points_list = self.interpolate_data_points(self.full_points_list)

     def read_power_matrix(self):
          """   
          Definition to upload power data, validate the model input, and create these variables. 
          
          The input model from the user sets the matrix to the correct model to be used in the 
          
          rest of the code. The power matrix here are manually converted to excel sheets from 
          
          the graphs in Small WEC Tool - Two-body point absorber. Since not all cases of energy 
          
          period are covered in the graph, I manually put in zero power for those cases that are not covered.
          
          It may make more sense to interpolate them. However, I suspect that the reason they are not 
          
          present in the matrix in because those cases don't happen. For instance, on model B, 
          
          the case of a 1.75 SWH with an EP of 2 seconds is not covered. So for these cases I 
          
          put in zero, just in case our data has it. I have also double checked all values in 
          
          these excel sheets, they are accurate with what the Small WEC Tool has to 6 decimal 
          
          places. If Small WEC Tool ever changes these values for any reason, it will not be 
          
          reflected in this model. It would be beneficial to draw the values directly from the
          
          website, but I did not have access to that and could not find anyone who would help me. 

          #TODO Check with Molly about whether the missing power output makes more sense to interpolate.
          
          Variables
          ----------

               self.power_matrix: contains a list in the format of :

                    [[input SWH, input EP, output power][input SWH, input EP, output power]

                    [input SWH, input EP, output power]...]

          """

          power_files = [
               os.path.join(WAVE_DATA_DIR, 'Power_Gen_Data', f'Model{model}AveragePower.xlsx')
               for model in ['A', 'B', 'C', 'D']
          ]
          if self.model == 'A' or self.model == 'a':
               power_list = pd.read_excel(power_files[0], skiprows=0)
          elif self.model == 'B' or self.model =='b':
               power_list = pd.read_excel(power_files[1], skiprows=0)  
          elif self.model == 'C' or self.model=='c':
               power_list = pd.read_excel(power_files[2], skiprows=0)
          elif self.model == 'D' or self.model =='d':
               power_list = pd.read_excel(power_files[3], skiprows=0)  
          else:
               print('That is not a valid model selection, choose: A, B, C, or D from the' 
                     'Small WEC Tool -> Two-body point absorber')
               return
          self.power_matrix = power_list.values.tolist()

          #TODO right now, the power excel sheets are all hand typed from the small wec tool since
          #I couldn't get the raw data to work. It may be beneficial to take directly from the website.

     def validate_inputs(self):
          
          # Definition to validate the users inputs, and turns the month input into the 
          # full word for use in graphs.

          if isinstance(self.outage_month,int) and 1 <= self.outage_month <=12:
               sample_month = self.outage_month
          else:
               print('The month is not valid, please enter an integer representation'
                     'of the month (1-12).')
               return
          self.sample_month = sample_month

          if isinstance(self.outage_length,int):
               outage_hours = self.outage_length
          else:
               print('The outage length is not valid, please enter an integer representation' 
                     'of the length in hours.')
               return
          self.outage_hours = outage_hours

          if isinstance(self.outage_starttime,int) and 0 <= self.outage_starttime <=23:
               start_time = self.outage_starttime
          else:
               print('The starttime is not valid, please enter an integer representation of'
                     'the hour from 0-23 (military time).')
               return
          self.start_time = start_time

          if isinstance(self.number_of_devices, int):
               multiplier = self.number_of_devices
          else:
               print('The number of devices is not valid, please enter an integer' 
                     'representation of how many devices will be in the array.')
               return
          self.multiplier = multiplier

          for string in self.plot:
            if string is None:
                pass
            elif isinstance(string, str):
                if string == 'energy':
                    continue
                if string == 'power':
                    continue
                if string == 'SWH':
                    continue
                if string == 'EP':
                    continue
                else:
                    print(f"{string} is not a valid plot input. Try 'energy', 'power'," 
                          f"'SWH', or 'EP'")
                    continue
            else:
                print(f"{self.plot} is not a valid plot input. Try 'energy', 'power'," 
                      f"'SWH', or 'EP'")
          
          if sample_month ==1:
               output_month = 'Janaury'
          elif sample_month ==2:
               output_month = 'February'
          elif sample_month ==3:
               output_month = 'March'
          elif sample_month == 4:
               output_month = 'April'
          elif sample_month ==5:
               output_month = 'May'
          elif sample_month ==6:
               output_month = 'June'
          elif sample_month == 7:
               output_month = 'July'
          elif sample_month==8:
               output_month = 'August'
          elif sample_month == 9:
               output_month = 'September'
          elif sample_month ==10:
               output_month = 'October'
          elif sample_month == 11:
               output_month = 'November'
          elif sample_month ==12:
               output_month = 'December'
          self.output_month =output_month

     def generate_random_sample(self):
          """   
          Definition to generate a single random sample from all of the files that 
          
          fits the users parameters.
     
          Variables
          ----------

          sample_swh: contains single list of all SWH in the specified month of the 
          
               randomly chosen year.

          sample_ep: contains single list of all the EP in the specified month of the 
          
               randomly chosen year.

          sample_hours: contains single list of all the corresponding hours in the 
          
               specified month of the randomly chosen year.

          new_sample_hours: creates a new list to randomly choose from to ensure that the 
          
               randomly chosen index will not exceed the length of the month in question.
      
                    i.e. A 48 hour outage will not begin on the last day of the month, because it 
                    
                    would proceed into the next month.  

          indices_hour_input: stores list of where all the chosen rounded time values are listed 
          
               in the 'all_hours' list 

                    i.e. [0,8,16,24,32....224] is the position of all midnights in the month of november.

          random_indices: chooses which indices throughout the month will be chosen, so in function 
          
               choosing which random day in the month will be sampled.

          random_sig_wave_height: now stores the interpolated random data set of SWH that we are using to 
          
               generate our single sample simulation.
          
          random_energy_period: now stores the interpolated random data set of EP that we are using to 
          
               generate our single sample simulation.

          """
          self.swh = 'Significant Wave Height'
          self.ep = 'Energy Period'
          self.month = 'Month'
          self.day = 'Day'
          self.hour = 'Hour'

          sample_month = self.outage_month
          outage_hours = self.outage_length
          hour_input = self.outage_starttime
          sample_year_index = random.randint(0, len(list(range(self.start_year, self.end_year+1))))
          year = list(range(self.start_year, self.end_year+1))[sample_year_index]
          # year = random.sample(list(range(self.start_year, self.end_year+1)), 1)[0]
          selected_year_data = {
               'Month':self.all_months[sample_year_index],
               'Day':self.all_days[sample_year_index],
               'Hour': self.all_hours[sample_year_index],
               'SignificantWaveHeight': self.sig_wave_heights[sample_year_index],
               'EnergyPeriod': self.energy_periods[sample_year_index]
          }

          filtered_data = {
               'Day': [],
               'Hour': [],
               'SignificantWaveHeight': [],
               'EnergyPeriod': []
          }
     
          for i, month in enumerate(selected_year_data['Month']):
               if month == sample_month:
                    filtered_data['Hour'].append(selected_year_data['Hour'][i])
                    filtered_data['SignificantWaveHeight'].append(selected_year_data['SignificantWaveHeight'][i])
                    filtered_data['EnergyPeriod'].append(selected_year_data['EnergyPeriod'][i])

          print(f"Data selected from {self.output_month} in {year} lasting {outage_hours} hours starting at {hour_input}:00")
          sample_swh = filtered_data['SignificantWaveHeight']
          sample_ep=filtered_data['EnergyPeriod']
          sample_hours = filtered_data['Hour']

          sample_swh = self.interpolate_data_points(sample_swh)
          sample_ep = self.interpolate_data_points(sample_ep)

          all_sample_hours = []
          for i in range (len(sample_hours)-1):
            current_hour = sample_hours[i]
            next_hour = sample_hours[i+1]
            all_sample_hours.append(current_hour)
            while (current_hour+1) %24 != next_hour:
                current_hour = (current_hour +1) %24
                all_sample_hours.append(current_hour)
          all_sample_hours.append(sample_hours[-1])

          random_sig_wave_height = []
          random_energy_period = []
          random_hours_selected = []

          new_sample_hours = all_sample_hours[:-outage_hours] 

          indices_hour_input = [i for i, hour in enumerate(new_sample_hours) if hour == hour_input]
          random_indices = random.choice(indices_hour_input)

          for index in range(random_indices,random_indices+(outage_hours+1),1): #outage_hours +1 is to account for starting at the zeroth position
               random_hours_selected.append(all_sample_hours[index])
               random_sig_wave_height.append(sample_swh[index])
               random_energy_period.append(sample_ep[index])

          self.random_sig_wave_height = random_sig_wave_height
          self.random_hours_selected = random_hours_selected
          self.random_energy_period = random_energy_period
          self.year = year
          self.indices_hour_input = indices_hour_input

     @staticmethod
     def bin_data(value, reference_list):
          """   
          This definition is used to return the closest values from the power matrix to the raw value.
          
          It compares the raw data, which is often accurate to many decimal places, and turns it into 
          
          the closest value from the Small WEC Tool power matrix.

               i.e. 1.876345 meter SWH turns into 1.75 meter (Small WEC Tool uses increments of .5 m 
               
               starting at 0.25 meters and ending at 8.75 meters.) 8.234 seconds turns into 8 seconds 
               
               (Small WEC Tool increments by 1 from 1-22.)

          """
          return min(reference_list, key = lambda x: abs(x-value))

     def find_binned_values(self):
          """   
          This definition inputs our single random sample that was created in 'generate_random_sample', 
          
          and uses the static method 'bin_data' to bin the SWH and EP to the closest values represented 
          
          in the power matrix. It then looks through the power matrix and finds where the combination of 
          
          SWH and EP are shown, and then returns the associated output power in kW. 
     
          Variables
          ----------
          reference_swh_list: takes all of the column 0 values from the power matrix, which correspond 
          
               to the SWH values. It is a flatlist in the form of [0.25, 0.75, 1.25, 1.75,...8.25, 8,75]

          reference_ep_list: takes all of the column 0 values from the power matrix, which correspond 
          
               to the EP values. It is a flatlist in the form of [1,2,3,...22]

          binned_swh_values: stores the randomly selected swh values rounded to the closest power 
          
               matrix SWH values

          binned_ep_values: stores the randomly selected ep values rounded to the closest power matrix 
          
               EP values

          power_output: stores all associated power outputs with the random data chosen.
          
          multiple_power: takes all power outputs in 'power_output' and multiplies each by the number 
          
               of devices in the array, provided by the user.

          """
          reference_swh_list = [sublist[0] for sublist in self.power_matrix]
          self.binned_swh_values = [self.bin_data(val, reference_swh_list) for val in self.random_sig_wave_height]
         
          reference_ep_list = [sublist[1] for sublist in self.power_matrix]
          self.binned_ep_values = [self.bin_data(value, reference_ep_list) for value in self.random_energy_period]
         
          power_dict = {(row[0], row[1]): row[2] for row in self.power_matrix}

          power_output = []

          for i in range(len(self.binned_swh_values)):
               key = (self.binned_swh_values[i], self.binned_ep_values[i])
               if key in power_dict:
                    power_output.append(power_dict[key])
       
          multiple_power = [x*self.number_of_devices for x in power_output]
          self.multiple_power = multiple_power
          self.reference_swh_list = reference_swh_list
          self.reference_ep_list = reference_ep_list

     def generate_sample_plots(self):
          """   
          This definition plots our randomly selected sample on three different graphs. 

          The first graph is the significant wave height.
          
          The second graph is the energy period.

          The third graph is the power output related to these two inputs, multiplied by the number 
          
          of devices in the array.

          Variables
          ----------
          plot_hours: sets where each data point will be plotted on the graph.

          tick_interval: adjusts the interval for the labels so they don't get too crowded.
          
          """

          plot_hours = list(range(len(self.random_hours_selected)))
          tick_interval = max(1, len(plot_hours) // 10) 
          self.plot_hours = plot_hours
          self.tick_interval = tick_interval
         
          if 'SWH' in self.plot:
            plt.figure(figsize=(10, 6))
            plt.plot(plot_hours, self.random_sig_wave_height, marker='o', linestyle='-')
            plt.title(f"Significant Wave Height Scenario for {self.output_month}, {self.year}")
            plt.xlabel(f"Time Frame Over {self.outage_hours} Hours")
            plt.ylabel('Significant Wave Height (m)')
            plt.xticks(np.arange(0, len(plot_hours), tick_interval), 
                    [f"{hour}:00" for hour in self.random_hours_selected[::tick_interval]], rotation=45)
            plt.grid(True)
            plt.tight_layout()
            plt.show()
          else:
               pass

          if 'EP' in self.plot:
            plt.figure(figsize=(10, 6))
            plt.plot(plot_hours, self.random_energy_period, marker='o', linestyle='-')
            plt.title(f"Energy Period Scenario for {self.output_month}, {self.year}")
            plt.xlabel(f"Time Frame Over {self.outage_hours} Hours")
            plt.ylabel('Energy Period (s)')
            plt.xticks(np.arange(0, len(plot_hours), tick_interval),
                       [f"{hour}:00" for hour in self.random_hours_selected[::tick_interval]],
                       rotation=45)
            plt.grid(True)
            plt.tight_layout()
            plt.show()
          else:
               pass

          if 'power' in self.plot:
            plt.figure(figsize=(10, 6))
            plt.fill_between(plot_hours, self.multiple_power, color='skyblue', alpha=0.4)
            plt.title(f"Power Output (kW) for {self.number_of_devices} devices during {self.output_month}, {self.year}")
            plt.xlabel(f"Time frame for simulated outage ({self.outage_hours} hours)")
            plt.ylabel('Average Power Output (kW)')
            plt.xticks(np.arange(0, len(plot_hours), tick_interval), 
                    [f"{hour}:00" for hour in self.random_hours_selected[::tick_interval]], rotation=45)
            plt.grid(True)
            plt.tight_layout
            plt.show()
          else:
               pass

     def generate_all_data(self):
          """   
          This definition gathers all of the data that our sample could have been taken from. 

               i.e. Our sample outage is 9 hours in November starting at 3 pm.
                
               The simulation chooses the 7th day of November in 1999 starting at 3pm, and taking all data 
               
               for the next 9 hours (plotted in 'generate_sample_plot'). This definition will gather every 
               
               day of November in every year starting at 3pm + 9 hours (10 data points)

          It uses all of this data to generate calculations and graphs. 

          It also prints out the total energy provided in the sample simulation in kWh.

          It also prints out the maximum and minimum energy output of all possible sample simulations in kWh.

          It also prints out the mean energy output of all possible sample simulations in kWh.

          It also prints out the standard deviation of energy output of all possible sample simulations in kWh.

          Variables
          ----------
          yearly_swh: now stores a list of lists of lists of SWH

                    i.e. [[[SWH for 11-1-1979 from 3pm-12am][SWH for 11-2-1979 from 3pm-12am]...
                    
                    [SWH for 11-31-1979 from 3pm-12am]]
                    
                    ...[[SWH for 11-1-2010 from 3pm-12am][SWH for 11-2-2010 from 3pm-12am]...
                    
                    [SWH for 11-31-2010 from 3pm-12am]]]

                    Length of each yearly list adjust for whichever month is being queried. 
                    
                    Novemeber is 31 days long, so the inner list is 31 elements long. 
                    
                    Each of those inner lists (31 for Nov) will contain however many data points were 
                    
                    determined to be needed from 'generate_random_sample' (10 in this case).
          
          yearly_ep: now stores a list of lists of EP

                    i.e. [[[EP for 11-1-1979 from 3pm-12am][EP for 11-2-1979 from 3pm-12am]...
                    
                    [EP for 11-31-1979 from 3pm-12am]]...[[EP for 11-1-2010 from 3pm-12am]
                    
                    [EP for 11-2-2010 from 3pm-12am]...[EP for 11-31-2010 from 3pm-12am]]]

                    Inner lengths adjust for whichever month is being queried. Novemeber is 31 days long, 
                    
                    so the inner list is 31 elements long.

                    Each of those inner lists (31 for Nov) will contain however many data points were 
                    
                    determined to be needed from 'generate_random_sample' ('get_hourly_data'). 
          
          full_list_swh: now contains a list of lists. each inner list contains each SWH point in our 
          
               set throughout every year.

                    i.e.  [[3 pm SWH for every Nov for every year][4pm SWH for every Nov for every year]...
                    
                    [12am SWH for every Nov for every year]]

                    There are as many lists as there are sample data times, so 10 in our scenario.

                    So each inner list is going to be (number of years provided)x(number of days in the given month) 
                    
                    in length.

          full_list_ep: now contains a list of lists. each inner list contains each EP point in our set 
          
               throughout every year.

                    i.e.  [[3pm EP for every Nov for every year][4pm EP for every Nov for every year]...
                    
                    [12am EP for every Nov for every year]]

                    There are as many lists as there are sample data times, so 10 in our scenario.

                    So each inner list is going to be (number of years provided)x(number of days in the given 
                    
                    month) in length.

          binned_full_swh_list: same as 'full_swh_list', but with the values rounded to our power matrix 
          
               SWH values.

          binned_full_ep_list: same as 'full_ep_list', but with the values rounded to our power matrix EP values.

          full_power_output: same format as 'full_swh_list' and 'full_ep_list', but it contains the associated 
          
               power outputs from the binned SWH and EP values.

                    i.e. [[3pm power output for every Nov for every year][4pm power output for every Nov for 
                    
                    every year]...[12am power output for every Nov for every year]]

          full_multiple_power: takes the power output from 'full_power_output' and multiplies each value by the 
          
               specified number of units in the array.

          binned_og_swh_list: same as 'yearly_swh', but takes each value and sets it to the closest value of SWH 
          
               in the power matrix.

          binned_og_ep_list: same as 'yearly_ep', but takes each value and sets it to the closest value of EP 
          
               in the power matrix.

          full_og_power_output: same format as 'yearly_swh and yearly_ep', but it outputs the associated power 
          
               output of each data set.

          final_og_multiple_power: same format as 'full_og_power_output', but it takes each value and multiplies 
          
               it by the number of devices in the array.

          full_energy: contains a list of the energy output for every single outage situation that could have 
          
               been drawn from as a flat list.

               This was done by integrating the power output for each and every scenario that could have been 
               
               picked from given the user's inputs.

          final_swh_deviation: list of standard deviation within all the SWH of the sample. 

               i.e. The standard deviation of every single 3pm SWH in every November over all historical data 
               
               provided. and then the standard deviation of every 4 pm SWH in every November over all historical 
               
               data provid...etc up until the end of your outage length. So if you have 10 data points in your 
               
               9 hour outage, you will get 10 standard deviations, one for each hour.
          
          final_ep_deviation: list of standard deviation within all the EP of the sample. 
          
               i.e. The standard deviation of every single 3pm EP in every November over all historical data 
               
               provided. and then the standard deviation of every 4 pm EP in every November over all historical 
               
               data provid...etc up until the end of your outage length. So if you have 10 data points in your 
               
               9 hour outage, you will get 10 standard deviations, one for each hour.

          final_mean_power: full list of mean power of the power output for every hour in the sample, 
          
               accounting for the number of devices in the array.

                    i.e. [[Mean power for every 3pm in Nov in history][Mean power for every 4pm in Nov in history]
                    
                    ...[Mean power for every 12am in Nov in history]]

          final_full_power_deviation: list of standard deviations of the power output for every hour in 
          
               the sample, accounting for the number of devices in the array.
               
                    i.e. [[std dev of power for every 3pm in Nov in history][std dev of power for every 4pm 
                    
                    in Nov in history]...[std dev of power for every 12am in Nov in history]]

          energy_std_dev: uses 'full_energy' to calculate the standard deviation of all energy outputs.

               i.e. std dev of energy of all 3pm-12am throughout all Novembers in the provided history as a 
               
               single number.
               
          mean_energy: uses 'full_energy' to calculate the mean of all energy outputs.

               i.e. mean energy of all 3pm-12am throughout all Novembers in the provided history as a single 
               
               number.

          time: sets the appropriate time frame that each power output lasts as a list for our sample set. 

          power: sets the power output associated with the time frame for our sample set.

          energy: calculates the energy output by integrating the power output over time for our sample set.
               
          """
          current_swh=[]
          monthly_swh=[]
          yearly_swh=[]
          current_ep=[]
          monthly_ep=[]
          yearly_ep=[]

          file_names = [elem for elem in os.listdir(self.file_dir) if 'csv' in elem or 'xlsx' in elem]
          for m in range(0, len(file_names)): 
               current_year_swh = self.full_swh_list[m] 
               current_year_ep=self.full_ep_list[m]
               current_month_swh = current_year_swh[self.outage_month-1] 
               current_month_ep = current_year_ep[self.outage_month-1]
               for x in range(0,len(self.indices_hour_input)):
                    for k in range(self.indices_hour_input[x], self.indices_hour_input[x]+ self.outage_hours): 
                         if k>=len(current_month_swh):
                              if current_swh:
                                   continue
                              break
                         current_swh.append(current_month_swh[k])
                         current_ep.append(current_month_ep[k])
                    if k>= len(current_month_swh):
                         break  
                    monthly_swh.append(current_swh)
                    monthly_ep.append(current_ep)
                    current_swh = []
                    current_ep=[]
               yearly_swh.append(monthly_swh)
               yearly_ep.append(monthly_ep)
               monthly_swh=[]
               monthly_ep=[]

          flat_list_swh = [item for sublist in yearly_swh for item in sublist]
          flat_list_ep = [element for sublist in yearly_ep for element in sublist]
          self.flat_list_swh = flat_list_swh
          self.flat_list_ep = flat_list_ep

          current_points_list_swh = []
          current_points_list_ep=[]
          full_list_swh = []
          full_list_ep=[]
          for j in range(0, len(flat_list_swh[0])):
               for i in range(0, len(flat_list_swh)):
                    current_list_swh=flat_list_swh[i]
                    current_list_ep=flat_list_ep[i]
                    current_point_swh = current_list_swh[j]
                    current_point_ep=current_list_ep[j]
                    current_points_list_swh.append(current_point_swh)
                    current_points_list_ep.append(current_point_ep)
               full_list_swh.append(current_points_list_swh)
               full_list_ep.append(current_points_list_ep)
               current_points_list_swh=[]
               current_points_list_ep=[]

          binned_full_swh_list = []
          binned_full_ep_list = []
          for j in range(len(full_list_swh)):
               binned_current_swh_list = [self.bin_data(vale, self.reference_swh_list) for vale in full_list_swh[j]]
               binned_current_ep_list = [self.bin_data(val, self.reference_ep_list) for val in full_list_ep[j]]
               binned_full_swh_list.append(binned_current_swh_list)
               binned_full_ep_list.append(binned_current_ep_list)
               binned_current_ep_list=[]
               binned_current_swh_list=[]

          full_power_output=[]
          current_power_output=[]
          for k in range(len(binned_full_swh_list)):
               now_swh_list = binned_full_swh_list[k]
               now_ep_list = binned_full_ep_list[k]
               for i in range(len(now_swh_list)):
                    for j in range(len(self.power_matrix)):
                         if now_swh_list[i] == self.power_matrix[j][0] and now_ep_list[i] == self.power_matrix[j][1]:
                              current_power_output.append((self.power_matrix[j][2]))
               full_power_output.append(current_power_output)
               current_power_output=[]
         
          full_multiple_power = []
          multiplier = self.number_of_devices
          for q in range(0, len(full_power_output)):
               int_mult_power = [x*multiplier for x in full_power_output[q]]
               full_multiple_power.append(int_mult_power)

          binned_og_swh_list = [] 
          binned_og_ep_list = []
          current_binned_og_swh_list=[]
          current_binned_og_ep_list = []
          for s in range(0, len(yearly_swh)):
               for g in range(0, len(yearly_swh[s])):
                    binned_swh = [self.bin_data (val, self.reference_swh_list) for val in yearly_swh[s][g]]
                    binned_ep = [self.bin_data (val, self.reference_ep_list) for val in yearly_ep[s][g]]
                    current_binned_og_swh_list.append(binned_swh)
                    current_binned_og_ep_list.append(binned_ep)
               binned_og_swh_list.append(current_binned_og_swh_list)
               current_binned_og_swh_list = []
               binned_og_ep_list.append(current_binned_og_ep_list)
               current_binned_og_ep_list = []

          full_og_power_output = []
          new_og_power_output = []
          current_og_power_output = []
          for b in range(0, len(binned_og_swh_list)): 
               og_swh_list = binned_og_swh_list[b] 
               og_ep_list = binned_og_ep_list[b]
               for p in range(0, len(og_swh_list)): 
                    for q in range(0, len(og_swh_list[p])): 
                         for h in range(len(self.power_matrix)): 
                              if og_swh_list[p][q] == self.power_matrix[h][0] and og_ep_list[p][q] == self.power_matrix[h][1]:
                                   current_og_power_output.append((self.power_matrix[h][2]))
                    new_og_power_output.append(current_og_power_output)
                    current_og_power_output=  []
               full_og_power_output.append(new_og_power_output)
               new_og_power_output = []

          full_og_multiple_power = []
          final_og_multiple_power = []
          for q in range(0, len(full_og_power_output)): 
               for g in range (0, len(full_og_power_output[q])): 
                    int_og_mult_power = [x*multiplier for x in full_og_power_output[q][g]]
                    full_og_multiple_power.append(int_og_mult_power)
                    int_og_mult_power = []
               final_og_multiple_power.append(full_og_multiple_power)
               full_og_multiple_power = []
          self.final_og_multiple_power = final_og_multiple_power

          now_energy = []     
          full_energy = []
          for i in range(0, len(final_og_multiple_power)): 
               now_power = final_og_multiple_power[i] 
               for j in range(0, len(now_power)): 
                    time2 = np.linspace(0,self.outage_hours,len(now_power[j]))
                    power2 = now_power[j]
                    energy2 = trapezoid(power2, time2)
                    now_energy.append(energy2)
                    full_energy.append(now_energy)
          self.full_energy = full_energy

          final_swh_deviation = []
          final_ep_deviation=[]
          final_full_power_deviation = []
          final_mean_power=[]
          for x in range(0, len(full_list_swh)):
               sample_swh_std_dev = np.std(full_list_swh[x], ddof = 1)
               sample_ep_std_dev=np.std(full_list_ep[x], ddof=1)
               sample_full_power_std_dev = np.std(full_multiple_power[x], ddof=1)
               mean_power = np.mean(full_multiple_power[x])
               final_mean_power.append(mean_power)
               final_swh_deviation.append(sample_swh_std_dev)
               final_ep_deviation.append(sample_ep_std_dev)
               final_full_power_deviation.append(sample_full_power_std_dev)
          energy_std_dev = np.std(full_energy, ddof =1)
          mean_energy = np.mean(full_energy)

          time = np.linspace(0, self.outage_hours, len(self.multiple_power))
          power = self.multiple_power
          energy = trapezoid(power,time) #energy output from our single sample simulation
         
          print(f"The total energy provided in this simulation is: {round(energy,2)} kWh")
          print(F"The maximum energy output of all possible simulations is {round(np.max(full_energy),2)} kWh")
          print(f"The minimum energy output of all possible simulations is {round(np.min(full_energy),2)} kWh")
          print(f"The mean energy output of all possible simulations is {round(mean_energy,2)} kWh")
          print(f"The standard deviation of all of the possible outputs is {round(energy_std_dev,2)} kWh")

     def generate_all_plots(self):
          """   
          This definition plots all of the possible samples that could have been taken as box plots. 

          The first graph is the significant wave height at each time period across all Novembers over all years.
          
          The red dots are our single generated SWH simulation plotted on top.
          
          The second graph is the energy period at each time period across all Novembers over all years. 
          
          The red dots are our single generated EP simulation plotted on top.

          The third graph is the power output for all devices at each time period across all Novembers over 
          
          all years. The red dots are our single generated power simulation plotted on top.

          """

          new_list_swh = []
          box_plot_list_swh=[]
          for i in range(0, len(self.flat_list_swh[0])): 
               for j in range(0, len(self.flat_list_swh)): 
                    new_list_swh.append(self.flat_list_swh[j][i])
               box_plot_list_swh.append(new_list_swh)
               new_list_swh=[]
         
          data_swh = []
          for i, swh_list in enumerate(box_plot_list_swh):
               for value in swh_list:
                    data_swh.append({"Category":i +1, "Value": value})
          df = pd.DataFrame(data_swh)
         
          if 'SWH' in self.plot:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Category', y='Value', data=df)
            plt.plot(self.plot_hours, self.random_sig_wave_height,
                     'ro', label='Simulated Random Significant Wave Height')
            plt.title(
                f"Boxplot of Significant Wave Heights for {self.output_month} for all years starting at {self.outage_starttime}:00")
            plt.xlabel(
                f"Time frame for simulated outage (over {self.outage_hours} hours)")
            plt.ylabel('Significant Wave Heights (m)')
            plt.xticks(np.arange(0, len(self.plot_hours), self.tick_interval), [
                       f"{hour}:00" for hour in self.random_hours_selected[::self.tick_interval]], rotation=45)
            plt.grid(True)
            plt.tight_layout()
            plt.legend()
            plt.show()
          else:
               pass

          new_list_ep = []
          box_plot_list_ep=[]
          for i in range(0, len(self.flat_list_ep[0])): 
               for j in range(0, len(self.flat_list_ep)): 
                    new_list_ep.append(self.flat_list_ep[j][i])
               box_plot_list_ep.append(new_list_ep)
               new_list_ep=[]
         
          data_ep = []
          for i, ep_list in enumerate(box_plot_list_ep):
               for value in ep_list:
                    data_ep.append({"Category":i +1, "Value": value})
          df = pd.DataFrame(data_ep)
         
          if 'EP' in self.plot:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Category', y='Value', data=df)
            plt.plot(self.plot_hours, self.random_energy_period,
                     'ro', label='Simulated Random Energy Period')
            plt.title(
                f"Boxplot of Energy Periods for {self.output_month} for all years starting at {self.outage_starttime}:00")
            plt.xlabel(
                f"Time frame for simulated outage (over {self.outage_hours} hours)")
            plt.ylabel('Energy Period (s)')
            plt.xticks(np.arange(0, len(self.plot_hours), self.tick_interval), [
                       f"{hour}:00" for hour in self.random_hours_selected[::self.tick_interval]], rotation=45)
            plt.grid(True)
            plt.tight_layout()
            plt.legend()
            plt.show()
          else:
               pass

          flat_list_power = [element for sublist in self.final_og_multiple_power for element in sublist]
          new_list_power = []
          box_plot_list_power=[]
          for i in range(0, len(flat_list_power[0])):
               for j in range(0, len(flat_list_power)):
                    new_list_power.append(flat_list_power[j][i])
               box_plot_list_power.append(new_list_power)
               new_list_power=[]

          data_power = []
          for i, power_list in enumerate(box_plot_list_power):
               for value in power_list:
                    data_power.append({"Category":i +1, "Value": value})
          df = pd.DataFrame(data_power)

          if 'power' in self.plot:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Category', y='Value', data=df)
            plt.plot(self.plot_hours, self.multiple_power,
                     'ro', label='Simulated Power Output')
            plt.title(
                f"Boxplot of Power Output for {self.number_of_devices} devices in {self.output_month} for all years starting at {self.outage_starttime}:00")
            plt.xlabel(
                f"Time frame for simulated outage (over {self.outage_hours} hours)")
            plt.ylabel('Power Output (kW)')
            plt.xticks(np.arange(0, len(self.plot_hours), self.tick_interval), [
                       f"{hour}:00" for hour in self.random_hours_selected[::self.tick_interval]], rotation=45)
            plt.grid(True)
            plt.tight_layout()
            plt.legend()
            plt.show()
          else:
               pass

          if 'energy' in self.plot:
            flat_energy = [
                item for sublist in self.full_energy for item in sublist]
            plot_energy = pd.Series(flat_energy)
            plt.figure(figsize=(2, 6))
            sns.boxplot(y=plot_energy)
            plt.title(
                f"Boxplot of Energy Production for every {self.outage_hours} hour timeframe starting at {self.outage_starttime}:00 with {self.number_of_devices} devices in {self.output_month} for all years ")
            plt.ylabel('Energy Produced (kWh)')
            plt.tight_layout()
            plt.grid(True)
            plt.show()
          else:
               pass


     def generate_full_year_boxplot(self):
          """   
          This definition plots our entire data set as a monthly boxplot. So each January is binned and plotted, 
          
          and so on throughout the year. The first graph is the significant wave height. The second graph is the 
          
          energy period. The third graph is the power output related to these two inputs, multiplied by the 
          
          number of devices in the array.

          Variables
          ----------
          yearly_binned_swh: is a list of lists. each inner list is every SWH of every month in the whole 
          
          dataset.

               i.e. [[SWH for every january][SWH for every february]...[SWH for every december]]
          
          yearly_binned_ep: is a list of lists. each inner list is every EP of every month in the whole dataset.
          
               i.e. [[EP for every january][EP for every february]...[EP for every december]]
          
          yearly_power_output: is a list of lists. each inner list is every power output of every month in the 
          
          whole dataset multiplied by the number of devices.
          
               i.e. [[power output for every january][power output for every february]...[power output for every december]]
          """
          
          swh_result = [[item for sublist in zip(*sublists)for item in sublist] for sublists in zip(*self.full_swh_list)]
          ep_result = [[item for sublist in zip(*sublists) for item in sublist] for sublists in zip(*self.full_ep_list)]
          yearly_binned_swh = []
          yearly_binned_ep = []
          for b in range(0, len(swh_result)): 
               yearly_bin_swh = [self.bin_data(val, self.reference_swh_list) for val in swh_result[b]]
               yearly_bin_ep = [self.bin_data(val, self.reference_ep_list)for val in ep_result[b]]
               yearly_binned_swh.append(yearly_bin_swh)
               yearly_binned_ep.append(yearly_bin_ep)
         
          yearly_power_output=[]
          current_yearly_power=[]
          for k in range(len(yearly_binned_swh)): 
               now_swh = yearly_binned_swh[k]
               now_ep=yearly_binned_ep[k]
               for i in range(len(now_swh)):
                    for j in range(len(self.power_matrix)):
                         if now_swh[i] ==self.power_matrix[j][0] and now_ep[i] == self.power_matrix[j][1]:
                              current_yearly_power.append((self.power_matrix[j][2])*self.multiplier)
               yearly_power_output.append(current_yearly_power)
               current_yearly_power=[]

          data_year_swh = []
          for i, box in enumerate(swh_result):
               for value in box:
                    data_year_swh.append({"Category":i+1, "Value": value})
          df = pd.DataFrame(data_year_swh)

          if 'SWH' in self.plot:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Category', y='Value', data=df)
            plt.title('Boxplot of Significant Wave Height for all years')
            plt.xlabel('Month')
            plt.ylabel('Significant Wave Height (m)')
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout
            plt.show()
          else:
               pass

          data_year_ep = []
          for i, box in enumerate(ep_result):
               for value in box:
                    data_year_ep.append({"Category":i+1, "Value": value})
          df = pd.DataFrame(data_year_ep)

          if 'EP' in self.plot:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Category', y='Value', data=df)
            plt.title('Boxplot of Energy Period for all years')
            plt.xlabel('Month')
            plt.ylabel('Energy Period (s)')
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout
            plt.show()
          else:
               pass

          data_year_power = []
          for i, box in enumerate(yearly_power_output):
               for value in box:
                    data_year_power.append({"Category":i+1, "Value": value})
          df = pd.DataFrame(data_year_power)

          if 'power' in self.plot:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Category', y='Value', data=df)
            plt.title('Boxplot of Power Output for all years')
            plt.xlabel('Month')
            plt.ylabel('Power Output (kW)')
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout
            plt.show()
          else:
               pass

if __name__ == "__main__":

     wave_profile = WaveProfileGenerator(
     file_dir=os.path.join(WAVE_DATA_DIR, 'Hobuck_Beach_MEA_Data'),
     model = 'd',
     outage_month = 11,
     outage_length = 13,
     outage_starttime = 13,
     number_of_devices = 2,
     start_year=1979,
     end_year=2010,
     plot = ['SWH', 'EP','power', 'energy']
     )

     data = wave_profile.get_wave_data_from_upload()
     interpolate = wave_profile.generate_interpolated_data()
     power = wave_profile.read_power_matrix()
     inputs = wave_profile.validate_inputs()
     random_sample = wave_profile.generate_random_sample()
     binned_values = wave_profile.find_binned_values()
     sample_plots = wave_profile.generate_sample_plots()
     all_data = wave_profile.generate_all_data()
     all_plots = wave_profile.generate_all_plots()
     yearly_boxplot = wave_profile.generate_full_year_boxplot()
   