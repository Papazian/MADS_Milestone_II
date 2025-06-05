"""
Module that reads in the mortgage data file and joins it with the treasury yields data file.
Data is cleaned and prepped for further analysis including:

Null Value Handling
Categorical Variable One Hot Encoding
"""

import argparse
import yaml
import numpy as np
import pandas as pd
import scipy as sp

parser = argparse.ArgumentParser(description="Optional filename with default fallback")

# Add optional file path arguments
parser.add_argument('mortgage_file_path', nargs='?', default='bin/nsmo_v50_1321_puf.csv',
                    help='Optional input filename (default: bin/nsmo_v50_1321_puf.csv)')
parser.add_argument('yield_file_path', nargs='?', default='bin/DGS30.csv',
                    help='Optional input filename (default: bin/DGS30.csv)')
parser.add_argument('out_file_path', nargs='?', default='bin/cleaned_and_joined.csv',
                    help='Optional input filename (default: bin/cleaned_and_joined.csv)')

args = parser.parse_args()

def read_file(file_path, url):
    """
    Function accepts a csv file path and backup URL. Attempts to read in the csv
    if the read fails, it attempts to obtain the file from the source URL. Then
    prompts the user to save the file at the provided path.
    """

    print(f"Using file: {args.mortgage_file_path}")

    try:
        file_df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f'File not found at {file_path}, attempting to obtain from {url}')
        file_df = pd.read_csv(url)

        # Prompt user with option to save file
        user_response = input(f'Source file downloaded. Save at {file_path} Y/N:')
        if user_response.upper() == 'Y':
            file_df.to_csv(file_path, index=False)

    return file_df

# Read in the raw CSV data, and then print the count of observations and variables
# https://www.fhfa.gov/sites/default/files/2024-06/nsmo_v50_1321_puf.csv

mortgage_file_path = args.mortgage_file_path
MORTGAGE_URL = 'https://www.fhfa.gov/sites/default/files/2024-06/nsmo_v50_1321_puf.csv'
raw_df = read_file(mortgage_file_path, MORTGAGE_URL)

yield_file_path = args.yield_file_path
YIELD_URL = ('https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23ebf3fb&chart_type'
             '=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode'
             '=fred&recession_bars'
             '=on&txtcolor=%23444444&ts=12&tts=12&width=1140&nt=0&thu=0&trc=0&show_legend'
             '=yes&show_axis_titles=yes&show_tooltip=yes&id=DGS30&scale=left&cosd=1977-02-15&coed'
             '=2025-05-30&line_color=%230073e6&link_values=false&line_style=solid&mark_type'
             '=none&mw'
             '=3&lw=3&ost=-99999&oet=99999&mma=0&fml=a&fq=Daily&fam=avg&fgst=lin&fgsnd'
             '=2020-02-01&line_index'
             '=1&transformation=lin&vintage_date=2025-06-02&revision_date=2025-06-02&nd=1977-02-15')
treasury_yields_df = read_file(yield_file_path, YIELD_URL)

# Load YAML files containing metadata into Python as dictionaries

# Load the labels of each variable into a dictionary
# https://www.fhfa.gov/sites/default/files/2024-06/nsmo_v50_labels.sas

with open('variable_labels.yaml', 'r', encoding='utf-8') as file:
    variable_labels_dict = yaml.safe_load(file)

# Load the the format of each variable into a dictionary
# https://www.fhfa.gov/sites/default/files/2024-06/nsmo_v50_labels.sas

with open('variable_formats.yaml', 'r', encoding='utf-8') as file:
    variable_formats_dict = yaml.safe_load(file)

# Load the categories for every categorical variable (exclude null categories) into a dictionary
# https://www.fhfa.gov/sites/default/files/2024-06/nsmo_v50_formats.sas

with open('categorical_variables_categories.yaml', 'r', encoding='utf-8') as file:
    categorical_variables_categories_dict = yaml.safe_load(file)

# Create a set of all variable formats

variable_formats_set = set(variable_formats_dict.values())

# Create a list of the categorical variables and a list of the numeric variables

categorical_variables = []
numeric_variables = []

categorical_variable_formats = set(categorical_variables_categories_dict.keys())
numeric_variable_formats = variable_formats_set - categorical_variable_formats

for col in raw_df.columns:
    if variable_formats_dict[col] in categorical_variable_formats:
        categorical_variables.append(col)
    elif variable_formats_dict[col] in numeric_variable_formats:
        numeric_variables.append(col)
    else:
        print("Error in bifurcation")

# Clean data by converting negative values and "." values (representing missing values)
# into null values (i.e., NaN)

excluded_columns = ['PSTATFM', 'rate_spread']

for col in raw_df.columns:
    if variable_formats_dict[col] in excluded_columns or col in excluded_columns:
        continue
    raw_df.loc[raw_df[col] < 0, col] = np.nan
    raw_df.loc[raw_df[col] == ".", col] = np.nan

# Create new columns based upon the observation date, which will be used in join to the main data

treasury_yields_df['observation_date'] = pd.to_datetime(treasury_yields_df['observation_date'])
treasury_yields_df['observation_year'] = treasury_yields_df['observation_date'].dt.year
treasury_yields_df['observation_month'] = treasury_yields_df['observation_date'].dt.month

# Calculate the average Treasury Yields over each Year and Month

average_treasury_yields_df = (treasury_yields_df.groupby(['observation_year', 'observation_month'])
                              ['DGS30'].mean())
average_treasury_yields_df = average_treasury_yields_df.to_frame().reset_index()

# Left join the Treasury Yields to the raw Mortgage Origination data using the composite keys of
# year and month

merged_df = pd.merge(left=raw_df,
                     right=average_treasury_yields_df,
                     left_on=['open_year', 'open_month'],
                     right_on=['observation_year', 'observation_month'],
                     how='left')

# Drop the redundant columns of composite keys used in join

merged_df = merged_df.drop(['observation_year', 'observation_month'], axis=1)

# Derivation of Beta, which is a new variable

NOTES = """
Re = Rf + β(Rm − Rf)
Re - Rf = β(Rm − Rf)
(Re - Rf) = β(Rm − Rf)
(Re - Rf) / (Rm − Rf) = β
β = (Re - Rf) / (Rm − Rf)

'rate_spread' = Re - 'PMMS'
'rate_spread' + 'PMMS' = Re 
Re = 'rate_spread' + 'PMMS'

β = ('rate_spread' + 'PMMS' - Rf) / (Rm − Rf)
β = ('rate_spread' + 'PMMS' - 'treasury_yield') / ('PMMS' − 'treasury_yield')
"""

# Calculation: Beta = ('rate_spread' + 'pmms' - 'treasury_yield') / ('pmms' − 'treasury_yield')

merged_df['Beta'] = ((merged_df['rate_spread'] + merged_df['pmms'] - merged_df['DGS30']) /
                     (merged_df['pmms'] - merged_df['DGS30']))

# Winsorize the Beta values to reduce the influence of outliers
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.winsorize.html

merged_df['Beta_winsorized'] = sp.stats.mstats.winsorize(merged_df['Beta'], limits=[0.05,0.05])

# Create dummy variables for each category for each categorical variable

processed_df = pd.get_dummies(merged_df, columns=categorical_variables)

# Remove the ".0" in many of the dummy variable due to the columns in the raw data being floats

new_columns_list = []
for col in processed_df.columns:
    new_col = col.replace(".0", "")
    new_columns_list.append(new_col)

processed_df.columns = new_columns_list

# Writing cleaned file
out_file_path = args.out_file_path
processed_df.to_csv(out_file_path, index=False)

# Writing variable YAML file
# Converting categorical and numeric variables to tuples for yaml compatability
yaml_data = {'categorical_variables': dict(zip(categorical_variables, categorical_variable_formats)),
             'numeric_variables': dict(zip(numeric_variables, numeric_variable_formats))}
with open('cleaned_variables.yaml', 'w', encoding='utf-8') as file:
    yaml.dump(yaml_data, file)
