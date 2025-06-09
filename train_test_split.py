"""Module performs:
-train test split
-imputation of missing numeric variables
-standardization of numeric variables
"""

import argparse
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser(description="Optional variables with default fallback")

# Add optional file path and target arguments
parser.add_argument('clean_data_file_path', nargs='?', default='bin/cleaned_and_joined.csv',
                    help='Optional input file path (default: bin/cleaned_and_joined.csv)')
parser.add_argument('target', nargs='?', default='Beta_winsorized',
                    help='Optional target variable name (default: Beta_winsorized)')
parser.add_argument('model_excluded_variables', nargs='?', default='model_excluded_variables.yaml',
                    help='''Optional yaml file path with list of variables to exclude
                    (default: model_excluded_variables.yaml)''')
parser.add_argument('cleaned_variables_path', nargs='?', default='cleaned_variables.yaml',
                    help='Optional cleaned variables file path (default: cleaned_variables.yaml)')
parser.add_argument('X_train_path', nargs='?', default='bin/X_train.csv',
                    help='Optional output filename (default: bin/x_train.csv)')
parser.add_argument('y_train_path', nargs='?', default='bin/y_train.csv',
                    help='Optional output filename (default: bin/y_train.csv)')
parser.add_argument('X_test_path', nargs='?', default='bin/X_test.csv',
                    help='Optional output filename (default: bin/X_test.csv)')
parser.add_argument('y_test_path', nargs='?', default='bin/y_test.csv',
                    help='Optional output filename (default: bin/y_test.csv)')

args = parser.parse_args()

# Read in the cleaned data
data = args.clean_data_file_path
df = pd.read_csv(data)

# Read in variables for later processing
model_excluded_variables_path = args.model_excluded_variables
cleaned_variables_path = args.cleaned_variables_path

with open(model_excluded_variables_path, 'r', encoding='utf-8') as file:
    model_excluded_variables = yaml.safe_load(file)

with open(cleaned_variables_path, 'r', encoding='utf-8') as file:
    cleaned_variables = yaml.safe_load(file)

# Update cleaned variables list to not include excluded variables
numeric_variables = cleaned_variables['numeric_variables']
numeric_variables = [variable for variable in numeric_variables
                     if variable not in model_excluded_variables]

# Filter data to remove observations missing target variable
target = args.target
df = df.dropna(subset=target)

# Filter data to remove excluded variables
X = df.drop(model_excluded_variables + [target], axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, random_state=0)

# Impute missing values for the numeric variables using the mean values
imputer = SimpleImputer(strategy='mean')

imputer.fit(X_train[numeric_variables])

X_train[numeric_variables] = imputer.transform(X_train[numeric_variables])
X_test[numeric_variables] = imputer.transform(X_test[numeric_variables])

# Scale values for the numeric variables
scaler = StandardScaler()
scaler.fit(X_train[numeric_variables])

X_train[numeric_variables] = scaler.transform(X_train[numeric_variables])
X_test[numeric_variables] = scaler.transform(X_test[numeric_variables])

# Write train_test_split files
X_train_path = args.X_train_path
y_train_path = args.y_train_path
X_test_path = args.X_test_path
y_test_path = args.y_test_path

X_train.to_csv(X_train_path, index=False)
y_train.to_csv(y_train_path, index=False)
X_test.to_csv(X_test_path, index=False)
y_test.to_csv(y_test_path, index=False)
