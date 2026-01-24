# for data manipulation
import pandas as pd
import numpy as np
import sklearn
Random_State = 42

# for creating a folder
import os

#for preparing preprocessor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, RobustScaler

# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split

# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder

# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

#for cleaning the texts
import re
from huggingface_hub import HfApi, CommitOperationAdd

# Define constants for the dataset and output paths
api = HfApi(token = os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/samdurai102024/predictive-maintenance-be/engine_data.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# List of original column names
columns_to_rename = [
    "Engine rpm",
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "lub oil temp",
    "Coolant temp",
    "Engine Condition"
]

# Create rename mapping (space → underscore)
rename_mapping = {col: col.replace(" ", "_") for col in columns_to_rename}

# Apply renaming
df = df.rename(columns=rename_mapping)

# Verify
print(df.columns)

# defined X predictors and y target datasets
target = "Engine_Condition"

# Split X and y with correct shapes
X = df.loc[:, df.columns != "Engine_Condition"].copy()
y = df.loc[:, ["Engine_Condition"]].copy()  # dataframe

# Perform train-validation-test split
# First variation, use one with stratify with 60:20:20 splitting and scaling
# splitting the data for 60:20:20 ratio between train, validation and test sets
# stratify ensures the training, validation and test sets have a similar distribution of the response variable
# Let's split data into temporary and test - 2 parts
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size = 0.2, random_state = Random_State, stratify = y, shuffle = True)

# First variation, use one with stratify with 60:20:20 splitting and scaling
# splitting the data for 60:20:20 ratio between train, validation and test sets
# stratify ensures the training, validation and test sets have a similar distribution of the response variable
# then we split the temporary set into train and validation
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size = 0.25, random_state = Random_State, stratify = y_temp, shuffle = True)

# Convert target values (not column names) to integer
y_train = y_train.astype(int)
y_val = y_val.astype(int)
y_test = y_test.astype(int)

#Feature Engineering
#1) Adding New categorical Feature rpm_category for Engine_rpm
#Risk-Based Categories
# -----------------------------
# Risk-Based Categories (TRAIN-derived thresholds)
# -----------------------------
def rpm_category(rpm):
    if rpm <= 742:
        return 'High Risk'
    elif rpm <= 886:
        return 'Moderate Risk'
    else:
        return 'Lower Risk'

# Apply consistently to all splits
df['rpm_category']       = df['Engine_rpm'].apply(rpm_category)
X_train['rpm_category'] = X_train['Engine_rpm'].apply(rpm_category)
X_val['rpm_category']   = X_val['Engine_rpm'].apply(rpm_category)
X_test['rpm_category']  = X_test['Engine_rpm'].apply(rpm_category)

#2)Adding New categorical Feature oil_pressure_category for Lub_oil_pressure
#Risk-Based Categories
# -----------------------------
# Risk-Based Categories (TRAIN-derived thresholds)
# -----------------------------
def oil_pressure_category(opc):
    if opc <= 2.5:
        return 'Low Risk'
    elif opc <= 5.0:
        return 'Mid Risk'
    else:
        return 'High Risk'

# Apply consistently to all splits
df['oil_pressure_category']       = df['Lub_oil_pressure'].apply(oil_pressure_category)
X_train['oil_pressure_category'] = X_train['Lub_oil_pressure'].apply(oil_pressure_category)
X_val['oil_pressure_category']   = X_val['Lub_oil_pressure'].apply(oil_pressure_category)
X_test['oil_pressure_category']  = X_test['Lub_oil_pressure'].apply(oil_pressure_category)

#3) Adding New categorical Feature fuel_pressure_category for Fuel_pressure
#Risk-Based Categories
# -----------------------------
# Risk-Based Categories (TRAIN-derived thresholds)
# -----------------------------
def fuel_pressure_category(fp):
    if fp <= 4.0:
        return 'Low Risk'
    elif fp <= 5.1:
        return 'Mid Risk'
    elif fp <= 6.2:
        return 'High Risk'
    else:
        return 'Very High Risk'

# Apply consistently to all splits
df['fuel_pressure_category']       = df['Fuel_pressure'].apply(fuel_pressure_category)
X_train['fuel_pressure_category'] = X_train['Fuel_pressure'].apply(fuel_pressure_category)
X_val['fuel_pressure_category']   = X_val['Fuel_pressure'].apply(fuel_pressure_category)
X_test['fuel_pressure_category']  = X_test['Fuel_pressure'].apply(fuel_pressure_category)

#4)Adding New categorical Feature coolant_pressure_category for Coolant_pressure
#Risk-Based Categories
# -----------------------------
# Risk-Based Categories (TRAIN-derived thresholds)
# -----------------------------
def coolant_pressure_category(cp):
    if cp <= 3.1:
        return 'Normal-to-Elevated'
    elif cp <= 3.8:
        return 'Critical Pressure'
    else:
        return 'Relief / Saturation Regime'

# Apply consistently to all splits
df['coolant_pressure_category']       = df['Coolant_pressure'].apply(coolant_pressure_category)
X_train['coolant_pressure_category'] = X_train['Coolant_pressure'].apply(coolant_pressure_category)
X_val['coolant_pressure_category']   = X_val['Coolant_pressure'].apply(coolant_pressure_category)
X_test['coolant_pressure_category']  = X_test['Coolant_pressure'].apply(coolant_pressure_category)

#5)Adding New categorical Feature fuel_oil_risk for Fuel_pressure and Lub_oil_temp
#Risk-Based Categories
# -----------------------------
# Risk-Based Categories (TRAIN-derived thresholds)
# -----------------------------
def add_fuel_oil_risk(df):
    df = df.copy()

    fp = df["Fuel_pressure"]
    ot = df["lub_oil_temp"]

    df["fuel_pressure_oil_temp"] = "Low Risk"

    df.loc[(fp > 5.1) & (ot < 77.2), "fuel_pressure_oil_temp"] = "High Risk"
    df.loc[(fp > 5.1) & (ot >= 77.2), "fuel_pressure_oil_temp"] = "Mid Risk"
    df.loc[(fp <= 5.1) & (ot < 77.2), "fuel_pressure_oil_temp"] = "Mid Risk"

    return df

# Apply consistently to all splits
df       = add_fuel_oil_risk(df)
X_train = add_fuel_oil_risk(X_train)
X_val   = add_fuel_oil_risk(X_val)
X_test  = add_fuel_oil_risk(X_test)

#6) Adding New categorical Feature coolant_temp_category for coolant temp
#Risk-Based Categories
# -----------------------------
# Risk-Based Categories (TRAIN-derived thresholds)
# -----------------------------
def coolant_temp_category(ct):
    if ct < 69.17:
        return "High"
    elif ct < 72.27:
        return "Elevated"
    elif ct < 80.55:
        return "Moderate"
    else:
        return "Lower"
# Apply consistently to all splits
df['coolant_temp_category']       = df['Coolant_temp'].apply(coolant_temp_category)
X_train['coolant_temp_category'] = X_train['Coolant_temp'].apply(coolant_temp_category)
X_val['coolant_temp_category']   = X_val['Coolant_temp'].apply(coolant_temp_category)
X_test['coolant_temp_category']  = X_test['Coolant_temp'].apply(coolant_temp_category)

# Loop all the columns in the dataset to convert object columns into categorical columns
for col in X_train.columns:
    if X_train[col].dtype == 'object':
        X_train[col] = pd.Categorical(X_train[col])
        X_val[col] = pd.Categorical(X_val[col])
        X_test[col] = pd.Categorical(X_test[col])

#Convert datasets into csv files
X_train.to_csv("X_train.csv",index=False)
X_val.to_csv("X_val.csv",index=False)
X_test.to_csv("X_test.csv",index=False)
y_train.to_csv("y_train.csv",index=False)
y_val.to_csv("y_val.csv",index=False)
y_test.to_csv("y_test.csv",index=False)

#Checking the files in the folder
def check_dataset_consistency(X_train, X_val, X_test, name_train="Train", name_val="Val", name_test="Test"):
    print("=== DATASET CONSISTENCY CHECK ===\n")

    # SHAPE CHECK
    print("SHAPE")
    print(f"{name_train}: {X_train.shape}")
    print(f"{name_val}:   {X_val.shape}")
    print(f"{name_test}:  {X_test.shape}\n")

    #COLUMN SET CHECK
    print("COLUMN SET DIFFERENCE")

    train_cols = set(X_train.columns)
    val_cols   = set(X_val.columns)
    test_cols  = set(X_test.columns)

    print("Missing in VAL:", train_cols - val_cols)
    print("Extra in VAL:", val_cols - train_cols)
    print()
    print("Missing in TEST:", train_cols - test_cols)
    print("Extra in TEST:", test_cols - train_cols)
    print()

    #ORDER CHECK
    print("COLUMN ORDER CHECK")
    if list(X_train.columns) == list(X_val.columns) == list(X_test.columns):
        print("Okay, ORDER MATCHES for all datasets\n")
    else:
        print("Not Okay, Column order mismatch!\n")

    #DTYPE CHECK
    print("DTYPE CHECK")
    for col in X_train.columns:
        dt_train = X_train[col].dtype
        dt_val = X_val[col].dtype
        dt_test = X_test[col].dtype
        if not (dt_train == dt_val == dt_test):
            print(f"Not Okay, Dtype mismatch in column '{col}': Train={dt_train}, Val={dt_val}, Test={dt_test}")
    print()

    print("CHECK ENDED")

check_dataset_consistency(X_train, X_val, X_test)

###Delete Old datasets
from huggingface_hub import HfApi, CommitOperationDelete, CommitOperationAdd

api = HfApi()

repo_id = "samdurai102024/predictive-maintenance-be"
#processed_folder = "processed_data/"   # our local folder with split files

#remove old processed files ONLY ---
tree = api.list_repo_tree(repo_id, repo_type="dataset")

delete_ops = []
for file in tree:
    path = file.path

    # Skip raw files (keep them always)
    if path.lower() in ["engine_data.csv", 'raw.scv']:
        continue

    # Delete only processed files (train/val/test splits)
    if any(x in path.lower() for x in [
        "train", "val", "test", "x_", "y_"
    ]):
        delete_ops.append(CommitOperationDelete(path_in_repo=path))

# Perform delete commit if needed
if delete_ops:
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=delete_ops,
        commit_message="Cleanup old processed dataset before upload"
    )
### end of deletion

#Checking data sets shape before upload
print("LOCAL SHAPES JUST BEFORE UPLOAD")
for f in ["X_train.csv", "X_val.csv", "X_test.csv", "y_train.csv", "y_val.csv", "y_test.csv"]:
    df = pd.read_csv(f)
    print(f, df.shape)

# New set upload and commmit changes
from huggingface_hub import HfApi, CommitOperationAdd

api = HfApi()

import os

base_path = "/content/"   # or wherever our new processed files are saved

files = [
    "X_train.csv", "X_val.csv", "X_test.csv",
    "y_train.csv", "y_val.csv", "y_test.csv"
]

for f in files:
    print("Checking:", f)
    print("CWD:", os.getcwd())
    print("Exists in CWD:", os.path.exists(f))
    print("Absolute path:", os.path.abspath(f))
    print("-" * 50)


#Define preprocessor funtion steps
# Define categorical and numerical column names (these are the original, raw column names)
categorical_cols = X_train.select_dtypes(include=['category', 'object']).columns
numerical_cols = X_train.select_dtypes(include=[np.number]).columns

#Define proprocessor function
def build_preprocessor(numerical_cols):
    """
    Builds a ColumnTransformer with:
    1. Explicit ordinal encodings for categorical risk features
    2. RobustScaler for numerical features
    3. This function ONLY defines the transformer
    4. Fitting happens inside Pipeline.fit()
    """

    ordinal_transformers = [
        (
            "rpm_category",
            OrdinalEncoder(
                categories=[["Lower Risk", "Moderate Risk", "High Risk"]],
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ),
            ["rpm_category"]
        ),
        (
            "oil_pressure_category",
            OrdinalEncoder(
                categories=[["Low Risk", "Mid Risk", "High Risk"]],
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ),
            ["oil_pressure_category"]
        ),
        (
            "fuel_pressure_category",
            OrdinalEncoder(
                categories=[["Low Risk", "Mid Risk", "High Risk", "Very High Risk"]],
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ),
            ["fuel_pressure_category"]
        ),
        (
            "coolant_pressure_category",
            OrdinalEncoder(
                categories=[
                    ["Normal-to-Elevated", "Relief / Saturation Regime", "Critical Pressure"]
                ],
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ),
            ["coolant_pressure_category"]
        ),
        (
            "fuel_pressure_oil_temp",
            OrdinalEncoder(
                categories=[["Low Risk", "Mid Risk", "High Risk"]],
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ),
            ["fuel_pressure_oil_temp"]
        ),
        (
            "coolant_temp_category",
            OrdinalEncoder(
                categories=[["Lower", "Moderate", "Elevated", "High"]],
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ),
            ["coolant_temp_category"]
        ),
    ]

    preprocessor = ColumnTransformer(
        transformers=ordinal_transformers + [
            ("num", RobustScaler(), numerical_cols)
        ],
        remainder="drop"
    )

    return preprocessor
###End of funtion build_preprocessor###

#Upload new files
files = ["X_train.csv","X_val.csv","X_test.csv","y_train.csv","y_val.csv","y_test.csv"]

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN missing")

api = HfApi(token=HF_TOKEN)

operations = []

for file_path in files:
    operations.append(
        CommitOperationAdd(
            path_in_repo=file_path.split("/")[-1],
            path_or_fileobj=file_path
        )
    )
#Commit Updated datasets after change
api.create_commit(
    repo_id="samdurai102024/predictive-maintenance-be",
    repo_type="dataset",
    operations=operations,
    commit_message="Updated datasets after change"
)

print("Files uploaded and committed successfully")
