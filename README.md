# Exno:1
Data Cleaning Process

# AIM
To read the given data and perform data cleaning and save the cleaned data to a file.

# Explanation
Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect ,incompleted , irrelevant , duplicated or improperly formatted. Data cleaning is not simply about erasing data ,but rather finding a way to maximize datasets accuracy without necessarily deleting the information.

# Algorithm
STEP 1: Read the given Data

STEP 2: Get the information about the data

STEP 3: Remove the null values from the data

STEP 4: Save the Clean data to the file

STEP 5: Remove outliers using IQR

STEP 6: Use zscore of to remove outliers

# Coding and Output
 # Step 1: Import Required Libraries

import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Read the Dataset

# Replace with your actual CSV file
df1 = pd.read_csv('Data_set.csv')
df1.head()

# Step 3: Dataset Information
df1.info()
df1.describe()

# Step 4: Handling Missing Values
# Check Null Values
df1.isnull()
df1.isnull().sum()

# Fill Missing Values with 0
df1_fill_0 = df1.fillna(0)
df1_fill_0

# Forward Fill
df1_ffill = df1.ffill()
df1_ffill

# Backward Fill
df1_bfill = df1.bfill()
df1_bfill

# Fill with Mean (Numerical Column Example)
df1['rating'] = df1['rating'].fillna(df1['rating'].mean())
df1

# Drop Missing Values
df1_dropna = df1.dropna()
df1_dropna

#Step 5: Save Cleaned Data
df1_dropna.to_csv('Data_setnew.csv', index=False)

# OUTLIER DETECTION
# Step 6: IQR Method (Using Data_setnew Dataset)
ra = pd.read_csv('Data_setnew.csv')
ra.head()
ra.info()
ra.describe()   

#Boxplot for Outlier Detection
sns.boxplot(x=ra['rating'])
plt.show()

# Calculate IQR
Q1 = ra['rating'].quantile(0.25)
Q3 = ra['rating'].quantile(0.75)
IQR = Q3 - Q1
print("IQR:", IQR)

# Detect Outliers
outliers_iqr = ra[
    (ra['rating'] < (Q1 - 1.5 * IQR)) |
    (ra['rating'] > (Q3 + 1.5 * IQR))
]
outliers_iqr

# Remove Outliers
ra_cleaned = ra[
    ~((ra['rating'] < (Q1 - 1.5 * IQR)) |
      (ra['rating'] > (Q3 + 1.5 * IQR)))
]
ra_cleaned

# Step 7: Z-Score Method

data = [1,12,15,18,21,24,27,30,33,36,39,42,45,48,51,
        54,57,60,63,66,69,72,75,78,81,84,87,90,93]

df_z = pd.DataFrame(data, columns=['values'])
df_z

# Calculate Z-Scores
z_scores = np.abs(stats.zscore(df_z))
z_scores

# Detect Outliers
threshold = 1.5
outliers_z = df_z[z_scores > threshold]
print("Outliers:")
outliers_z

# Remove Outliers
df_z_cleaned = df_z[z_scores <= threshold]
df_z_cleaned

<img width="959" height="537" alt="image" src="https://github.com/user-attachments/assets/85412d07-9a55-481d-bdec-4a9ad738f528" />

<img width="947" height="523" alt="image" src="https://github.com/user-attachments/assets/0cb4a368-f134-4d46-8c10-7133119a0e7c" />

<img width="958" height="534" alt="image" src="https://github.com/user-attachments/assets/4573b1b8-2786-4d1f-94f6-9d776cbc1714" />

<img width="956" height="532" alt="image" src="https://github.com/user-attachments/assets/00c1c060-70ce-4773-9dd3-65db52b252e4" />

<img width="959" height="530" alt="image" src="https://github.com/user-attachments/assets/73038542-65bb-4509-9564-7036cc35b224" />

<img width="959" height="536" alt="image" src="https://github.com/user-attachments/assets/810f98f0-d552-47a2-a903-11abea2daab1" />


[ilovepdf_merged.pdf](https://github.com/user-attachments/files/25248981/ilovepdf_merged.pdf)

# Result
The given data has been cleaned successfully using data cleaning process.
