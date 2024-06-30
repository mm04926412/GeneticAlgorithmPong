import pandas as pd
from sklearn.utils import resample

# Load the dataset
df = pd.read_csv("playerOutput.csv")

# Check the structure of the dataframe
print(df.head())

# Define the column names
col1 = 'leftHeld'
col2 = 'RightHeld'

# Count the occurrences of each combination
combination_counts = df.groupby([col1, col2]).size().reset_index(name='counts')
print(combination_counts)

# Define the desired proportion for each combination
desired_proportion = 1  # 4x the data across all categories

# Calculate the number of samples for each combination to achieve the desired proportion
total_samples = 5000
desired_counts = int(total_samples * desired_proportion)

# Resample each combination to the desired count
def resample_combination(df, col1_value, col2_value, n_samples,invert=False):
    subset = df[(df[col1] == col1_value) & (df[col2] == col2_value)]
    if invert:
        subset[col1] = ~subset[col1]
        subset[col2] = ~subset[col2]
    if len(subset) > n_samples:
        return resample(subset, replace=False, n_samples=n_samples, random_state=42)
    else:
        return resample(subset, replace=True, n_samples=n_samples, random_state=42)

# Create a balanced dataset
# Not moving is basically pointless so we only consider active control states
balanced_df = pd.concat([
    resample_combination(df, True, False, desired_counts),
    resample_combination(df, False, True, desired_counts),
])

# Shuffle the balanced dataset
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the balanced dataset to a new CSV file
balanced_df.to_csv("balanced_playerOutput.csv", index=False)