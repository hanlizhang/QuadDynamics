import pandas as pd

# List the file names
file_names = [
    "circle_traj_1.csv",
    "circle_traj_2.csv",
    "circle_traj_3.csv",
    "circle_traj_4.csv",
    "circle_traj_5.csv",
    "circle_traj_6.csv",
    "circle_traj_7.csv",
    "circle_traj_8.csv",
    "circle_traj_9.csv",
    "circle_traj_10.csv",
]

# Initialize an empty DataFrame to store the combined data
combined_data = pd.DataFrame()

# Loop through each file and concatenate it to the combined_data DataFrame
for file_name in file_names:
    data = pd.read_csv(file_name)  # Read the CSV file into a DataFrame
    combined_data = pd.concat([combined_data, data], ignore_index=True)

# Save the combined data to a new CSV file
combined_data.to_csv("const1_cir5.csv", index=False)
