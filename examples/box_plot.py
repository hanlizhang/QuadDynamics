import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
data = pd.read_csv(
    "cost_for_diff_rad_wind1.csv", index_col=0
)  # Assuming the first row contains the radius values

# Transpose the data to have radii as columns and costs as rows
# data_transposed = data.T

# Plot the box plots
data.plot(kind="box")
plt.xlabel("Radius")
plt.ylabel("Cost")
plt.title("Box Plots for Each Radius")
plt.show()
