import pandas as pd
import multiprocessing

# Create a DataFrame with two sets of values ID and Value
data_df = pd.DataFrame({
    'ID': range(1, 10001),
    'Value': range(3, 20002, 2)  # Generate 10000 odd numbers starting from 3
})

# Define a function to calculate the sum of a vector
def calculate_sum(values):
    total_sum = sum(values)
    return total_sum

# Split the 'Value' column into chunks
chunk_size = 1000
value_chunks = [data_df['Value'][i:i+chunk_size] for i in range(0, len(data_df['Value']), chunk_size)]

# Create a Pool of 4 worker processes, this is required by multiprocessing
pool = multiprocessing.Pool(processes=4)

# Map the calculate_sum function to each chunk of data in parallel
results = pool.map(calculate_sum, value_chunks)

# Close the pool to free up resources, if the pool won't be used further
pool.close()

# Combine the partial results to get the total sum
total_sum = sum(results)

# Compute the mean by dividing the total sum by the total length of the column 'Value'
mean_value = total_sum / len(data_df['Value'])

# Print the mean value
print(mean_value)

