import pandas as pd
import numpy as np

def generate_car_matrix(dataset):
     df = pd.read_csv(dataset) 
     result_df = df.pivot(index='id_1', columns='id_2', values='car')

    # Set the diagonal values to 0 using numpy.fill_diagonal
     np.fill_diagonal(result_df.values, 0)

    # Reset the index to make 'id_2' values as columns
     result_df = result_df.reset_index()
    

     return result_df

dataset_path = 'dataset-1.csv'
generate_car_matrix(dataset_path)

    # Write your logic here
1.Read the dataset:
- generate_car_matrix function takes "dataset" as an argument.
- read the csv file into data frame using df.
2.Pivot the dataframe.
- according to rule make id_1 as index and id_2 as colunm.
- and daigonal set to 0 using np.fill_diagonal().
3.Return the result
- using return result_df

    


def get_type_count(df)->dict:
    # Write your logic here 
    df['car_type'] = pd.cut(df['car'], bins=[float('-inf'), 15, 25, float('inf')],
                            labels=['low', 'medium', 'high'], right=False)
    
    
    type_counts = df['car_type'].value_counts().to_dict()
    
    sorted_type = dict(sorted(type_counts.items()))
    
    return sorted_type

dataset_path = 'dataset-1.csv'
df = pd.read_csv(dataset_path)
get_bus_indexes(df)




def get_bus_indexes(df):
   mean_bus_value = df['bus'].mean()

    # Identify indices where 'bus' values are greater than twice the mean
    bus_indexes = df[df['bus'] > 2 * mean_bus_value].index.tolist()

    # Sort the indices in ascending order
    bus_indexes.sort()

    return bus_indexes


dataset_path = 'dataset-1.csv'
df = pd.read_csv(dataset_path)
get_bus_indexes(df)






def filter_routes(df):
    # Filter rows where the average of 'truck' values is greater than 7
    filtered_df = df.groupby('route')['truck'].mean() > 7

    # Get the routes that meet the condition
    selected_routes = filtered_df[filtered_df].index.tolist()

    # Sort the list of routes
    selected_routes.sort()

    return selected_routes

dataset_path = 'dataset-1.csv'
df = pd.read_csv(dataset_path)
filter_routes(df)



def multiply_matrix(df):
    # Create a copy of the input DataFrame to avoid modifying the original
    df = pd.read_csv("C:/Users/Admin/Desktop/dataset-1.csv")

    # Apply the specified logic to each value in the DataFrame
    modified_df = df.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)

    # Round the resulting values to 1 decimal place
    modified_df = modified_df.round(1)

    return modified_df

multiply_matrix(result_matrix)

:
    


def time_check(df) :
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Create a new column 'day_of_week' to represent the day of the week (Monday=0, Sunday=6)
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    # Create a mask for correct timestamps (covering a full 24-hour period and all 7 days of the week)
    correct_timestamps_mask = (
        (df['timestamp'].dt.time == pd.Timestamp('00:00:00').time()) &  # Check for 12:00:00 AM
        (df['timestamp'].dt.hour == 23) & (df['timestamp'].dt.minute == 59) & (df['timestamp'].dt.second == 59) &  # Check for 11:59:59 PM
        (df['day_of_week'].nunique() == 7)  # Check for all 7 days of the week
    )

    # Group by (id, id_2) and check if any timestamp is incorrect for each group
    result_series = ~df.groupby(['id', 'id_2'])['timestamp'].transform(lambda x: all(correct_timestamps_mask.loc[x.index]))

    return result_series


dataset_df = pd.read_csv('C:/Users/Admin/Desktop/dataset-2.csv')

# Call the function with your DataFrame
verification_result = verify_timestamps(dataset_df)

# Display the result
print(verification_result)

