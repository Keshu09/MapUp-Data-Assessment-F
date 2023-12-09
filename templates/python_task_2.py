import pandas as pd


def calculate_distance_matrix(df):
  distance_df = df.copy()

    # Pivot the DataFrame to create a distance matrix
    distance_matrix = distance_df.pivot(index='id_start', columns='id_end', values='distance')

    # Fill NaN values with 0, assuming 0 distance for missing routes
    distance_matrix = distance_matrix.fillna(0)

    # Make the matrix symmetric by taking the element-wise maximum between A to B and B to A
    distance_matrix = pd.DataFrame(np.maximum(distance_matrix.T.values, distance_matrix.values), 
                                    index=distance_matrix.columns, columns=distance_matrix.index)

    # Perform cumulative sum along the columns to get cumulative distances
    distance_matrix = distance_matrix.cumsum(axis=1)

    return distance_matrix


dataset_df = pd.read_csv('dataset-3.csv')

# Call the function with your DataFrame
distance_matrix_result = calculate_distance_matrix(dataset_df)

# Display the result
print(distance_matrix_result)



def unroll_distance_matrix(df):
    distance_matrix = distance_matrix.reset_index()

    # Melt the DataFrame to convert it to the long format
    unrolled_df = pd.melt(distance_matrix, id_vars='id_start', var_name='id_end', value_name='distance')

    # Exclude rows where id_start is the same as id_end
    unrolled_df = unrolled_df[unrolled_df['id_start'] != unrolled_df['id_end']]

    # Reset the index and drop the original index column
    unrolled_df = unrolled_df.reset_index(drop=True)

    return unrolled_df


distance_matrix_result = calculate_distance_matrix(dataset_df)

# Call the function with the distance matrix DataFrame
unrolled_df_result = unroll_distance_matrix(distance_matrix_result)

# Display the result
print(unrolled_df_result)

    


def find_ids_within_ten_percentage_threshold(df, reference_id):
     reference_rows = df[df['id_start'] == reference_value]

    # Calculate the average distance for the reference value
    average_distance = reference_rows['distance'].mean()

    # Calculate the lower and upper bounds for the 10% threshold
    lower_bound = average_distance - 0.1 * average_distance
    upper_bound = average_distance + 0.1 * average_distance

    # Filter the DataFrame for rows within the 10% threshold
    within_threshold = df[(df['distance'] >= lower_bound) & (df['distance'] <= upper_bound)]

    # Extract and sort the unique values from the id_start column
    result_list = sorted(within_threshold['id_start'].unique())

    return result_list


reference_value = 1  
result_list = find_ids_within_ten_percentage_threshold(df, reference_value)

# Display the result
print(result_list)







    

def calculate_toll_rate(df):
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    # Add new columns for each vehicle type and calculate toll rates
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        column_name = f'{vehicle_type}_rate'
        unrolled_df[column_name] = unrolled_df['distance'] * rate_coefficient

    return df

result_with_toll_rates = calculate_toll_rate(unrolled_df)

# Display the result
print(result_with_toll_rates)




def calculate_time_based_toll_rates(df):
    time_ranges_weekday = [(pd.to_datetime('00:00:00').time(), pd.to_datetime('10:00:00').time()),
                           (pd.to_datetime('10:00:00').time(), pd.to_datetime('18:00:00').time()),
                           (pd.to_datetime('18:00:00').time(), pd.to_datetime('23:59:59').time())]

    time_ranges_weekend = [(pd.to_datetime('00:00:00').time(), pd.to_datetime('23:59:59').time())]

    discount_factors_weekday = [0.8, 1.2, 0.8]
    discount_factor_weekend = 0.7

    # Create empty columns for start_day, start_time, end_day, and end_time
    df['start_day'] = ''
    df['start_time'] = pd.to_datetime('00:00:00').time()
    df['end_day'] = ''
    df['end_time'] = pd.to_datetime('23:59:59').time()

    # Iterate over unique (id_start, id_end) pairs
    for index, row in df.iterrows():
        # Assign start_day and end_day values (Monday to Sunday)
        df.at[index, 'start_day'] = 'Monday'
        df.at[index, 'end_day'] = 'Sunday'

        # Assign discount factors based on weekday or weekend
        if pd.to_datetime(row['start_day']).weekday() < 5:  # Weekday (Monday - Friday)
            discount_factors = discount_factors_weekday
        else:  # Weekend (Saturday and Sunday)
            discount_factors = [discount_factor_weekend]

        # Iterate over time ranges and assign discount factors
        for i, (start_time, end_time) in enumerate(time_ranges_weekday if discount_factors == discount_factors_weekday else time_ranges_weekend):
            mask = (row['start_time'] >= start_time) & (row['end_time'] <= end_time)
            #df.at[index, f'car_rate_{i+1}'] = row['car_rate'] * discount_factors[i] if mask else 0

    return df
result_with_toll_rates = calculate_toll_rate(df)

result_with_time_based_rates = calculate_time_based_toll_rates(result_with_toll_rates)

# Display the result
print(result_with_time_based_rates)


  
