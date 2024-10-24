import pandas as pd
import numpy as np
import datetime


def calculate_distance_matrix(df) -> pd.DataFrame:
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame): DataFrame containing 'id_start', 'id_end', and 'distance' columns.

    Returns:
        pandas.DataFrame: Distance matrix with unique IDs as both rows and columns.
    """
    #df = pd.read_csv('dataset-2.csv')
    if 'id_start' not in df.columns or 'id_end' not in df.columns or 'distance' not in df.columns:
        raise ValueError("Dataframe must contain 'id_start', 'id_end' and 'distance' column.")
    unique_ids = pd.unique(df[['id_start' , 'id_end']].values.ravel('K'))
    distance_matrix = pd.DataFrame(np.inf, index =  unique_ids, columns = unique_ids)
    np.fill_diagonal(distance_matrix.values,0)
    for  _,row in df.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        distance = row['distance']
        distance_matrix.loc[id_start, id_end] = min(distance_matrix.loc[id_start, id_end], distance)
        distance_matrix.loc[id_end, id_start] = min(distance_matrix.loc[id_end, id_start], distance)
    for k in unique_ids:
        for i in unique_ids:
            for j in unique_ids:
                distance_matrix.loc[i, j] = min(distance_matrix.loc[i, j],distance_matrix.loc[i, k] + distance_matrix.loc[k, j])
    return distance_matrix
df = pd.read_csv(r"C:\Users\sikha\OneDrive\Desktop\Answers\MapUp-DA-Assessment-2024\datasets\dataset-2.csv")
results = calculate_distance_matrix(df)
print(results)

def unroll_distance_matrix(distance_matrix)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here
    unrolled_data = []
    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            if id_start != id_end:
                distance = distance_matrix.loc[id_start, id_end]
                if distance < np.inf:
                    unrolled_data.append({
                        'id_start' : id_start,
                        'id_end' : id_end,
                        'distance' : distance

                    })
    unrolled_df = pd.DataFrame(unrolled_data)
    return unrolled_df
unrolled_df = unroll_distance_matrix(results)
print(unrolled_df)


def find_ids_within_ten_percentage_threshold(unrolled_df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here
    reference_average_distance = unrolled_df[unrolled_df['id_start'] == reference_id]['distance'].mean()
    lower_threshold = reference_average_distance * 0.9
    upper_threshold = reference_average_distance * 1.1
    id_avg_distances = unrolled_df.groupby('id_start')['distance'].mean()
    ids_within_threshold = id_avg_distances[
        (id_avg_distances >= lower_threshold) & (id_avg_distances <= upper_threshold)
    ].index.tolist()
    return sorted(ids_within_threshold)

reference_id = 1001479
result = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id)
print(result)


def calculate_toll_rate(unrolled_df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here
    toll_rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    for vehicle, rate in toll_rates.items():
        unrolled_df[vehicle] = unrolled_df['distance'] * rate

    return unrolled_df
data = {
    'id_start': [1001400] * 10,
    'id_end': [
        1001402, 1001404, 1001406, 1001408, 1001410, 
        1001412, 1001414, 1001416, 1001418, 1001420
    ],
    'distance': [
        9.7, 29.9, 45.9, 67.6, 78.7,
        94.3, 112.6, 131.6, 151.3, 182.4
    ]
}

unrolled_df = pd.DataFrame(data)
result_df = calculate_toll_rate(unrolled_df)
print(result_df)


def calculate_time_based_toll_rates(unrolled_df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    time_intervals = [
        (datetime.time(0, 0, 0), datetime.time(10, 0, 0)),
        (datetime.time(10, 0, 0), datetime.time(18, 0, 0)),
        (datetime.time(18, 0, 0), datetime.time(23, 59, 59))
    ]
    
    weekday_discounts = [0.8, 1.2, 0.8]
    weekend_discount = 0.7
    expanded_data = []

    for _, row in unrolled_df.iterrows():
        for day in days_of_week:
            is_weekend = day in ['Saturday', 'Sunday']
            for (start_time, end_time), discount in zip(time_intervals, weekday_discounts):
                discount = weekend_discount if is_weekend else discount
                expanded_data.append({
                    'id_start': row['id_start'], 'id_end': row['id_end'], 'distance': row['distance'],
                    'start_day': day, 'start_time': start_time, 'end_day': day, 'end_time': end_time,
                    'moto': row['moto'] * discount, 'car': row['car'] * discount, 'rv': row['rv'] * discount,
                    'bus': row['bus'] * discount, 'truck': row['truck'] * discount
                })
    return pd.DataFrame(expanded_data)
result_df = calculate_time_based_toll_rates(unrolled_df)
print(result_df)

