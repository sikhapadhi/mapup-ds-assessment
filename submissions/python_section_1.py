from typing import Dict, List

import pandas as pd
import re
import polyline
from math import radians,sin,cos,sqrt,atan2
from typing import List



def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    # Your code goes here.
    i = 0
    while i < len(lst):
        start = i
        end = min (i + n - 1,len(lst) -1)

        while start < end:
            lst[start], lst[end] = lst[end], lst[start]
            start += 1
            end -= 1

        i += n
    
    return lst
print(reverse_by_n_elements([1,2,3,4,5,6,7,8],3))
print(reverse_by_n_elements([1,2,3,4,5],2))
print(reverse_by_n_elements([10,20,30,40,50,60,70],4))


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Your code here
    length_dict = {}
    for string in lst:
        length = len(string)
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(string)
    return dict(sorted(length_dict.items()))
print(group_by_length(['apple', 'bat', 'car', 'elephant', 'dog', 'bear']))
print(group_by_length(['one', 'two', 'three', 'four']))

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    # Your code here
    def _flatten(d: any, parent_key: str = '') -> Dict[str, any]:
        items = {}
        for k,v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v,dict):
                items.update(_flatten(v,new_key))
            elif isinstance(v,list):
                for i,item in enumerate(v):
                    list_key = f"{new_key}[{i}]"
                    if isinstance(item,dict):
                        items.update(_flatten(item,list_key))
                    else:
                        items[list_key] = item
            else:
                items[new_key] = v
        return items
    return _flatten(nested_dict)
nested_dict = {
    "road" : {
        "name" : "Highway 1",
        "length" : 350,
        "sections" : [{"id": 1, "condition": {"pavement": "good", "traffic": "moderate"}}]
    }
}
flattened = flatten_dict(nested_dict)
print(flattened)

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Your code here
    def backtrack(start: int):
        if start == len(nums):
            result.append(nums[:])
            return
        seen = set()
        for i in range(start, len(nums)):
            if nums[i] in seen:
                continue
            seen.add(nums[i])
            nums[start] , nums[i] = nums[i] , nums[start]
            backtrack(start + 1)
            nums[start] , nums[i] = nums[i] , nums[start]
    nums.sort()
    result = []
    backtrack(0)
    return result
input_list = [1, 1, 2]
print(unique_permutations(input_list))

pass


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    date_pattern = r'\b\d{2}-\d{2}-\d{4}\b|\b\d{2}/\d{2}/\d{4}\b|\b\d{4}\.\d{2}\.\d{2}\b'
    valid_dates = re.findall(date_pattern, text)
    return valid_dates
text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
print(find_all_dates(text))

def haversine(lat1,lon1,lat2,lon2):
    R = 6371000
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    coordinates = polyline.decode(polyline_str)
    df = pd.DataFrame(coordinates , columns = ['latitude' , 'longitude'])
    df['distance'] = 0.0
    for i in range (1,len(df)):
        lat1,lon1 = df.iloc[i - 1]['latitude'],df.iloc[i - 1]['longitude']
        lat2,lon2 = df.iloc[i]['latitude'],df.iloc[i]['longitude']
        df.at[i,'distance'] = haversine(lat1,lon1,lat2,lon2)
    return df
polyline_str = "u{~vFvyys@fE`FzC~Ds@jA"
df = polyline_to_dataframe(polyline_str)
print(df)

def rotate_matrix(matrix: List[List[int]]) -> List[List[int]]:
    n = len(matrix)
    rotated = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]
    return rotated

def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    # Your code here
    rotated_matrix = rotate_matrix(matrix)
    n = len(matrix)
    transformed_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i])
            col_sum = sum(rotated_matrix[k][j] for k in range(n))
            transformed_matrix[i][j] = (row_sum + col_sum - rotated_matrix[i][j])
    return transformed_matrix
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
final_matrix = rotate_and_multiply_matrix(matrix)
for row in final_matrix:
    print(row)


def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here
    all_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    all_times = pd.date_range("00:00:00", "23:59:59", freq="T").time  # All minutes in a 24-hour day
    
    # Convert time columns to datetime.time type for proper comparison
    df['startTime'] = pd.to_datetime(df['startTime']).dt.time
    df['endTime'] = pd.to_datetime(df['endTime']).dt.time
    
    # Function to check if a pair covers all days and all times
    def is_complete(group):
        days_covered = set(group['startDay'].unique())
        times_covered = set()
        
        for _, row in group.iterrows():
            # Add times within the start and end range
            if row['startDay'] == row['endDay']:
                times_covered.update(pd.date_range(row['startTime'], row['endTime'], freq='T').time)
            else:
                # If the day spans across multiple days, handle those time ranges separately
                times_covered.update(all_times)  # Consider it as fully covering the day
                
        # Check if all days and times are covered
        all_days_covered = days_covered == set(all_days)
        all_times_covered = times_covered == set(all_times)
        return all_days_covered and all_times_covered

    # Group by (id, id_2) and apply the completeness check
    completeness_check = df.groupby(['id', 'id_2']).apply(is_complete)
    
    return completeness_check
df = pd.read_csv(r"C:\Users\sikha\OneDrive\Desktop\Answers\MapUp-DA-Assessment-2024\datasets\dataset-1.csv")  # Load dataset-1.csv
result = time_check(df)  # Get the boolean series
print(result)
    