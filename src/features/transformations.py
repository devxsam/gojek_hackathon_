import pandas as pd
from haversine import haversine

from src.utils.time import robust_hour_of_iso_date


def driver_distance_to_pickup(df: pd.DataFrame) -> pd.DataFrame:
    df["driver_distance"] = df.apply(
        lambda r: haversine(
            (r["driver_latitude"], r["driver_longitude"]),
            (r["pickup_latitude"], r["pickup_longitude"]),
        ),
        axis=1,
    )
    return df

import pandas as pd
import numpy as np

def driver_historical_completed_bookings(df: pd.DataFrame) -> pd.DataFrame:
    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'], utc=True, errors='coerce')
    df_sorted = df.sort_values(by=['driver_id', 'event_timestamp']).copy()
    
   
    df_sorted['is_accepted'] = (df_sorted['participant_status'].str.lower() == 'accepted').astype(int)
    df_sorted['accepted_cumulative'] = df_sorted.groupby('driver_id')['is_accepted'].cumsum()
    df_sorted['offer_count_cumulative'] = df_sorted.groupby('driver_id')['is_accepted'].cumcount() + 1
    
 
    df_sorted["historical_bookings"] = (
        df_sorted['accepted_cumulative'] / df_sorted['offer_count_cumulative']
    ).shift(1)
    
    
    global_mean = df_sorted['is_accepted'].mean()
    df_sorted["historical_bookings"] = df_sorted["historical_bookings"].fillna(global_mean)
    
    df = df.merge(
        df_sorted[['order_id', 'driver_id', 'historical_bookings']], 
        on=['order_id', 'driver_id'],                              
        how='left'
    )
    return df

def hour_of_day(df: pd.DataFrame) -> pd.DataFrame:
    df["event_hour"] = df["event_timestamp"].apply(robust_hour_of_iso_date)
    return df


def time_to_accept_ride(df: pd.DataFrame) -> pd.DataFrame:
    
    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'], format='mixed', utc=True)
    
    df['year'] = df['event_timestamp'].dt.year
    df['month'] = df['event_timestamp'].dt.month
    df['day'] = df['event_timestamp'].dt.day
    df['hour'] = df['event_timestamp'].dt.hour
    df['minute'] = df['event_timestamp'].dt.minute
    
   
    morning_peak_hours_utc = [7, 8, 9] 
    evening_peak_hours_utc = [17, 18, 19]
    peak_hours = set(morning_peak_hours_utc + evening_peak_hours_utc)

    df['is_peak_hour'] = df['hour'].apply(lambda x: 1 if x in peak_hours else 0)
    
    return df


def add_actual_idle_time(df: pd.DataFrame) -> pd.DataFrame:
    
    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'], utc=True, format='mixed')
    df_sorted = df.sort_values(by=['driver_id', 'event_timestamp']).copy()
    
    completed_timestamps = df_sorted.apply(
        lambda row: row['event_timestamp'] if row['is_completed'] == 1 else pd.NaT, axis=1
    )

    df_sorted['last_completed_timestamp'] = completed_timestamps.groupby(df_sorted['driver_id']).ffill().shift(1)

    df_sorted['actual_idle_time_minutes'] = (df_sorted['event_timestamp'] - df_sorted['last_completed_timestamp']).dt.total_seconds() / 60
    
    df_sorted['actual_idle_time_minutes'] = df_sorted['actual_idle_time_minutes'].fillna(0)

    df = df.merge(df_sorted[['order_id', 'actual_idle_time_minutes']], on='order_id', how='left', suffixes=('', '_new'))

    return df


def add_profitability_ratios(df: pd.DataFrame) -> pd.DataFrame:
    epsilon = 1e-6
    df['pickup_to_trip_ratio'] = df['driver_distance'] / (df['trip_distance'].fillna(0) + epsilon)
    
    df['total_commitment_distance'] = df['driver_distance'] + df['trip_distance'].fillna(0)

    return df

def add_time_distance_interaction(df: pd.DataFrame) -> pd.DataFrame:
    df['peak_distance_penalty'] = df['driver_distance'] * df['is_peak_hour']
    return df

