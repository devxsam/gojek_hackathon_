import pandas as pd
from sklearn.model_selection import train_test_split
import locale

from src.features.transformations import (
    driver_distance_to_pickup,
    hour_of_day,
    driver_historical_completed_bookings,
    time_to_accept_ride,
    add_actual_idle_time,
    add_profitability_ratios,
    add_time_distance_interaction

)
from src.utils.store import AssignmentStore


def main():

    try:
        
        locale.setlocale(locale.LC_ALL, 'C') 
    except locale.Error:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

    store = AssignmentStore()

    dataset = store.get_processed("dataset.csv")

    dataset = apply_feature_engineering(dataset)

    store.put_processed("transformed_dataset.csv", dataset)


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:

    return (
        df.pipe(driver_distance_to_pickup)
        .pipe(hour_of_day)
        .pipe(time_to_accept_ride)
        .pipe(add_actual_idle_time)
        .pipe(add_profitability_ratios)
        .pipe(driver_historical_completed_bookings)
        .pipe(add_time_distance_interaction)
    )



if __name__ == "__main__":
    main()
