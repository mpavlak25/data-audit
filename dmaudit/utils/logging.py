import os
import pandas as pd


def log_data_to_csv(log_path, record_dict):

    record = pd.DataFrame.from_records(record_dict, index=[0])

    if not os.path.exists(log_path):
        record.to_csv(log_path, mode='w', index=False, header=True)
    else:
        record.to_csv(log_path, mode='a', index=False, header=False)

