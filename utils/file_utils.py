import os


def get_file_names(upload_folder) -> list:
    data_op = [file for file in os.listdir(upload_folder) if file != 'dataset.zip']
    return data_op if len(data_op) != 0 else None
