import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np

def normalize_data(df, columns):
    """
    Normalize specified columns of a DataFrame using Min-Max scaling.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): A list of column names to normalize.

    Returns:
        pd.DataFrame: The DataFrame with normalized columns.
    """
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def encode_categorical(df, columns):
    """
    Encode specified categorical columns of a DataFrame using Label Encoding.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): A list of column names to encode.

    Returns:
        pd.DataFrame: The DataFrame with encoded categorical columns.
    """
    for col in columns:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
    return df

def pad_sequences(sequences, max_length, padding_value=0):
    """
    Pad sequences to a fixed maximum length.

    Args:
        sequences (list or np.ndarray): The input sequences.
        max_length (int): The desired maximum length of the sequences.
        padding_value (int or float): The value to use for padding.

    Returns:
        np.ndarray: The padded sequences.
    """
    padded = np.full((len(sequences), max_length), padding_value)
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_length)
        padded[i, :length] = seq[:length]
    return padded

if __name__ == '__main__':
    # Example Usage
    data = {'col1': [1, 2, 3, 4, 5],
            'col2': ['A', 'B', 'A', 'C', 'B'],
            'col3': [10, 20, 30, 40, 50]}
    df = pd.DataFrame(data)

    # Normalize 'col1' and 'col3'
    df_normalized = normalize_data(df.copy(), ['col1', 'col3'])
    print("Normalized DataFrame:")
    print(df_normalized)

    # Encode 'col2'
    df_encoded = encode_categorical(df.copy(), ['col2'])
    print("\nEncoded DataFrame:")
    print(df_encoded)

    # Example padding for sequences
    sequences = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    max_len = 5
    padded_sequences = pad_sequences(sequences, max_len)
    print("\nPadded Sequences:")
    print(padded_sequences)