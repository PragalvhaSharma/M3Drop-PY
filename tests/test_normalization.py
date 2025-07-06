import numpy as np
import pandas as pd
import pytest
from m3d.normalization import M3DropConvertData, M3DropCleanData

def test_M3DropConvertData_dataframe_counts():
    # Create a sample DataFrame
    data = {'cell1': [1, 2, 0], 'cell2': [0, 5, 6], 'cell3': [7, 0, 9]}
    df = pd.DataFrame(data, index=['gene1', 'gene2', 'gene3'])

    # Expected output
    counts = df.values.astype(float)
    sf = np.sum(counts, axis=0)
    median_sf = np.median(sf)
    expected_output = (counts / sf) * median_sf
    
    # Run the function
    output = M3DropConvertData(df, is_counts=True)
    
    # Assert the output is as expected
    assert isinstance(output, pd.DataFrame)
    np.testing.assert_allclose(output.values, expected_output)

def test_M3DropConvertData_log_transformed_dataframe():
    # Create a sample log-transformed DataFrame
    pseudocount = 1
    data = {'cell1': [np.log2(1 + pseudocount), np.log2(2 + pseudocount), 0], 
            'cell2': [0, np.log2(5 + pseudocount), np.log2(6 + pseudocount)], 
            'cell3': [np.log2(7 + pseudocount), 0, np.log2(9 + pseudocount)]}
    df = pd.DataFrame(data, index=['gene1', 'gene2', 'gene3'])
    
    # Expected output (un-log-transformed)
    expected_output = 2**df.values - pseudocount
    
    # Run the function
    output = M3DropConvertData(df, is_log=True, pseudocount=pseudocount)
    
    # Assert the output is as expected
    np.testing.assert_allclose(output, expected_output)

def test_M3DropConvertData_numpy_array_counts():
    # Create a sample numpy array
    arr = np.array([[1, 0, 7], [2, 5, 0], [0, 6, 9]])
    
    # Expected output
    counts = arr.astype(float)
    sf = np.sum(counts, axis=0)
    median_sf = np.median(sf)
    expected_output = (counts / sf) * median_sf
    
    # Run the function
    output = M3DropConvertData(arr, is_counts=True)
    
    # Assert the output is as expected
    assert isinstance(output, np.ndarray)
    np.testing.assert_allclose(output, expected_output)


def test_M3DropCleanData_dataframe():
    # Create a sample DataFrame
    data = {'cell1': [1, 2, 0], 'cell2': [0, 5, 6], 'cell3': [7, 0, 9]}
    df = pd.DataFrame(data, index=['gene1', 'gene2', 'gene3'])

    # Expected output
    # M3DropCleanData with is_counts=True performs CPM normalization
    # and filters genes with low expression (and cells with low gene counts, though not triggered here)
    # The filtering of genes here is complex, so we will test the output type and basic properties
    
    # Run the function
    output = M3DropCleanData(df, is_counts=True)
    
    # Assert the output is a dictionary with 'data' and 'labels'
    assert 'data' in output
    assert 'labels' in output
    assert isinstance(output['data'], pd.DataFrame)
