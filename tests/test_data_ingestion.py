import os
import pandas as pd
import pytest
from unittest.mock import patch, mock_open
from io import StringIO

from src.data_pipeline.data_ingestion import (
    load_params, load_data, preprocess_data, save_data
)

# Fixture sample CSV
sample_csv = """v1,v2,Unnamed: 2,Unnamed: 3,Unnamed: 4
spam,Buy now,,, 
ham,Hello friend,,, 
"""

def test_load_params_valid(tmp_path):
    yaml_content = "data_ingestion:\n  test_size: 0.2\n"
    param_path = tmp_path / "params.yaml"
    param_path.write_text(yaml_content)

    params = load_params(str(param_path))
    assert "data_ingestion" in params
    assert params["data_ingestion"]["test_size"] == 0.2

def test_load_data_valid():
    df = pd.read_csv(StringIO(sample_csv))
    assert df.shape == (2, 5)
    assert "v1" in df.columns

def test_preprocess_data():
    df = pd.read_csv(StringIO(sample_csv))
    processed = preprocess_data(df.copy())
    assert "target" in processed.columns
    assert "text" in processed.columns
    assert processed.shape == (2, 2)

def test_save_data(tmp_path):
    train = pd.DataFrame({"text": ["a"], "target": ["b"]})
    test = pd.DataFrame({"text": ["c"], "target": ["d"]})

    save_data(train, test, tmp_path)
    saved_train = pd.read_csv(tmp_path / "raw" / "train.csv")
    saved_test = pd.read_csv(tmp_path / "raw" / "test.csv")

    assert saved_train.shape == (1, 2)
    assert saved_test.shape == (1, 2)
