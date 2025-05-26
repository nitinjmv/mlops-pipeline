import pandas as pd
from src.data_pipeline.data_preprocessing import transform_text, preprocess_df

def test_transform_text():
    raw_text = "Hello!!! This is a TEST, with numbers 123 and symbols #@!"
    result = transform_text(raw_text)
    assert isinstance(result, str)
    assert 'test' in result
    assert 'hello' in result

def test_preprocess_df():
    sample_data = pd.DataFrame({
        "text": ["Free entry now", "Free entry now", "Call me now!"],
        "target": ["spam", "spam", "ham"]
    })
    processed = preprocess_df(sample_data)
    assert processed.shape[0] == 2  # Duplicate should be dropped
    assert all(col in processed.columns for col in ["text", "target"])
