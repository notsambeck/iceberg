import pandas as pd

output_df = pd.read_json('data/test.json')
output_df.set_index('id', inplace=True)

def pickle_chunks(data, chunks=10, filepath='data/test_data_'):
    
