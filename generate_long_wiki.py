from datasets import load_dataset
import pandas as pd


def filter_data(df):
    # Select articles that:
    # 1. Have 20000 to 30000 characters in range.
    # 2. Have less than 260 break line characters. 
    # Last condition helps to remove degenerate articles.
    select = list(map(lambda x: \
        len(x) > 20000 and 
        x.count('\n') < 260 and 
            len(x) < 30000, df.text))
    selected_df = df[select]
    long_text_only = selected_df[['text']]
    long_text_only.reset_index(inplace=True)
    
    return long_text_only

