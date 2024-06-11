import pandas as pd

def load_crop_data(csv_file):
    return pd.read_csv(csv_file)

def recommend_crops(soil_type, season, crop_df):
    crops = crop_df[(crop_df['Soil Type'] == soil_type) & (crop_df['Season'] == season)][['Corp1', 'Corp2', 'Corp3']].values[0]
    return crops
