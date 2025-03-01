import pandas as pd

file_path = 'src/data/image.csv'
image_df = pd.read_csv(file_path)
def get_df(start_index,end_index):
    final_column = image_df[['photo_id', 'photo_image_url']]
    return final_column[start_index:end_index]   