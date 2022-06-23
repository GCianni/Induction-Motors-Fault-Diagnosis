import time
import pandas as pd
from ETL import get_datafiles, filter_data, get_featured_labeled_data
from upsampling import upsampling_data

# Get datafiles -> Filter each df -> Extract Feature & Index label for each df -> append dfs -> save dfs -> upsampling -> [normalized, model] -> 

DATAFILE_PATH = r'C:\\Monografia\\data\\MAFAULDA'
LABEL_PATH = r'C:\\Monografia\\data\\Target.txt'
SAVE_PATH = r'C:\\Monografia\\data\\Extracted_Feature_dataset\\'
FILE_SAVE_PATH = r'C:\\Monografia\\data\\Extracted_Feature_dataset\\Balanced_Features_Data.csv'
SAMPLE_FREQ = 50000

if __name__ == '__main__':
    data = []

    start = time.time()
    dataframes, label, files_names = get_datafiles(DATAFILE_PATH, LABEL_PATH)
    
    filter_dataframes = filter_data(dataframes=dataframes, cutoff_freq=20000, fs=SAMPLE_FREQ)
    feature_df = [get_featured_labeled_data(label=label[i], dataframe=dataframe, fs=SAMPLE_FREQ,
                            window_size=500, window_step=450, name = str(files_names[i]))
                    for i, dataframe in enumerate(filter_dataframes)]
    print('len of one df:', len(feature_df[0]))
    print('len dfs:',len(feature_df))

    data = pd.concat(feature_df, axis=0, sort=False)
    data.reset_index(drop=True, inplace=True)
    print(len(data))
    print(data)
    data.to_csv(SAVE_PATH+'Features_Data.csv')

    X, y = upsampling_data(data)
    data = pd.concat([X,y], axis =1)
    data.to_csv(SAVE_PATH+'Balanced_Features_Data.csv')

    print(time.time()-start)
