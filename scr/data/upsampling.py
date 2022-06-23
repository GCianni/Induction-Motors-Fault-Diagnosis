from sklearn.utils import resample
import pandas as pd

FILE_READ_PATH = r'C:\\Monografia\\data\\Extracted_Feature_dataset\\Features Data.csv'
FILE_SAVE_PATH = r'C:\\Monografia\\data\\Extracted_Feature_dataset\\Balanced_Features_Data.csv'

def upsampling_data(data):
    X = data.iloc[: , :-1]
    y = data.iloc[: , -1]
    max_sample = y.value_counts()[y.value_counts().idxmax()]

    while(len(y.value_counts().value_counts()) != 1):
        min_sample_class = y.value_counts().idxmin() 

        X_min = X[y == min_sample_class]
        y_min = y[y == min_sample_class]
        X_dif = X[y != min_sample_class]
        y_dif = y[y != min_sample_class]

        X_min, y_min = resample(X_min, y_min,
                            replace=True, n_samples= max_sample, random_state = 123)
        
        X = pd.concat([X_min,X_dif], axis=0, sort=False)
        X.reset_index(drop=True, inplace=True)
        y = pd.concat([y_min,y_dif], axis=0, sort=False)
        y.reset_index(drop=True, inplace=True)
    return X, y     

df = pd.read_csv(FILE_READ_PATH, index_col=0)
print(df)
X, y = upsampling_data(df)
df = pd.concat([X,y], axis =1)
df.to_csv(FILE_SAVE_PATH)