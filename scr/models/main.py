import numpy as np
import pandas as pd
import dummylog 
import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import PredefinedSplit, train_test_split
from dimensionality_reduction import set_dim_reduction 
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from metaheuristics import get_metaheuristic
from models import get_clf, save_clf


def split_data(X_train, X_val, y_train, y_val):
    # Split Data to Train and Validation
    split_index = [-1]*len(X_train) + [0]*len(X_val)
    X = np.concatenate((X_train, X_val), axis=0)
    y = np.concatenate((y_train, y_val), axis=0)
    pds = PredefinedSplit(test_fold=split_index)
    return X, y, pds

FILE_READ_PATH = r'C:\\Monografia\\datasets\\extracted_feature_datasets\\Balanced_Features_Data.csv'
RESULT_PATH =  r'C:\\Monografia\\results\\'
if __name__ == '__main__':
    
    dl = dummylog.DummyLog() 
    dl.logger.info('Log File is Created Successfully')

    df = pd.read_csv(FILE_READ_PATH, index_col=0)
    dl.logger.info('Feature Dataset Reading was Finished')

    X = df.iloc[: , :-1]
    y = df.iloc[: , -1]
    

    X_train, X_aux, y_train, y_aux = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_aux, y_aux, test_size=0.5, random_state=42, shuffle=True)
    X_cv, y_cv, pds = split_data(X_train, X_val, y_train, y_val)

    for reduction_meth in ['PCA']:# ,'FeatureAgg'
        for clf_meth in ['KNN', 'XGBoost']:#'RandomForest', 'NeuralNetwork', 'LogisticRegression', 
            for metaheurisc_meth in ['RandomSearch', 'GeneticSearch']:
                
                str_inter_name = clf_meth+'_'+reduction_meth+'_'+metaheurisc_meth

                reduction_method = set_dim_reduction(reduction_meth)
                clf, search_space = get_clf(clf_meth)
                
                model = Pipeline(steps=[
                                ('Scaler',  MinMaxScaler()),
                                ('Dim Reduction', reduction_method),
                                ('clf', clf)
                                ])

                search_method = get_metaheuristic(method=metaheurisc_meth, estimator=model, pds=pds, search_space_dict=search_space)
                
                start = time.time()
                search_method.fit(X=X_cv, y=y_cv)
                dtime = time.time() - start

                dl.logger.info(str_inter_name+' Training Time: '+str(dtime)+'sec')
                
                best_model = search_method.best_estimator_
                dl.logger.info(str_inter_name+' Best Model: '+str(best_model))
                
                y_pred = best_model.predict(X_test)                
                test_acc = accuracy_score(y_test, y_pred)
                dl.logger.info(str_inter_name+' Test Acc: '+str(test_acc))

                print('Teste Score:', test_acc)
                save_clf(best_model, str_inter_name)

                history = pd.DataFrame(search_method.cv_results_).sort_values("mean_test_score", ascending=False).head()
                history.to_csv(RESULT_PATH+str_inter_name+'_history.csv')
    

# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.GenericUnivariateSelect.html#sklearn.feature_selection.GenericUnivariateSelect
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFpr.html#sklearn.feature_selection.SelectFpr
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html#sklearn.feature_selection.f_classif