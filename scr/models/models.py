from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier


import pickle

def get_clf(estimator):
    rf_params = {
        'clf__n_estimators': [1, 2, 5, 10, 25, 20, 30],
        'clf__max_depth': [10, 15, 20, 30, 50],
        'clf__min_samples_leaf': [1, 2, 4, 8, 16, 32],
        'clf__bootstrap': [True, False],
        'clf__criterion': ['gini', 'entropy']
    }

    """
    ada_params = {
        'clf__base_estimator': [BernoulliNB(),SGDClassifier(), DecisionTreeClassifier(), Perceptron()],
        'clf__n_estimators': [10, 50, 100, 250, 500, 750, 1000],
        'clf__learning_rate': [0.001, 0.01, 0.1, 0.5, 1, 10],
    }
    """

    xgb_params = {
        'clf__max_depth': [2, 3, 4, 5, 6, 10],
        'clf__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'clf__n_estimators': [8, 32, 64, 128, 256, 512, 1024],
        'clf__colsample_bytree': [0.1, 0.3, 0.7, 0.9, 1]
    }

    ann_params = {
        'clf__hidden_layer_sizes': [(2, 2), (2, 2, 2), (2, 2, 2, 2),
                               (4, 4), (4, 4, 4), (4, 4, 4, 4),
                               (8, 8), (8, 8, 8), (8, 8, 8, 8),
                               (32, 32), (32, 32, 32), (32, 32, 32, 32),
                               (64, 64), (64, 64, 64), (64, 64, 64, 64),
                               (128, 128), (128, 128, 128), (128, 128, 128, 128),
                               (256, 256), (256, 256, 256), (256, 256, 256, 256)],
        'clf__activation': ['identity', 'logistic', 'tanh', 'relu'],
        'clf__solver': ['sgd', 'adam'],
        'clf__alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1],
        'clf__learning_rate': ['constant', 'adaptive']
    }

    logreg_params = {
        'clf__penalty':['l2','none'],
        'clf__solver': ['newton-cg', 'lbfgs','sag', 'saga'],
        'clf__C':[0.01, 0.05, 0.1, 0.5, 1, 2]
    }

    knn_params = {
        'clf__n_neighbors':[5, 10, 15, 20, 25, 30, 35, 40, 50],
        'clf__weights': ['uniform', 'distance'],
        'clf__p':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }


    param_dict = {
        'RandomForest': [rf_params, RandomForestClassifier(random_state=0)],

        'XGBoost': [xgb_params, XGBClassifier(num_class=10,
                                              verbosity=0,
                                              objective='multi:softmax',
                                              use_label_encoder=False)],

        'NeuralNetwork': [ann_params, MLPClassifier(tol=1e-3,
                                                     max_iter=100,
                                                     early_stopping=True,
                                                     n_iter_no_change=4,
                                                     random_state=0)],

        'LogisticRegression': [logreg_params, LogisticRegression(random_state=0, dual=False)],

        'KNN': [knn_params, KNeighborsClassifier()],
        #'Adaboost': [ada_params, AdaBoostClassifier(random_state=0, algorithm='SAMME')],
    }

    search_space = param_dict[estimator][0]
    clf = param_dict[estimator][1]

    return clf, search_space

def save_clf(clf, model_title):
    MODEL_SAVE_PATH = r'C:\\Monografia\\models\\'
    with open(MODEL_SAVE_PATH+model_title+'_model.pkl','wb') as f:
        pickle.dump(clf,f)