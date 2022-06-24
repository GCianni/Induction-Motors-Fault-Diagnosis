from sklearn.model_selection import RandomizedSearchCV
import sklearn
from sklearn.metrics import accuracy_score
from evolutionary_search import EvolutionaryAlgorithmSearchCV
import warnings

warnings.filterwarnings("ignore")

def get_metaheuristic (method:str, estimator, pds, search_space_dict):
    if method == 'RandomSearch':
        return RandomizedSearchCV(estimator=estimator, cv=pds, param_distributions=search_space_dict, n_jobs=-1)
    elif method == 'GeneticSearch':
        searcher = EvolutionaryAlgorithmSearchCV(estimator=estimator, params=search_space_dict, scoring="accuracy", cv=pds,
                                        verbose=1,
                                        population_size=20,#10,
                                        gene_mutation_prob=0.1,
                                        gene_crossover_prob=0.5,
                                        tournament_size=3,
                                        generations_number=10,#25,
                                        n_jobs=1)
        return searcher
    else:
        return 'error'