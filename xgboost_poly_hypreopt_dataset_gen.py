import pandas as pd
from hyperopt import tpe,hp,Trials,fmin,STATUS_OK,STATUS_FAIL,space_eval
from hyperopt.early_stop import no_progress_loss
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
import numpy as np
import json,pickle
from sklearn.preprocessing import PolynomialFeatures


# Helper Functions
def random_integer_mapping(nucleotide_sequence):

    new_rep = []

    mapping = {'T' : 0,
              'C' : 1,
              'A' : 2,
              'G' : 3}

    for nt in nucleotide_sequence:
        nt_transform = mapping[nt]
        new_rep.append(nt_transform)

    return new_rep

def compliment_integer_mapping(nucleotide_sequence):

    new_rep = []

    mapping = {'T' : 1.5,
              'C' : 0.5,
              'A' : -1.5,
              'G' : -0.5}

    for nt in nucleotide_sequence:
        nt_transform = mapping[nt]
        new_rep.append(nt_transform)

    return new_rep


def eeip_mapping(nucleotide_sequence):

    new_rep = []

    mapping = {'T' : 0.1335,
              'C' : 0.1340,
              'A' : 0.1260,
              'G' : 0.0806}

    for nt in nucleotide_sequence:
        nt_transform = mapping[nt]
        new_rep.append(nt_transform)

    return new_rep



def dna_walk_mapping(nucleotide_sequence):

    count = 0

    new_rep = []

    for nt in nucleotide_sequence:
        if nt == 'C' or nt == 'T':
            count+=1
            new_rep.append(count)
        if nt == 'A' or nt == 'G':
            count-=1
            new_rep.append(count)
            
    return new_rep

def sequence_transform(dataset=None,mapping=None,
                       sequence_column_idx=None,intensity_column_idx=None):

    n,_ = dataset.shape
    
    sequence_vector = dataset.iloc[:,sequence_column_idx]
    intensity_column_vector = dataset.iloc[:,intensity_column_idx]

    new_data_set = []
    
    for idx in range(0,n):

        seq = sequence_vector[idx]
        
        seq_transform = mapping(seq)

        intensity = intensity_column_vector[idx]

        seq_transform.append(intensity)

        new_data_set.append(seq_transform)

    return pd.DataFrame(new_data_set) 



def safe_objective_function(model_instance=None, parameters=None, X_train=None,y_train=None,splits=5,validation_random_state=200,penalty=1e9):

    try:

        model = clone(model_instance)

        model.set_params(**parameters)

        kfold = KFold(n_splits=splits,shuffle=True,random_state=validation_random_state)

        neg_mse_scores = cross_val_score(model,X_train,y_train,cv=kfold,scoring='neg_mean_squared_error')
        

        mse_scores = -1 * neg_mse_scores

        mean_mse = mse_scores.mean()

        if not np.isfinite(mean_mse) or mean_mse > 1e8:
            return {"loss" : penalty, "status" : STATUS_OK}
        
        else:
            return {"loss" : float(mean_mse), "status" : STATUS_OK}
        
    except Exception as e:
        return {"loss" : penalty, "status" : STATUS_FAIL}
    


# Global Parameters
mappings = [random_integer_mapping,compliment_integer_mapping,
           eeip_mapping,dna_walk_mapping]


mapping_names = ['random_integer', 'compliment_integer',
                'eeip', 'dna_walk']



xgb_space = {
    'max_depth' : hp.pchoice('max_depth',[(0.5,6), (0.3,3), (0.2,10)]),
    'n_estimators' : 500,
    'min_child_weight' : hp.loguniform('min_child_weight',np.log(1),np.log(10)),
    'subsample' : hp.pchoice('subsample',[(0.5,0.5),(0.25,0.3),(0.25,0.7)]),
    'learning_rate' : hp.loguniform('learning_rate',np.log(1e-3),np.log(0.5)),
    'colsample_bytree' : hp.pchoice('colsample_bytree',[(0.5,0.5),(0.25,0.3),(0.25,0.7)]),
    'lambda': hp.loguniform('lambda', np.log(0.1), np.log(10)),
    'alpha': hp.loguniform('alpha', np.log(1e-6), np.log(1)),
    'gamma': hp.loguniform('gamma', np.log(1e-6), np.log(1))
}

df_cy3 = pd.read_csv('../data/3_prime_Cy3.csv')



train_residuals = {}
test_residuals = {}

hyperopt_data = []



for idx in range(0,len(mappings)):

    seq_transform_function = mappings[idx]
    mapping_name = mapping_names[idx]

    transformed_df = sequence_transform(dataset=df_cy3, mapping=seq_transform_function,
                                sequence_column_idx=0,intensity_column_idx=1)
    
    poly = PolynomialFeatures(2, interaction_only=True)

    X_train, X_test, y_train, y_test = train_test_split(transformed_df.iloc[:,:-1], 
                                                                    transformed_df.iloc[:,-1],
                                                                    test_size=0.30, random_state=42)
    
    X_train = poly.fit_transform(X_train)
    X_test = poly.fit_transform(X_test)
    

    
    print(f'Currently Working on the following sequence transformation: {mapping_names[idx]}')
    print("\n")

    

    for repeat in range(0,10):


        xgb = XGBRegressor()

        trials = Trials()
        best = fmin(fn=lambda p:safe_objective_function(
            model_instance=xgb,
            parameters=p,
            X_train=X_train,
            y_train=y_train
        ),
                    space=xgb_space,
                    algo=tpe.suggest,
                    max_evals=100,
                    trials=trials,
                    timeout=300,
                    rstate=None,
                    early_stop_fn=no_progress_loss(10),
                    return_argmin=True)
        
        
        
        best_param_values = space_eval(xgb_space, best)

        final_xgb_model = XGBRegressor(**best_param_values)
        final_xgb_model.fit(X_train, y_train)

        y_train_pred = final_xgb_model.predict(X_train)
        y_train_residuals = y_train.values - y_train_pred


        y_test_pred = final_xgb_model.predict(X_test)

        y_test_residuals = y_test.values - y_test_pred



        train_residual_key_name = "cy3_" + "xgb_" + mapping_name + "_train_residuals_" + f"r{repeat}"
        train_residuals[train_residual_key_name] = y_train_residuals

        
        test_residual_key_name = "cy3_" + "xgb_"  + mapping_name + "_test_residuals_" + f"r{repeat}"
        test_residuals[test_residual_key_name] = y_test_residuals

        




        r2 = final_xgb_model.score(X_test,y_test)
        data_row = ['cy3','xgb',mapping_name,r2,2,repeat]
        hyperopt_data.append(data_row)

        trials_file_name = f'cy3_poly_xgb_tree_{mapping_name}_{repeat}_trials.pkl'

        with open(trials_file_name,"wb") as outfile:
            pickle.dump(trials,outfile)

        





df_cy5 = pd.read_csv('../data/3_prime_Cy5.csv')



for idx in range(0,len(mappings)):

    seq_transform_function = mappings[idx]
    mapping_name = mapping_names[idx]

    transformed_df = sequence_transform(dataset=df_cy5, mapping=seq_transform_function,
                                sequence_column_idx=0,intensity_column_idx=1)
    

    poly = PolynomialFeatures(2, interaction_only=True)

    X_train, X_test, y_train, y_test = train_test_split(transformed_df.iloc[:,:-1], 
                                                                    transformed_df.iloc[:,-1],
                                                                    test_size=0.30, random_state=42)
    
    X_train = poly.fit_transform(X_train)
    X_test = poly.fit_transform(X_test)
    

    
    print(f'Currently Working on the following sequence transformation: {mapping_names[idx]}')
    print("\n")

    

    for repeat in range(0,10):


        xgb = XGBRegressor()

        trials = Trials()
        best = fmin(fn=lambda p:safe_objective_function(
            model_instance=xgb,
            parameters=p,
            X_train=X_train,
            y_train=y_train
        ),
                    space=xgb_space,
                    algo=tpe.suggest,
                    max_evals=100,
                    trials=trials,
                    timeout=300,
                    rstate=None,
                    early_stop_fn=no_progress_loss(10),
                    return_argmin=True)
        
        
        
        best_param_values = space_eval(xgb_space, best)

        final_xgb_model = XGBRegressor(**best_param_values)
        final_xgb_model.fit(X_train, y_train)

        y_train_pred = final_xgb_model.predict(X_train)
        y_train_residuals = y_train.values - y_train_pred


        y_test_pred = final_xgb_model.predict(X_test)

        y_test_residuals = y_test.values - y_test_pred



        train_residual_key_name = "cy5_" + "xgb_" + mapping_name + "_train_residuals_" + f"r{repeat}"
        train_residuals[train_residual_key_name] = y_train_residuals

        test_residual_key_name = "cy5_" + "xgb_" + mapping_name + "_test_residuals_" + f"r{repeat}"
        test_residuals[test_residual_key_name] = y_test_residuals

        




        r2 = final_xgb_model.score(X_test,y_test)
        data_row = ['cy5','xgb',mapping_name,r2,2,repeat]

        hyperopt_data.append(data_row)

        trials_file_name = f'cy5_poly_xgb_{mapping_name}_{repeat}_trials.pkl'

        with open(trials_file_name,"wb") as outfile:
            pickle.dump(trials,outfile)





output_df = pd.DataFrame(hyperopt_data)

colnames = ['dataset', 'model_name', 'mapping_name', 'r2', 'polynomial_degree','repeat_num']
output_df.to_csv(path_or_buf='xgb_poly_hyperopt_data.csv',header=colnames)


train_residuals_df = pd.DataFrame(train_residuals)
train_residuals_df.to_csv(path_or_buf='xgb_poly_train_residuals.csv')

test_residuals_df = pd.DataFrame(test_residuals)
test_residuals_df.to_csv(path_or_buf='xgb_poly_test_residuals.csv')
