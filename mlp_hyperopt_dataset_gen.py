import pandas as pd
from hyperopt import tpe,hp,Trials,fmin,STATUS_OK,STATUS_FAIL,space_eval
from hyperopt.early_stop import no_progress_loss
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler,QuantileTransformer
import numpy as np
import json,pickle
from math import log


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

        layers = tuple(int(x) for x in parameters['hidden_layer_sizes'])
        parameters['hidden_layer_sizes'] = layers

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



mlp_space = {
    'loss' : 'squared_error',
    'hidden_layer_sizes' : hp.choice('hidden_layer_sizes',
                                     [(hp.quniform('n1_layer1', 10,100,1),),
                                     (hp.quniform('n2_layer1',10,100,1),hp.quniform('n2_layer2',10,100,1)),
                                     (hp.quniform('n3_layer1',10,100,1),hp.quniform('n3_layer2',10,100,1),hp.quniform('n3_layer3',10,100,1))]), 
    'activation' : 'relu',
    'solver' : hp.choice('solver', ['sgd', 'adam']),
    'alpha' : hp.loguniform('alpha',-11.51,-4.605),
    'batch_size' : 'auto',
    'learning_rate' : hp.choice('lr',['constant', 'invscaling', 'adaptive']),
    'learning_rate_init' : hp.loguniform('learning_rate_init',-11.51,-4.605),
    'max_iter' : hp.choice('max_iter',[200,400,600,800,1000,1200,1400]),
    'random_state' : 6
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

    X_train, X_test, y_train, y_test = train_test_split(transformed_df.iloc[:,:-1], 
                                                                    transformed_df.iloc[:,-1],
                                                                    test_size=0.30, random_state=42)
    
    qt = QuantileTransformer(n_quantiles=5, random_state=0, output_distribution='normal')
    X_train = qt.fit_transform(X_train)


    y_train_scaler = StandardScaler()
    scaled_y_train_data = y_train_scaler.fit_transform(y_train.to_numpy().reshape(-1, 1))

    
    X_test = qt.fit_transform(X_test)


    y_test_scaler = StandardScaler()
    scaled_y_test_data = y_test_scaler.fit_transform(y_test.to_numpy().reshape(-1, 1))

    
    print(f'Currently Working on the following sequence transformation: {mapping_names[idx]}')
    print("\n")

    

    for repeat in range(0,10):


        mlp = MLPRegressor()

        trials = Trials()
        best = fmin(fn=lambda p:safe_objective_function(
            model_instance=mlp,
            parameters=p,
            X_train=X_train,
            y_train=scaled_y_train_data.ravel()
        ),
                    space=mlp_space,
                    algo=tpe.suggest,
                    max_evals=100,
                    trials=trials,
                    timeout=300,
                    rstate=None,
                    early_stop_fn=no_progress_loss(10),
                    return_argmin=True)
        
        
        
        best_param_values = space_eval(mlp_space, best)

        best_layers = tuple(int(x) for x in best_param_values['hidden_layer_sizes'])
        best_param_values['hidden_layer_sizes'] = best_layers

        final_mlp_model = MLPRegressor(**best_param_values)
        final_mlp_model.fit(X_train, scaled_y_train_data.ravel())

        y_train_pred = final_mlp_model.predict(X_train)

        inverted_y_train_pred = y_train_scaler.inverse_transform(y_train_pred.reshape(-1, 1)).ravel()


        y_train_residuals = y_train - inverted_y_train_pred


        y_test_pred = final_mlp_model.predict(X_test)
        inverted_y_test_pred = y_test_scaler.inverse_transform(y_test_pred.reshape(-1,1)).ravel()




        y_test_residuals = y_test - inverted_y_test_pred



        train_residual_key_name = "cy3_" + "mlp_" + mapping_name + "_train_residuals_" + f"r{repeat}"
        train_residuals[train_residual_key_name] = y_train_residuals

        
        test_residual_key_name = "cy3_" + "mlp_" + mapping_name + "_test_residuals_" + f"r{repeat}"
        test_residuals[test_residual_key_name] = y_test_residuals

        




        r2 = final_mlp_model.score(X_test,scaled_y_test_data.ravel())

        data_row = ['cy3','mlp',mapping_name,r2,1,repeat]
        
        hyperopt_data.append(data_row)

        trials_file_name = f'cy3_mlp_{mapping_name}_{repeat}_trials.pkl'

        with open(trials_file_name,"wb") as outfile:
            pickle.dump(trials,outfile)

        





df_cy5 = pd.read_csv('../data/3_prime_Cy5.csv')



for idx in range(0,len(mappings)):

    seq_transform_function = mappings[idx]
    mapping_name = mapping_names[idx]

    transformed_df = sequence_transform(dataset=df_cy5, mapping=seq_transform_function,
                                sequence_column_idx=0,intensity_column_idx=1)

    X_train, X_test, y_train, y_test = train_test_split(transformed_df.iloc[:,:-1], 
                                                                    transformed_df.iloc[:,-1],
                                                                    test_size=0.30, random_state=42)
    

    qt = QuantileTransformer(n_quantiles=5, random_state=0, output_distribution='normal')
    X_train = qt.fit_transform(X_train)


    y_train_scaler = StandardScaler()
    scaled_y_train_data = y_train_scaler.fit_transform(y_train.to_numpy().reshape(-1, 1))

    
    X_test = qt.fit_transform(X_test)


    y_test_scaler = StandardScaler()
    scaled_y_test_data = y_test_scaler.fit_transform(y_test.to_numpy().reshape(-1, 1))

    
    print(f'Currently Working on the following sequence transformation: {mapping_names[idx]}')
    print("\n")

    

    for repeat in range(0,10):


        mlp = MLPRegressor()

        trials = Trials()
        best = fmin(fn=lambda p:safe_objective_function(
            model_instance=mlp,
            parameters=p,
            X_train=X_train,
            y_train=scaled_y_train_data.ravel()
        ),
                    space=mlp_space,
                    algo=tpe.suggest,
                    max_evals=100,
                    trials=trials,
                    timeout=300,
                    rstate=None,
                    early_stop_fn=no_progress_loss(10),
                    return_argmin=True)
        
        
        
        best_param_values = space_eval(mlp_space, best)
        best_layers = tuple(int(x) for x in best_param_values['hidden_layer_sizes'])
        best_param_values['hidden_layer_sizes'] = best_layers

        final_mlp_model = MLPRegressor(**best_param_values)
        final_mlp_model.fit(X_train, scaled_y_train_data.ravel())

        y_train_pred = final_mlp_model.predict(X_train)

        inverted_y_train_pred = y_train_scaler.inverse_transform(y_train_pred.reshape(-1, 1)).ravel()


        y_train_residuals = y_train - inverted_y_train_pred


        y_test_pred = final_mlp_model.predict(X_test)
        inverted_y_test_pred = y_test_scaler.inverse_transform(y_test_pred.reshape(-1,1)).ravel()

        y_test_residuals = y_test - inverted_y_test_pred


        train_residual_key_name = "cy5_" + "mlp_" + mapping_name + "_train_residuals_" + f"r{repeat}"
        train_residuals[train_residual_key_name] = y_train_residuals

        test_residual_key_name = "cy5_" + "mlp_" + mapping_name + "_test_residuals_" + f"r{repeat}"
        test_residuals[test_residual_key_name] = y_test_residuals

        r2 = final_mlp_model.score(X_test,scaled_y_test_data.ravel())
        data_row = ['cy5','mlp',mapping_name,r2,1,repeat]

        hyperopt_data.append(data_row)

        trials_file_name = f'cy5_mlp_{mapping_name}_{repeat}_trials.pkl'

        with open(trials_file_name,"wb") as outfile:
            pickle.dump(trials,outfile)





output_df = pd.DataFrame(hyperopt_data)

colnames = ['dataset', 'model_name', 'mapping_name', 'r2', 'polynomial_degree','repeat_num']
output_df.to_csv(path_or_buf='mlp_hyperopt_data.csv',header=colnames)


mlp_train_residuals_df = pd.DataFrame(train_residuals)
mlp_train_residuals_df.to_csv(path_or_buf='mlp_train_residuals.csv')

mlp_test_residuals_df = pd.DataFrame(test_residuals)
mlp_test_residuals_df.to_csv(path_or_buf='mlp_test_residuals.csv')