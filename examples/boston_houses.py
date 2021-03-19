# -*- coding: utf-8 -*-
from __future__ import print_function
from sklearn.datasets import load_boston
from sklearn import metrics
from gmdhpy.gmdh import Regressor
from gmdhpy.plot_model import PlotModel


if __name__ == '__main__':

    boston = load_boston()

    n_samples = boston.data.shape[0]

    train_data_is_the_first_half = False
    n = n_samples // 2
    if train_data_is_the_first_half:
        train_x = boston.data[:n]
        train_y = boston.target[:n]
        test_x = boston.data[n:]
        test_y = boston.target[n:]
    else:
        train_x = boston.data[n:]
        train_y = boston.target[n:]
        test_x = boston.data[:n]
        test_y = boston.target[:n]

    params = {
        'admix_features': True,                   # default value
        'criterion_type': 'validate',             # default value
        'seq_type' : 'mode1' ,                    # default value
        'max_layer_count': 100,                   # default value is sys.maxsize
        'criterion_minimum_width': 5,             # default value
        'stop_train_epsilon_condition' : 0.0001,  # default value is 0.001
        'manual_best_neurons_selection' : False,  # default value
        'ref_functions': 'linear_cov',            # default value
        'normalize': True,                        # default value
        'layer_err_criterion': 'top',             # default value
        'n_jobs': 1,                              # default value
        'feature_names': boston.feature_names,
        'l2_bis':(1e-5,1e-4,1e-3,0.01,0.1,1.0,10.0)
    }

    
    model = Regressor(**params)
    '''
    model = Regressor(ref_functions=('linear_cov',),
                     criterion_type='validate',
                      feature_names=boston.feature_names,
                      criterion_minimum_width=5,
                      stop_train_epsilon_condition=0.001,
                      layer_err_criterion='top',
                      l2=0.5,
                      n_jobs='max')
    '''
    model.fit(train_x, train_y)

    # Now predict the value of the second half:
    y_pred = model.predict(test_x)
    mse = metrics.mean_squared_error(test_y, y_pred)
    mae = metrics.mean_absolute_error(test_y, y_pred)
    r2  = metrics.r2_score(test_y, y_pred)

    print("mse error on test set: {mse:0.2f}".format(mse=mse))
    print("mae error on test set: {mae:0.2f}".format(mae=mae))
    print("R²  score on test set: {r2:0.4f}".format(r2=r2))

    print(model.get_selected_features_indices())
    print(model.get_unselected_features_indices())

    print("Selected features: {}".format(model.get_selected_features()))
    print("Unselected features: {}".format(model.get_unselected_features()))
    print()
    print()
    print()
    print(model.describe())

    PlotModel(model, filename='boston_house_model', plot_neuron_name=True, view=True).plot()
