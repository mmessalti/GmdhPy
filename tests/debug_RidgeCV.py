from gmdhpy.gmdh import Regressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics



def plot_th_scatter(x_plot,y_plot,x_scatter, y_scatter, title='' ): 
    # setting the axes at the centre
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.title(title) 

    # plot the function
    plt.plot(x_plot,y_plot, 'r')
    plt.scatter(x_scatter, y_scatter,color="black")
    
    # show the plot
    plt.show()

def plot_th_scatter_black_blue_green(x_plot,y_plot,x_scatter, y_scatter, title='' ): 
    # setting the axes at the centre
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.title(title) 

    # plot the function
    plt.plot(x_plot,y_plot, 'r')
    plt.scatter(x_scatter, y_scatter,color="black")
    
    # show the plot
    plt.show()

if __name__ == '__main__':
    
    # Construction of My Artificial Dataset
    
    dataset_size = 100
    x                = np.linspace(0,1,dataset_size) # 100 linearly spaced numbers between 0.0 and 1.0
    x_theorical_plot = np.linspace(0,1,dataset_size)
    np.random.seed(101)
    np.random.shuffle(x)

    y1_th = np.exp(x) * np.cos( 2 * np.pi * np.sin(np.pi*x) ) + 5 
    y1_theorical_plot = np.exp(x_theorical_plot) * np.cos( 2 * np.pi * np.sin(np.pi*x_theorical_plot) ) + 5 

    y2_th = np.exp(x) * np.sin( 2 * np.pi * np.cos(np.pi*x) ) + 5 
    y2_theorical_plot = np.exp(x_theorical_plot) * np.sin( 2 * np.pi * np.cos(np.pi*x_theorical_plot) ) + 5

    noise1 = np.random.normal(0,0.2,x.size)
    y1_noisy = y1_th + noise1
    y2_noisy = y2_th + noise1

    plot_th_scatter(x_plot=x_theorical_plot, y_plot=y1_theorical_plot, x_scatter=x , y_scatter=y1_noisy, title='Artificial DataSet 1')
    plot_th_scatter(x_plot=x_theorical_plot, y_plot=y2_theorical_plot, x_scatter=x , y_scatter=y2_noisy, title='Artificial DataSet 2')

    ones = np.ones(x.shape)
    x_square = x**2
    temp = np.vstack((ones, x, x_square))
    dataset = temp.T

    # Train and Test Set
    n_samples = dataset.data.shape[0]
    train_data_is_the_first_half = False
    n = n_samples // 2
    if train_data_is_the_first_half:
        train_x = dataset[:n]
        train_y1 = y1_noisy[:n]
        train_y2 = y2_noisy[:n]
        test_x = dataset[n:]
        test_y1 = y1_noisy[n:]
        test_y2 = y2_noisy[n:]
    else:
        train_x = dataset[n:]
        train_y1 = y1_noisy[n:]
        train_y2 = y2_noisy[n:]
        test_x = dataset[:n]
        test_y1 = y1_noisy[:n]
        test_y2 = y2_noisy[:n]
    feature_names = ['ones','x','x*x']

    # Models

    model_y1 = Regressor(ref_functions=('linear_cov',),
                    normalize=True,
                    criterion_minimum_width=5,
                    stop_train_epsilon_condition=0.0001,
                    layer_err_criterion='top',
    #                  l2=0.01,
                    l2_bis=(0.0001,0.001,0.01,0.1,1.0,10.0),
                    feature_names=feature_names )
    model_y1.fit(train_x, train_y1)
    print()
    print("model_y1 :")
    print(model_y1.describe())

    # Now predict the value of the second half:
    y1_pred = model_y1.predict(test_x)

    # Selected/unselected features:
    print("Selected features: {}".format(model_y1.get_selected_features()))
    print("Unselected features: {}".format(model_y1.get_unselected_features()))


    # error
    mae = metrics.mean_absolute_error(test_y1, y1_pred)
    mse = metrics.mean_squared_error(test_y1, y1_pred)

    print("mae error on test set: {mae:0.2f}".format(mae=mae))
    print("mse error on test set: {mse:0.2f}".format(mse=mse))


    # setting the axes at the centre
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # plot the function
    plt.scatter(train_x[:,1], train_y1,color="black")
    plt.scatter(test_x[:,1], y1_pred,color="blue")
    # plt.scatter(test_x[:,1], test_y1,color="green")
    plt.plot(x_theorical_plot,y1_theorical_plot, 'r')

    # show the plot
    plt.show()

    print()
    print(' end of debug_RidgeCV.py')
