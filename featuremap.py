import util
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')


factor = 2.0


class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta

    def fit(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        Xt = X.transpose()
        self.theta = np.linalg.solve(np.matmul(Xt,X), np.matmul(Xt, y))
        return self.theta

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        if k < 2:
            return X
        output = np.zeros((X.shape[0], k+1))
        output[:, 0:2] = X
        for i in range(2, k+1):
            output[:,i] = np.power(X[:,1], i)
        return output

    def create_sin(self, k, X):
        """
        Generates a sin with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        if k < 2:
            output = np.zeros((X.shape[0], 3))
            output[:, 0:2] = X
            output[:,2] = np.sin(X[:,1])
            return output
        
        output = np.zeros((X.shape[0], k+2))
        output[:, 0:2] = X
        for i in range(2, k+1):
            output[:,i] = np.power(X[:,1], i)
        output[:,k+1] = np.sin(X[:,1])
        return output

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        return np.matmul(X, self.theta)


def run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.png'):
    train_x, train_y=util.load_dataset(train_path,add_intercept=True)
    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-factor*np.pi, factor*np.pi, 1000)
    plt.figure()
    plt.scatter(train_x[:, 1], train_y)

    for k in ks:
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''
        linearModel = LinearModel()
        if sine:
            X = linearModel.create_sin(k, train_x)
            valid_X = linearModel.create_sin(k, plot_x)
        else:
            X = linearModel.create_poly(k, train_x)
            valid_X = linearModel.create_poly(k, plot_x)
        
        linearModel.fit(X, train_y)
        plot_y = linearModel.predict(valid_X)
        
        '''
        Here plot_y are the predictions of the linear model on the plot_x data
        '''
        plt.ylim(-2, 2)
        plt.plot(plot_x[:, 1], plot_y, label='k=%d' % k)

    plt.legend()
    plt.savefig(filename)
    plt.clf()

def main(train_path, small_path, eval_path):
    '''
    Run all experiments
    '''
    # Test run
    run_exp(train_path, sine=False, ks=[3], filename='large-poly3.png')
    
    # Run large_data_poly
    run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='large-poly.png')
    
    # Run large_data_sine
    run_exp(train_path, sine=True, ks=[1, 2, 3, 5, 10, 20], filename='large-sine.png')
    
    # Run small_data_poly
    run_exp(small_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='small-poly.png')
    
    # Run small_data_sine
    run_exp(small_path, sine=True, ks=[1, 2, 3, 5, 10, 20], filename='small-sine.png')


if __name__ == '__main__':
    main(train_path='train.csv',
        small_path='small.csv',
        eval_path='test.csv')
