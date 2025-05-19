# This module defines a base class for calculating the log likelihood of continuous probability distributions.
# The LogLikelihood class is an abstract base class that requires subclasses to implement the _compute_likelihood method,
# which calculates the likelihood of given data based on specific parameters (theta).
# 
# The class uses a singleton pattern to ensure that only one instance of LogLikelihood can be created.
# It contains a nested Dataset class, which is a TypedDict that holds the source of the data and the data array itself.
# The constructor of the Dataset class initializes the log likelihood calculation and stores relevant details.
# A factory method, for_parameters_and_datasets, is provided to create instances of the LogLikelihood class with the necessary parameters and datasets.

from abc import ABC, abstractmethod
from typing import TypedDict, List, Dict
import numpy as np

class LogLikelihood(ABC):
    """
    Base class for loglikelihood function for any continuous probability distribution.
    This class must be extended and _compute_likelihood function must be overridden.
    """

    # This line creates a unique instance key for the LogLikelihood class, which can be used to ensure that 
    # only one instance of the class exists (singleton pattern) or to differentiate instances in some way.
    __instance_key = object()

    class Dataset(TypedDict):
        """
        Key-valued dataset. 
        'source' is the name of the source of the data and 'x' is the data array.
        """
        source: str
        x: List

    def __init__(self, instance_key, theta_sets: Dict[str, List], datasets: List[Dataset]):
        """
        Initialize the LogLikelihood class.
        """
        if instance_key != LogLikelihood.__instance_key:
            raise ValueError("Invalid instance key, use the instantiate function")
        
        self.theta_sets = theta_sets
        self.datasets = datasets
        self._total_loglikelihood = self._compute_total_loglikelihood()
        self._max_loglikelihood_details = self._get_max_loglikelihoods()

    @abstractmethod
    def _compute_likelihood(self, x, **theta):
        """
        Subclass should override this method to compute the likelihood of the data.
        """
        ...

    @classmethod
    def for_parameters_and_datasets(cls, theta_sets: Dict[str, List], datasets: List[Dataset]):
        """
        Factory method to create an instance of the LogLikelihood class.
        """
        return cls(LogLikelihood.__instance_key, theta_sets, datasets)
        

    # This function prepares combinations of parameters (theta) from the supplied simulated values.
    # It uses numpy's meshgrid to create a grid of parameter combinations. For each parameter in 
    # theta_sets, it generates a grid of values, allowing for the exploration of all possible 
    # combinations of parameters. The resulting combinations are returned as a dictionary, where 
    # each key corresponds to a parameter name and the value is a flattened array of the combinations.
    def _prepare_combinations_for_theta(self) -> Dict[str, List]:
        """
        Prepare combinations of parameters from the list of supplied simulated values
        For example, if we have two parameters theta1 and theta2, and we have 3 simulated values for theta1
        and 2 simulated values for theta2, we will have 6 combinations.
        
        This function returns combinations as a dictionary of values, keeping the positional indices intact"""
        
        theta_grid = None
        theta_name_grid_index = {}

        for i, (theta_name, theta_values) in enumerate(self.theta_sets.items()):
            if i == 0:
                theta_grid = np.meshgrid(theta_values)
            else:
                theta_grid = np.meshgrid(theta_grid, theta_values)

            theta_name_grid_index[theta_name] = i

        return {theta_name: theta_grid[theta_index].flatten() for theta_name, theta_index in theta_name_grid_index.items()}    
        
    def _compute_total_loglikelihood(self):
        """
        Compute the total loglikelihood for all data sources in self._datasets
        It uses the _prepare_combinations_for_theta function to get the combinations of parameters
        """

        total_loglikelihood = {}

        def _get_single_name_value_for_theta(index, theta_combinations):
            """
            Get the value of a single parameter for a given index and theta combinations.
            This function retrieves the value of each parameter at a specific index from the provided
            dictionary of parameter combinations (theta_combinations).
            """
            # Create a dictionary comprehension to construct a new dictionary
            return {
                # For each key-value pair in the theta_combinations dictionary,
                # the key (parameter name) is kept the same, and the value is taken
                # from the list of values at the specified index.
                theta_combinations_k: theta_combinations_v[index]
                for theta_combinations_k, theta_combinations_v in theta_combinations.items()
            }
        theta_combinations = self._prepare_combinations_for_theta()
        num_theta_values = len(list(theta_combinations.values())[0])

        # Create dictionaries of tuples with format (theta, likelihood) for each data source
        # Iterate over each dataset in the list of datasets stored in self._datasets
        for ds in self.datasets:
            # Create a list comprehension to calculate the loglikelihood for each combination of parameters
            llh = [
                (
                    # Get the current combination of parameters for the index 'i'
                    _get_single_name_value_for_theta(i, theta_combinations),
                    # Calculate the loglikelihood for the observations in the current dataset 'ds'
                    # using the parameters obtained from the previous function call
                    self.get_loglikelihood_for_observations(ds['x'], **_get_single_name_value_for_theta(i, theta_combinations)),
                )
                # Loop over the range of the number of theta values to create combinations
                for i in range(num_theta_values)
            ]

            # Store the loglikelihood results in the total_llh dictionary, using the source of the dataset as the key
            total_llh[ds['source']] = llh

            # Return the dictionary containing total loglikelihoods for all datasets
            return total_llh

        def get_loglikelihood_for_observations(self, x, **theta):
            """
            Compute the loglikelihood for a given set of observations and parameters.
            """
            return np.sum(np.log(self._compute_likelihood(x, **theta)))
        
        def _get_max_loglikelihoods(self):
            """
            Iterate over all log-likelihoods and return the maximum loglikelihood for each data source.
            """
            # Create a dictionary comprehension that iterates over each key-value pair in self._total_loglikelihood
            return {
                # For each key 'k' (data source) in the dictionary, find the maximum log-likelihood value
                # from the list 'v' (which contains tuples of (theta, likelihood)) using the max function.
                # The key argument specifies that the maximum should be determined based on the second element
                # of each tuple (the likelihood value), which is accessed with t[1].
                k: max(v, key=lambda t: t[1]) for k, v in self._total_loglikelihood.items()
            }
            

from scipy.stats import expon

class ExponentialLogLikelihoodFunctionAnalysis(LogLikelihood):
    """
    Class for studying the likelihood function of Exponential distribution with parameter lambd
    """

    def _compute_likelihood(self, x, lambd):
        """
        Compute the likelihood function for the Exponential distribution.
        """
        return expon.pdf(x, loc=0, scale=1/lambd)
    








# Test the ExponentialLogLikelihoodFunctionAnalysis class

def test_exponential_loglikelihood_function_analysis():
    datasets = [
        {
            'source': 'Dataset 1',
            'x': np.linspace(start=200, stop=300, num=1000)
        },
        {
            'source': 'Dataset 2',
            'x': np.linspace(start=2, stop=8, num=1000)
        },
    ]
    
    theta_sets = {'lambd': np.linspace(start=0, stop=3, num=500)}

    analysis = ExponentialLogLikelihoodFunctionAnalysis.for_parameters_and_datasets(theta_sets=theta_sets, datasets=datasets).plot()

if __name__ == "__main__":
    test_exponential_loglikelihood_function_analysis()