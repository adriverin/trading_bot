import numpy as np
import pandas as pd
from plotting import *


def characteristic_function():
    # Generate frequencies from a given distribution
    # Frequency lies between -inf and inf; here we use -100 and 100 for the time being

    omega_array = np.linspace(start=-100, stop=100, num=100)
    
    def _cf_gaussian(omega, mu, sigma2):
        return np.exp(1j * omega * mu - 0.5 * sigma2 * omega**2)
    
    def _cf_uniform(omega, b, a):
        return (np.exp(1j * omega * b) - np.exp(1j * omega * a)) / (1j * omega) / (b - a)
    
    def _cf_exponential(omega, lambd): 
        return lambd / (lambd - 1j * omega)

    def _cf_poisson(omega, lambd):
        return np.exp(lambd * (np.exp(1j * omega) - 1))
    
    def _generate_cf_omega_values(cf_omega, **theta):
        """ 
        Generate cf(omega) values for all given frequencies and use the real part only for plotting
        """
        
        return pd.DataFrame([{'omega': omega, 'cf(omega)': cf_omega(omega, **theta).real} for omega in omega_array])
    


    if __name__ == "__main__":
        mu = 200
        sigma2 = 10
        b = 990
        a = 90
        lambd = 10.0
        poisson_lambda = 10.0

        cf_gaussian = (' Gaussian phi(omega; mu, sigma2)', _generate_cf_omega_values(_cf_gaussian, mu=mu, sigma2=sigma2))
        cf_uniform = (' Uniform phi(omega; a, b)', _generate_cf_omega_values(_cf_uniform, a=a, b=b))
        cf_exponential = (' Exponential phi(omega; lambd)', _generate_cf_omega_values(_cf_exponential, lambd=lambd))
        cf_poisson = (' Poisson phi(omega; lambd)', _generate_cf_omega_values(_cf_poisson, lambd=poisson_lambda))

        plot_characteristic_function([cf_gaussian, cf_uniform, cf_exponential, cf_poisson])    

characteristic_function()
