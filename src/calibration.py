"""
Module de calibration des modèles Hull-White.
Utilise l'optimisation Nelder-Mead pour minimiser l'écart entre prix modèle et prix marché.
"""

import numpy as np
from scipy.optimize import minimize
from typing import List, Dict, Tuple
from src.hull_white import HullWhiteModel
from src.sinusoidal_hw import SinusoidalHullWhiteModel
from src.monte_carlo import MonteCarloPricer

class Calibrator:
    def __init__(self, maturities: List[float], observed_yields: List[float]):
        """
        Args:
            maturities (List[float]): Maturités en années (ex: [1, 2, ..., 30]).
            observed_yields (List[float]): Taux observés correspondant (en décimales, ex: 0.0122).
        """
        self.maturities = np.array(maturities)
        self.observed_yields = np.array(observed_yields)
        # Calcul des prix de marché observés : P = exp(-y * T)
        self.observed_prices = np.exp(-self.observed_yields * self.maturities)

    def _rmse(self, model_prices: np.array) -> float:
        """Calcule la Root Mean Squared Error entre prix modèle et marché."""
        return np.sqrt(np.mean((model_prices - self.observed_prices)**2))

    def calibrate_standard_hw(self, r0: float) -> Dict:
        """
        Calibre le modèle Hull-White Standard (Solution Analytique).
        Optimise: kappa, theta, sigma.
        """
        # Fonction objectif à minimiser
        def objective(params):
            kappa, theta, sigma = params
            hw = HullWhiteModel(kappa, theta, sigma)
            model_prices = []
            for T in self.maturities:
                # Utilisation de la formule fermée
                price = hw.price_zero_coupon(t=0, T=T, r_t=r0)
                model_prices.append(price)           
            return self._rmse(np.array(model_prices))

        # Initial Guesses
        initial_guess = [0.1, 0.03, 0.01] # kappa, theta, sigma
        # Bounds kappa [0.01, 5], theta [0.001, 0.2], sigma [0.001, 0.05])
        bounds = [(0.01, 5.0), (0.001, 0.2), (0.001, 0.05)]

        result = minimize(objective, initial_guess, method='Nelder-Mead', 
                          bounds=bounds, tol=1e-3)
        return {
            "kappa": result.x[0],
            "theta": result.x[1],
            "sigma": result.x[2],
            "rmse": result.fun,
            "success": result.success
        }

    def calibrate_sinusoidal_hw(self, r0: float, fixed_omega: float) -> Dict:
        """
        Calibre le modèle Sinusoidal HW (Monte Carlo).
        Optimise: kappa_0, A, theta, sigma.
        Omega est FIXÉ selon l'analyse de périodicité.
        """
        # Fonction objectif
        def objective(params):
            kappa_0, A, theta, sigma = params
            model = SinusoidalHullWhiteModel(kappa_0=kappa_0, A=A, omega=fixed_omega, theta=theta, sigma=sigma)
            
            # Utilisation du Pricer Monte Carlo (200 paths)
            mc_pricer = MonteCarloPricer(model, n_paths=200, dt=0.05)
            
            model_prices = []
            for T in self.maturities:
                # Simulation MC pour chaque maturité
                price = mc_pricer.price_bond(r0=r0, T_maturity=T)
                model_prices.append(price)
            
            err = self._rmse(np.array(model_prices))
            #print(f"  Iter RMSE: {err:.5f} | Params: k0={kappa_0:.3f}, A={A:.3f}, th={theta:.3f}, sig={sigma:.3f}")
            return err

        # Initial Guesses
        initial_guess = [0.3, 0.2, 0.03, 0.01] 
        # Bounds: A in [0, 1]
        bounds = [(0.01, 5.0), (0.0, 1.0), (0.001, 0.2), (0.001, 0.05)]

        print(f"Début calibration Sinusoidal HW (Monte Carlo, omega={fixed_omega:.6f})...")
        
        result = minimize(objective, initial_guess, method='Nelder-Mead', 
                          bounds=bounds, tol=1e-3, 
                          options={'maxiter': 50}) 
        return {
            "kappa_0": result.x[0],
            "A": result.x[1],
            "omega": fixed_omega,
            "theta": result.x[2],
            "sigma": result.x[3],
            "rmse": result.fun,
            "success": result.success
        }