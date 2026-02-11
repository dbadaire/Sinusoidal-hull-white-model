"""
Moteur de simulation Monte Carlo pour le pricing d'obligations.
Implémente le schéma d'Euler-Maruyama pour les processus Hull-White.
"""

import numpy as np
from typing import Union
from src.hull_white import HullWhiteModel
from src.sinusoidal_hw import SinusoidalHullWhiteModel

class MonteCarloPricer:
    def __init__(self, model: Union[HullWhiteModel, SinusoidalHullWhiteModel], 
                 n_paths: int = 200, dt: float = 0.05):
        """
        Args:
            model: Instance du modèle (Standard ou Sinusoidal).
            n_paths (int): Nombre de simulations.
            dt (float): Pas de temps en années (Papier: 0.05).
        """
        self.model = model
        self.n_paths = n_paths
        self.dt = dt

    def simulate_short_rates(self, r0: float, T_horizon: float) -> np.ndarray:
        """
        Simule les trajectoires du taux court r_t jusqu'à T_horizon.
        Retourne une matrice (n_steps, n_paths).
        """
        n_steps = int(T_horizon / self.dt)
        rates = np.zeros((n_steps + 1, self.n_paths))
        rates[0, :] = r0
        
        # Pré-génération des aléas (Brownien)
        # Z ~ N(0, 1)
        Z = np.random.normal(0, 1, (n_steps, self.n_paths))
        sqrt_dt = np.sqrt(self.dt)

        for i in range(n_steps):
            t = i * self.dt
            r_prev = rates[i, :]
            
            # Détermination du kappa courant
            if isinstance(self.model, SinusoidalHullWhiteModel):
                # Omega est passé en rad/jour, converti ici en rad/an pour t en années
                # Attention : Pour simplifier, on assume que self.model.omega est déjà 
                # à la bonne échelle ou que t est traité uniformément.
                # Vu votre omega ~0.0008 (rad/jour), il faut faire * 365.
                omega_annual = self.model.omega * 365.0
                k_t = self.model.kappa_0 + self.model.A * np.sin(omega_annual * t)
            else:
                k_t = self.model.kappa

            # Drift: k_t * (theta - r_t)
            drift = k_t * (self.model.theta - r_prev) * self.dt
            
            # Diffusion: sigma * dW
            diffusion = self.model.sigma * sqrt_dt * Z[i, :]
            
            # Euler-Maruyama
            rates[i + 1, :] = r_prev + drift + diffusion

        return rates

    def price_bond(self, r0: float, T_maturity: float) -> float:
        """
        Calcule le prix P(0, T) par espérance des facteurs d'actualisation.
        P(0,T) = E[ exp( - integral(r_s ds) ) ]
        """
        rates = self.simulate_short_rates(r0, T_maturity)
        
        # Approximation de l'intégrale par somme de Riemann
        # On exclut le dernier point pour la somme (convention left-point)
        integral_r = np.sum(rates[:-1, :], axis=0) * self.dt
        discount_factors = np.exp(-integral_r)
        price = np.mean(discount_factors)
        return price