"""
Définition du modèle Sinusoidal Hull-White.
Extension avec retour à la moyenne variable dans le temps : kappa_t = kappa_0 + A * sin(omega * t)
"""

import numpy as np

class SinusoidalHullWhiteModel:
    def __init__(self, kappa_0: float, A: float, omega: float, theta: float, sigma: float):
        """
        Args:
            kappa_0 (float): Vitesse de base (Baseline mean reversion).
            A (float): Amplitude de la variation sinusoïdale.
            omega (float): Fréquence angulaire (calibrée sur le cycle ~20-22 ans).
            theta (float): Moyenne long terme.
            sigma (float): Volatilité constante.
        """
        self.kappa_0 = kappa_0
        self.A = A
        self.omega = omega
        self.theta = theta
        self.sigma = sigma

    def kappa_t(self, t: float) -> float:
        """
        Retourne la vitesse de retour à la moyenne instantanée.
        Eq: kappa_t = kappa_0 + A * sin(omega * t)
        """
        # Dans ce code, on suppose t en ANNÉES.
        # Omega doit donc être en radians/AN pour le calcul sin(omega * t).
        # Si omega est fourni en rad/jour, il faut le multiplier par 365.
        return self.kappa_0 + self.A * np.sin(self.omega * t)