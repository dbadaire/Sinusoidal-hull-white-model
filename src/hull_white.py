"""
Implémentation du modèle Hull-White standard (1 facteur).
Utilise la solution analytique fermée pour le pricing des obligations Zéro-Coupon.
"""

import numpy as np

class HullWhiteModel:
    def __init__(self, kappa: float, theta: float, sigma: float):
        """
        Args:
            kappa (float): Vitesse de retour à la moyenne (Mean reversion speed).
            theta (float): Taux moyen long terme (Long-term mean rate).
            sigma (float): Volatilité (Volatility).
        """
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma

    def B(self, t: float, T: float) -> float:
        """
        Calcule le terme B(t,T) de la solution affine.
        Eq: B(t,T) = (1 - exp(-kappa * (T - t))) / kappa
        """
        dt = T - t
        return (1 - np.exp(-self.kappa * dt)) / self.kappa

    def A(self, t: float, T: float) -> float:
        """
        Calcule le terme A(t,T) de la solution affine.
        D'après le papier (Section 5.1, Eq pour A(t,T)).
        """
        dt = T - t
        B_val = self.B(t, T)
        
        # Terme 1: Ajustement de convexité et moyenne
        # log A(t,T) = (theta - sigma^2 / (2*kappa^2)) * (B(t,T) - dt) - (sigma^2 * B(t,T)^2) / (4*kappa)
        term1 = (self.theta - (self.sigma**2) / (2 * self.kappa**2)) * (B_val - dt)
        term2 = (self.sigma**2 * B_val**2) / (4 * self.kappa)
        
        return np.exp(term1 - term2)

    def price_zero_coupon(self, t: float, T: float, r_t: float) -> float:
        """
        Calcule le prix P(t,T) d'un bond Zéro-Coupon.
        """
        if t == T:
            return 1.0
        A_val = self.A(t, T)
        B_val = self.B(t, T)
        return A_val * np.exp(-B_val * r_t)