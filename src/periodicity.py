"""
Module d'analyse de la périodicité pour le modèle Sinusoidal Hull-White.
Implémente la Transformée de Fourier (FFT) et les tests de stationnarité.
"""

from typing import Tuple, Optional, Dict, Union
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.stats.diagnostic import acorr_ljungbox

class PeriodicityAnalyzer:
    """
    Classe dédiée à l'analyse spectrale et statistique des séries temporelles de taux.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialise l'analyseur avec le DataFrame des taux.
        Args:
            data (pd.DataFrame): DataFrame avec les dates en index et les maturités en colonnes.
        """
        self.data = data
        self._sampling_rate = 1 / 365.25  # Fréquence journalière en années

    def _get_clean_series(self, tenor: str) -> np.ndarray:
        """
        Extrait et nettoie une série temporelle pour une maturité donnée.
        Raises:
            ValueError: Si la maturité (tenor) n'existe pas.
        """
        if tenor not in self.data.columns:
            raise ValueError(f"La maturité '{tenor}' est absente des données. Colonnes disponibles:{self.data.columns.tolist()}")
        
        # Suppression des NaNs pour garantir la stabilité de la FFT
        return self.data[tenor].dropna().values

    def compute_fft(self, tenor: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcule la Transformée de Fourier Rapide (FFT) sur la série dé-trendée.

        La série est centrée (soustraction de la moyenne) avant l'analyse pour
        éliminer la composante DC (fréquence 0).

        Args:
            tenor (str): La maturité à analyser (ex: '30Y').

        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - periods: Tableau des périodes en années.
                - magnitude: Tableau des magnitudes spectrales correspondantes.
        """
        series = self._get_clean_series(tenor)
        
        # Detrending (Soustraction de la moyenne)
        signal = series - np.mean(series)
        n_samples = len(signal)

        if n_samples == 0:
            return np.array([]), np.array([])

        # Calcul de la FFT
        yf = fft(signal)
        xf = fftfreq(n_samples, self._sampling_rate)

        # Filtrage pour ne garder que le spectre positif (Nyquist)
        mask = xf > 0
        frequencies = xf[mask]
        magnitude = np.abs(yf[mask])

        # Conversion Fréquence -> Période (T = 1/f)
        periods = 1 / frequencies

        return periods, magnitude

    def find_dominant_period(self, tenor: str, 
                           min_period: float = 1.0, 
                           max_period: float = 30.0) -> Optional[float]:
        """
        Identifie la période dominante (pic spectral) dans une plage donnée.

        Args:
            tenor (str): Maturité cible.
            min_period (float): Borne inférieure de recherche (années).
            max_period (float): Borne supérieure de recherche (années).

        Returns:
            Optional[float]: La période dominante en années, ou None si aucun pic.
        """
        periods, magnitude = self.compute_fft(tenor)
        
        # Masque booléen pour la plage de recherche
        mask = (periods >= min_period) & (periods <= max_period)
        
        if not np.any(mask):
            return None

        filtered_periods = periods[mask]
        filtered_mag = magnitude[mask]

        # Identification de l'index du maximum local
        peak_idx = np.argmax(filtered_mag)
        
        return filtered_periods[peak_idx]

    def run_statistical_tests(self, tenor: str) -> Dict[str, float]:
        """
        Exécute les tests de stationnarité (ADF) et d'autocorrélation (Ljung-Box).

        Args:
            tenor (str): Maturité cible.

        Returns:
            Dict[str, float]: Dictionnaire contenant les p-values.
        """
        series = self._get_clean_series(tenor)
        
        # Test Augmented Dickey-Fuller (ADF)
        adf_result = adfuller(series)
        adf_pvalue = adf_result[1]

        # Test Ljung-Box (lag=30 comme dans le papier [cite: 102])
        lb_result = acorr_ljungbox(series, lags=[30], return_df=True)
        lb_pvalue = lb_result['lb_pvalue'].iloc[0]

        return {
            "ADF p-value": round(adf_pvalue, 4),
            "Ljung-Box p-value": round(lb_pvalue, 6) # Souvent 0.000
        }