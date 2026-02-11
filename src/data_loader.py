import pandas as pd
import numpy as np
import os

class YieldDataLoader:
    """
    Charge et nettoie les données de taux du Trésor US (FRED).
    """
    def __init__(self, file_path):
        self.file_path = file_path
        # Mapping
        self.column_map = {
            'DGS1': '1Y',
            'DGS2': '2Y',
            'DGS3': '3Y',
            'DGS5': '5Y',
            'DGS7': '7Y',
            'DGS10': '10Y',
            'DGS20': '20Y',
            'DGS30': '30Y'
        }

    def load_data(self):
        """
        Lit le CSV, filtre les dates et nettoie les valeurs non numériques.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Le fichier {self.file_path} est introuvable.")

        try:
            # index_col=0 prend la 1ère colonne (les dates) quel que soit son nom
            # parse_dates=True demande à pandas de parser l'index comme des dates
            df = pd.read_csv(self.file_path, index_col=0, parse_dates=True)
        except Exception as e:
            raise ValueError(f"Erreur lors de la lecture du CSV. Vérifiez le format. Détails: {e}")

        df.index.name = 'DATE'
        df = df.rename(columns=self.column_map)

        # Filtrage des colonnes pour ne garder que celles du papier
        cols_to_keep = list(self.column_map.values())
        available_cols = [c for c in cols_to_keep if c in df.columns]
        if not available_cols:
            raise ValueError(f"Aucune colonne valide trouvée. Colonnes présentes : {df.columns.tolist()}")
            
        df = df[available_cols]

        # [cite_start]Filtrage Temporel: Jan 1990 - Dec 2022 [cite: 8]
        # On s'assure que l'index est bien trié avant de slicer
        df = df.sort_index()
        try:
            df = df.loc['1990-01-01':'2022-12-31']
        except KeyError:
            print("Attention: La plage de dates demandée n'est pas entièrement couverte.")

        # Nettoyage et conversion en float
        df = df.replace('.', np.nan)
        df = df.astype(float)

        # Suppression des lignes avec NaN
        original_len = len(df)
        df = df.dropna()
        print(f"Données chargées. {original_len - len(df)} lignes supprimées (NaN/Jours fériés).")
        
        if not df.empty:
            print(f"Période: {df.index.min().date()} à {df.index.max().date()}")

        # Conversion des taux: Le modèle mathématique travaille en décimales (ex: 1.22% -> 0.0122)
        df = df / 100.0

        return df

