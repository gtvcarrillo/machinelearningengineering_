import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

class TratamentoTabela:
    def __init__(self):
        pass

    def tratamento_dados(self, X):
        
        X.columns = [c.strip().lower() for c in X.columns]
        X = X.rename(columns={
            "distance (m)": "distance_m",
            "elapsed time (s)": "elapsed_s",
            "elevation gain (m)": "elev_m",
            "average heart rate (bpm)": "hr_bpm"
        })
        X["timestamp"] = pd.to_datetime(X["timestamp"], dayfirst=True, errors="coerce")
        X.dropna(subset=["timestamp"], inplace=True)

        return X

    def fit(self, X):
        pass

    def transform(self, X):

        X = self.tratamento_dados(X)
        
        X["dist_km"] = X["distance_m"] / 1000.0
        X["dur_h"]   = X["elapsed_s"] / 3600.0
        X["pace_min_km"] = (X["elapsed_s"] / 60.0) / X["dist_km"].replace(0, np.nan)
        X["speed_kmh"]   = X["dist_km"] / X["dur_h"].replace(0, np.nan)

        agg = X.groupby("athlete").agg(
            gender=("gender","first"),
            n_runs=("dist_km","count"),
            total_km=("dist_km","sum"),
            avg_km=("dist_km","mean"),
            std_km=("dist_km","std"),
            max_km=("dist_km","max"),
            med_pace_min_km=("pace_min_km","median"),
            avg_pace_min_km=("pace_min_km","mean"),
            std_pace_min_km=("pace_min_km","std"),
            avg_speed_kmh=("speed_kmh","mean"),
            std_speed_kmh=("speed_kmh","std"),
            avg_hr=("hr_bpm","mean"),
            std_hr=("hr_bpm","std"),
            elev_gain_total=("elev_m","sum"),
            elev_gain_avg=("elev_m","mean"),
            elev_gain_std=("elev_m","std"),
            first_ts=("timestamp","min"),
            last_ts=("timestamp","max")
        ).reset_index()
        
        weeks = ((agg["last_ts"] - agg["first_ts"]).dt.days.clip(lower=1)) / 7.0
        agg["weekly_km"] = agg["total_km"] / weeks.replace({0: np.nan})
        
        # Melhor 10k e Riegel
        runs_10k = X[(X["dist_km"] >= 9.5) & (X["dist_km"] <= 10.5)].copy()
        best10 = (runs_10k.assign(t_hours=runs_10k["elapsed_s"] / 3600.0)
                            .sort_values(["athlete", "t_hours"])
                            .drop_duplicates("athlete"))[["athlete", "dist_km", "elapsed_s"]]
        agg = agg.merge(best10.rename(columns={"dist_km": "best10_dist_km", "elapsed_s": "best10_s"}),
                        on="athlete", how="left")
        agg["best10_h"] = agg["best10_s"] / 3600.0

        return agg

class Selector:
    def __init__(self, selected_features):
        self.colunas = selected_features

    def fit(self, X):
        pass

    def transform(self, X):
        X = X[self.colunas]

        return X

class GenerateScore:
    def __init__(self, modelo):
        self.model = modelo

    def fit(self, X):
        pass

    def predict(self, X):
        proba = self.model.predict(X)
        return proba
















    
        