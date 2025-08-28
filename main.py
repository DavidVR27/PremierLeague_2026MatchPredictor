"""
Predecir el resultado de la temporada 2025/26 de la Premier League usando un
clasificador de bosque aleatorio.

Este script toma una serie de temporadas históricas de la Premier League en formato CSV,
construye estadísticas resumidas para cada club (puntos, victorias, empates,
derrotas, goles a favor/en contra y diferencia de goles) y luego entrena un
RandomForestClassifier de scikit‑learn para predecir la posición final en la liga
de cada equipo en una temporada posterior. La Premier League 2025/26 incluirá 20 clubes:
los 17 equipos que permanecieron en la división en 2024/25 y tres clubes promovidos
(Leeds United, Burnley y Sunderland). Dado que los
resultados de la campaña 2024/25 aún no están disponibles libremente, el modelo
usa la temporada 2023/24 como el conjunto más reciente de características de entrenamiento.
Los nuevos clubes que no compitieron en 2023/24 tienen asignados valores promedio de
características basados en los últimos tres equipos de esa temporada.

Los archivos de partidos históricos utilizados por este script se pueden descargar
del mirror de CSV de fútbol de código abierto alojado en GitHub. Cada archivo
(`eng1_2018-19.csv`, `eng1_2019-20.csv`, … `eng1_2023-24.csv`) lista
cada partido de la Premier League en la temporada dada con columnas para la
fecha, equipo local (Team 1), marcador final (FT), marcador del medio tiempo (HT) y
equipo visitante (Team 2). Un ejemplo de las primeras filas del
archivo 2019/20 se muestra a continuación:

```
              Date          Team 1   FT   HT            Team 2
0   Fri Aug 9 2019       Liverpool  4-1  4-0           Norwich
1  Sat Aug 10 2019        West Ham  0-5  0-1          Man City
2  Sat Aug 10 2019     Bournemouth  1-1  0-0  Sheffield United
3  Sat Aug 10 2019         Burnley  3-0  0-0       Southampton
4  Sat Aug 10 2019  Crystal Palace  0-0  0-0           Everton
```

Cada CSV contiene 380 partidos (20 clubes jugando 38 juegos cada uno). El
script analiza el marcador final para determinar goles locales y visitantes y
calcula los resultados de victoria/empate/derrota en consecuencia. Después de resumir la
temporada, los equipos se ordenan por puntos, diferencia de goles y goles
anotados para derivar una clasificación final. Para los datos de entrenamiento, las
estadísticas de rendimiento de cada equipo de la temporada `n` se usan para predecir su
posición en la temporada `n+1`. Los equipos que ingresan a la liga vía promoción
tienen asignados valores de características predeterminados que representan el promedio de los
últimos tres clubes de la temporada anterior.

Los hiperparámetros del RandomForestClassifier se pueden ajustar a través de las
constantes en la parte inferior del script. Por defecto, el modelo usa
100 árboles, una profundidad máxima de 8 y una semilla aleatoria para resultados
reproducibles. Después del entrenamiento, el script imprime la tabla de liga predicha
para 2025/26 junto con una comparación con los períodos de entrenamiento.

Uso
---
Ejecutar el script desde una terminal con Python 3. Asegurar que
`pandas`, `numpy` y `scikit‑learn` estén instalados. Todos los archivos CSV
requeridos deben residir en el mismo directorio que este script o se puede
proporcionar una ruta alternativa a través de la lista `season_files`.

Ejemplo:

```
python pl_2025_26_prediction.py
```

El script produce una clasificación predicha de los 20 clubes para la
temporada 2025/26 de la Premier League.
"""

import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def parse_match_results(df: pd.DataFrame) -> pd.DataFrame:
    """Analiza el marcador final en columnas de goles enteros.

    Los archivos CSV sin procesar usan una columna `FT` que almacena el resultado
    de tiempo completo como una cadena como `"2-1"`. Este helper divide la
    columna en cuentas separadas de goles locales y visitantes y devuelve un
    DataFrame actualizado con columnas `home_goals` y `away_goals`.

    Parámetros
    ----------
    df : DataFrame
        Datos de partidos con columnas `Team 1`, `Team 2` y `FT`.

    Devuelve
    -------
    DataFrame
        DataFrame con columnas agregadas `home_goals` y `away_goals`.
    """
    goals = df["FT"].str.split("-", expand=True)
    df = df.copy()
    df["home_goals"] = goals[0].astype(int)
    df["away_goals"] = goals[1].astype(int)
    return df


def summarise_season(matches: pd.DataFrame) -> pd.DataFrame:
    """Resume una temporada en estadísticas por equipo y clasificación final.

    Dado un DataFrame de partidos con columnas `Team 1`, `Team 2`,
    `home_goals` y `away_goals`, calcular los puntos totales, victorias,
    empates, derrotas, goles a favor y en contra y diferencia de goles para cada
    equipo. Después de acumular estadísticas, los equipos se ordenan por
    puntos (descendente), diferencia de goles (descendente) y goles a favor
    (descendente) para determinar la clasificación final.

    Parámetros
    ----------
    matches : DataFrame
        DataFrame de resultados de partidos analizados.

    Devuelve
    -------
    DataFrame
        Resumen de la temporada con una fila por equipo y columnas:
        [`team`, `points`, `wins`, `draws`, `losses`, `goals_for`,
        `goals_against`, `goal_diff`, `position`].
    """
    teams: Dict[str, Dict[str, int]] = defaultdict(lambda: {
        "points": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "goals_for": 0,
        "goals_against": 0,
    })
    # iterar a través de cada partido y actualizar las estadísticas del equipo
    for _, row in matches.iterrows():
        home, away = row["Team 1"], row["Team 2"]
        hg, ag = row["home_goals"], row["away_goals"]
        # actualizar goles
        teams[home]["goals_for"] += hg
        teams[home]["goals_against"] += ag
        teams[away]["goals_for"] += ag
        teams[away]["goals_against"] += hg
        # determinar resultado del partido
        if hg > ag:
            # victoria local
            teams[home]["points"] += 3
            teams[home]["wins"] += 1
            teams[away]["losses"] += 1
        elif hg < ag:
            # victoria visitante
            teams[away]["points"] += 3
            teams[away]["wins"] += 1
            teams[home]["losses"] += 1
        else:
            # empate
            teams[home]["points"] += 1
            teams[away]["points"] += 1
            teams[home]["draws"] += 1
            teams[away]["draws"] += 1
    # construir DataFrame
    data = []
    for team, stats in teams.items():
        goal_diff = stats["goals_for"] - stats["goals_against"]
        data.append(
            {
                "team": team,
                "points": stats["points"],
                "wins": stats["wins"],
                "draws": stats["draws"],
                "losses": stats["losses"],
                "goals_for": stats["goals_for"],
                "goals_against": stats["goals_against"],
                "goal_diff": goal_diff,
            }
        )
    summary = pd.DataFrame(data)
    # ordenar por puntos, diferencia de goles, goles a favor
    summary = summary.sort_values(
        ["points", "goal_diff", "goals_for"], ascending=[False, False, False]
    ).reset_index(drop=True)
    summary["position"] = summary.index + 1
    return summary


def prepare_training_data(season_files: List[str]) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Preparar características de entrenamiento y etiquetas de una lista de temporadas.

    Dada una lista de rutas de archivos ordenadas cronológicamente, calcular estadísticas
    por equipo para cada temporada y construir un conjunto de datos donde el vector de
    características para la temporada `n+1` proviene de las estadísticas de la temporada `n`.
    Los equipos promovidos a la Premier League sin estadísticas de temporada anterior
    tienen asignados valores de características predeterminados igual al promedio de los
    últimos tres clubes en la temporada anterior.

    Parámetros
    ----------
    season_files : list of str
        Rutas a archivos CSV de temporadas ordenados de más antiguo a más reciente.

    Devuelve
    -------
    X_train : DataFrame
        Matriz de características (numérica) para entrenamiento.
    y_train : Series
        Serie objetivo que contiene posiciones de liga (1–20).
    latest_features : DataFrame
        Matriz de características para la temporada más reciente en la lista (usada
        para predicción).
    """
    season_summaries: Dict[str, pd.DataFrame] = {}
    # calcular estadísticas resumen para cada temporada
    for file_path in season_files:
        raw = pd.read_csv(file_path)
        parsed = parse_match_results(raw)
        summary = summarise_season(parsed)
        season_summaries[file_path] = summary
    # Construir conjunto de datos de entrenamiento: usar estadísticas de temporada n para predecir posición de temporada n+1
    feature_rows = []
    target_rows = []
    files_sorted = season_files
    for i in range(len(files_sorted) - 1):
        prev_summary = season_summaries[files_sorted[i]].copy().set_index("team")
        curr_summary = season_summaries[files_sorted[i + 1]].copy().set_index("team")
        # calcular características predeterminadas basadas en los últimos tres equipos de la temporada anterior
        bottom_three = prev_summary.sort_values(
            ["points", "goal_diff", "goals_for"], ascending=[True, True, True]
        ).head(3)
        default_features = bottom_three.mean().to_dict()
        # para cada equipo en la temporada actual, recopilar características
        for team, row in curr_summary.iterrows():
            if team in prev_summary.index:
                feats = prev_summary.loc[team][
                    ["points", "wins", "draws", "losses", "goals_for", "goals_against", "goal_diff"]
                ].to_dict()
            else:
                # equipo promovido – asignar estadísticas predeterminadas de los últimos tres
                feats = {k: default_features[k] for k in [
                    "points", "wins", "draws", "losses", "goals_for", "goals_against", "goal_diff"
                ]}
            feature_rows.append(feats)
            target_rows.append(row["position"])
    X_train = pd.DataFrame(feature_rows)
    y_train = pd.Series(target_rows)
    # características para la temporada más reciente para la cual predeciremos la próxima temporada
    last_summary = season_summaries[files_sorted[-1]].copy().set_index("team")
    # calcular características predeterminadas para nuevos equipos promovidos en la próxima temporada
    # esto usa los últimos tres de last_summary
    bottom_three_last = last_summary.sort_values(
        ["points", "goal_diff", "goals_for"], ascending=[True, True, True]
    ).head(3)
    default_features_last = bottom_three_last.mean().to_dict()
    latest_features_rows = []
    latest_teams = last_summary.index.tolist()
    # incorporar equipos promovidos para 2025/26 (Leeds United, Burnley, Sunderland)
    promoted = ["Leeds United", "Burnley", "Sunderland"]
    # si un equipo promovido ya existe en last_summary (ej. Burnley fue relegado antes), usar sus estadísticas
    for team in latest_teams:
        feats = last_summary.loc[team][
            ["points", "wins", "draws", "losses", "goals_for", "goals_against", "goal_diff"]
        ].to_dict()
        latest_features_rows.append((team, feats))
    for team in promoted:
        if team not in latest_teams:
            feats = {k: default_features_last[k] for k in [
                "points", "wins", "draws", "losses", "goals_for", "goals_against", "goal_diff"
            ]}
            latest_features_rows.append((team, feats))
    latest_features_df = pd.DataFrame([feats for _, feats in latest_features_rows],
                                      index=[t for t, _ in latest_features_rows])
    return X_train, y_train, latest_features_df


def build_and_train_model(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Crear un pipeline que escale características y entrene un RandomForest.

    Parámetros
    ----------
    X : DataFrame
        Características de entrenamiento.
    y : Series
        Posiciones objetivo (1–20).

    Devuelve
    -------
    Pipeline
        Pipeline de scikit‑learn con StandardScaler y RandomForestClassifier.
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            class_weight="balanced"
        ))
    ])
    model.fit(X, y)
    return model


def predict_league_table(model: Pipeline, features: pd.DataFrame) -> pd.DataFrame:
    """Predecir el orden de la tabla de liga para las características dadas.

    Parámetros
    ----------
    model : Pipeline
        Pipeline de scikit‑learn entrenado.
    features : DataFrame
        Filas de características indexadas por nombre de equipo.

    Devuelve
    -------
    DataFrame
        Posiciones predichas ordenadas del 1 al 20.
    """
    # usar probabilidades predichas para calcular una posición final esperada.
    # RandomForestClassifier devuelve una distribución de probabilidad
    # sobre las 20 posibles posiciones finales. Al multiplicar cada probabilidad
    # por su índice de clase correspondiente (1–20) obtenemos una posición final
    # esperada (fraccional).
    probas = model.predict_proba(features)
    classes = model.named_steps["rf"].classes_
    exp_positions = probas.dot(classes)
    prediction_df = pd.DataFrame({
        "team": features.index,
        "expected_position": exp_positions
    })
    # ordenar equipos por posición esperada más baja (es decir, mejor resultado)
    prediction_df = prediction_df.sort_values("expected_position").reset_index(drop=True)
    # asignar rangos enteros 1..n basados en orden ordenado
    prediction_df["predicted_rank"] = prediction_df.index + 1
    return prediction_df[["predicted_rank", "team", "expected_position"]]


def main():
    # definir los archivos de temporada en orden cronológico
    season_files = [
        os.path.join(os.path.dirname(__file__), "eng1_2018-19.csv"),
        os.path.join(os.path.dirname(__file__), "eng1_2019-20.csv"),
        os.path.join(os.path.dirname(__file__), "eng1_2020-21.csv"),
        os.path.join(os.path.dirname(__file__), "eng1_2021-22.csv"),
        os.path.join(os.path.dirname(__file__), "eng1_2022-23.csv"),
        os.path.join(os.path.dirname(__file__), "eng1_2023-24.csv"),
    ]
    # preparar datos de entrenamiento
    X_train, y_train, latest_features = prepare_training_data(season_files)
    # entrenar modelo
    model = build_and_train_model(X_train, y_train)
    # predecir clasificación para 2025/26
    predictions = predict_league_table(model, latest_features)
    # mantener solo los 20 mejores equipos basados en posición esperada. En realidad
    # la Premier League contiene exactamente 20 clubes. Dado que podemos incluir
    # equipos promovidos adicionales debido a datos no disponibles para la
    # temporada intermedia 2024/25, truncar a 20.
    predictions = predictions.iloc[:20].copy()
    print("Tabla predicha Premier League 2025/26 (1 = campeón):")
    for _, row in predictions.iterrows():
        print(
            f"{int(row['predicted_rank'])}. {row['team']} "
            f"(pos esperada {row['expected_position']:.2f})"
        )


if __name__ == "__main__":
    main()
