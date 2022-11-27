import pandas as pd
import numpy as np

from typing import List, Tuple, Union, Dict

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
import geocoder
import os
import random

import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import datetime
import re
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor

from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    GroupKFold,
)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Normalizer
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

import warnings

warnings.simplefilter("ignore")


class Config:
    train_path = "../input/train_v2.csv"
    test_path = "../input/test_v2.csv"
    sample_submission_path = "../input/sample_submission.csv"
    output_path = "../submission/"
    seed = 42
    n_splits = 5


random.seed(Config.seed)
os.environ["PYTHONHASHSEED"] = str(Config.seed)
np.random.seed(Config.seed)


def preprocessing(df_train: pd.DataFrame):

    # 雪は花粉０そうなので0
    df_train.loc[df_train["pollen_utsunomiya"] == -9998, "pollen_utsunomiya"] = 0
    df_train.loc[df_train["pollen_chiba"] == -9998, "pollen_chiba"] = 0
    df_train.loc[df_train["pollen_tokyo"] == -9998, "pollen_tokyo"] = 0

    # 欠損補完
    df_train = df_train.replace("欠測", np.nan)
    df_train["precipitation_tokyo"] = df_train["precipitation_tokyo"].astype(float)
    df_train["temperature_chiba"] = df_train["temperature_chiba"].astype(float)
    df_train["temperature_tokyo"] = df_train["temperature_tokyo"].astype(float)

    df_train["winddirection_chiba"] = df_train["winddirection_chiba"].fillna(0)
    df_train["winddirection_chiba"] = df_train["winddirection_chiba"].astype(int)
    df_train["winddirection_tokyo"] = df_train["winddirection_tokyo"].fillna(0)
    df_train["winddirection_tokyo"] = df_train["winddirection_tokyo"].astype(int)

    df_train["windspeed_chiba"] = df_train["windspeed_chiba"].astype(float)
    df_train["windspeed_tokyo"] = df_train["windspeed_tokyo"].astype(float)

    np.random.seed(Config.seed)
    lgb_imp = IterativeImputer(
        estimator=LGBMRegressor(random_state=Config.seed, n_estimators=1000),
        max_iter=10,
        initial_strategy="mean",
        imputation_order="ascending",
        verbose=-1,
        random_state=Config.seed,
    )

    df_train = pd.DataFrame(lgb_imp.fit_transform(df_train), columns=df_train.columns)

    # 型がfloatになってしまっているので、もどす。
    df_train[
        ["winddirection_chiba", "winddirection_tokyo", "winddirection_utsunomiya"]
    ] = (
        df_train[
            ["winddirection_chiba", "winddirection_tokyo", "winddirection_utsunomiya"]
        ]
        .round()
        .astype(int)
    )
    df_train[
        [
            "precipitation_tokyo",
            "temperature_chiba",
            "temperature_tokyo",
            "windspeed_chiba",
            "windspeed_tokyo",
        ]
    ] = df_train[
        [
            "precipitation_tokyo",
            "temperature_chiba",
            "temperature_tokyo",
            "windspeed_chiba",
            "windspeed_tokyo",
        ]
    ].astype(
        float
    )
    df_train["datetime"] = df_train["datetime"].astype(int)

    return df_train


def train_lightgbm_with_cv_log(
    _df: pd.DataFrame,  # 学習データ
    df_test: pd.DataFrame,  # テストデータ
    target_label: str,  # target label
    label_cols: List[str] = [
        "pollen_utsunomiya",
        "pollen_chiba",
        "pollen_tokyo",
    ],
    unused_label: List[str] = [
        "datetime",
        "datetime_dt",
        "year",
    ],
) -> Union[np.array, List[np.float], pd.DataFrame]:

    # print(f"========={target_label}==========")

    _df = _df.copy()
    df_test = df_test.copy()

    cols = [col for col in _df.columns if col not in label_cols + unused_label]

    folds = KFold(n_splits=Config.n_splits, random_state=Config.seed)
    scores = []
    prediction = np.zeros(len(df_test))
    imps_list = []

    _val_scores = np.zeros(len(_df))

    for fold, (train_idx, val_idx) in enumerate(folds.split(_df)):
        np.random.seed(Config.seed)

        df_train = _df.iloc[train_idx].reset_index(drop=True)
        df_val = _df.iloc[val_idx].reset_index(drop=True)

        model = LGBMRegressor(random_state=Config.seed, n_estimators=1000)

        label = target_label

        model.fit(
            df_train[cols],
            # np.log1p(df_train[label]),
            np.log1p(df_train[label] / 4),
            # eval_set=(df_val[cols], np.log1p(df_val[label])),
            eval_set=(df_val[cols], np.log1p(df_val[label] / 4)),
            eval_metric="fair",
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=1000, verbose=False
                ),  # early_stopping用コールバック関数
                lgb.log_evaluation(0),
            ],  # コマンドライン出力用コールバック関数
        )

        # validation
        val_pred = model.predict(df_val[cols], num_iteration=model.best_iteration_)
        val_pred = np.expm1(val_pred).round() * 4
        val_score = mean_absolute_error(df_val[label], val_pred)
        scores.append(val_score)

        _val_scores[val_idx] = val_pred

        _pred = model.predict(df_test[cols], num_iteration=model.best_iteration_)
        _pred = np.expm1(_pred).round() * 4
        prediction += _pred / Config.n_splits
        prediction = np.where(prediction < 0, 0, prediction)

        imps = model.feature_importances_
        imps_list.append(imps)

    imps = np.mean(imps_list, axis=0)
    df_imps = pd.DataFrame({"columns": _df[cols].columns.tolist(), "feat_imp": imps})
    df_imps = df_imps.sort_values("feat_imp", ascending=False).reset_index(drop=True)

    return prediction, scores, df_imps, _val_scores


# 時間
def to_datetime(time: datetime.datetime) -> datetime.datetime:
    time = str(time)
    try:
        time = datetime.datetime.strptime(time, "%Y%m%d%H")
    except ValueError:
        time = re.sub("24$", "23", time)
        time = datetime.datetime.strptime(time, "%Y%m%d%H")
        time += datetime.timedelta(hours=1)
    return time


def time_feat(df: pd.DataFrame) -> pd.DataFrame:
    df["datetime_dt"] = df["datetime"].apply(to_datetime)
    df["year"] = df["datetime_dt"].dt.year
    df["month"] = df["datetime_dt"].dt.month
    df["day"] = df["datetime_dt"].dt.day
    df["hour"] = df["datetime_dt"].dt.hour
    df["hour_sin"] = np.sin(df["datetime_dt"].dt.hour * (2 * np.pi / (24 + 1)))
    df["hour_cos"] = np.cos(df["datetime_dt"].dt.hour * (2 * np.pi / (24 + 1)))
    df["weekday"] = df["datetime_dt"].dt.weekday
    df["day_of_year"] = df["datetime_dt"].dt.dayofyear

    return df


def add_precipitation_zero_count_feat(df: pd.DataFrame) -> pd.DataFrame:
    # https://comp.probspace.com/competitions/pollen_counts/discussions/saru_da_mon-Post5943fd8142f960c070d7

    th = 0.5
    for pref in ["utsunomiya", "chiba", "tokyo"]:
        count_list_all = []
        for year in [2017, 2018, 2019, 2020]:
            _df = df[df.year == year].reset_index(drop=True)
            count_list_by_year = []
            n_count = 0
            for e in _df[f"precipitation_{pref}"]:
                if e < th:
                    n_count += 1
                else:
                    n_count = 0
                count_list_by_year.append(np.log1p(n_count))
            count_list_all.extend(count_list_by_year)
        df[f"precipitation_{pref}_count_{th}_2"] = count_list_all

    return df


def remove_noise(
    df: pd.DataFrame,
    window_length: int,
    polyorder: int = 2,
    overwite: bool = False,
    cols=[],
) -> pd.DataFrame:
    _df = df.copy()
    for col in cols:
        np.random.seed(Config.seed)
        # _df[f"{col}_smooth_{window_length}"] = savgol_filter(_df[col])
        if overwite:
            _df[col] = savgol_filter(
                _df[col], window_length=window_length, polyorder=polyorder
            )
        else:
            _df[f"{col}_smooth_{window_length}"] = savgol_filter(
                _df[col], window_length=window_length, polyorder=polyorder
            )
            _df.loc[
                _df[f"{col}_smooth_{window_length}"] < 0,
                f"{col}_smooth_{window_length}",
            ] = 0

    return _df


# ラグ特徴/ローリング特徴量
def add_lag_feat(df: pd.DataFrame, feat: List[str], group: str) -> pd.DataFrame:
    outputs = [df]

    grp_df = df.groupby(group)  # year ごとにシフトする。 各年1~6月期間しかないのでこのようにする。

    for lag in [1, 2, 3, 4, 5]:
        # for lag in list(range(1, 25)):
        # shift
        outputs.append(grp_df[feat].shift(lag).add_prefix(f"shift_{lag}_"))
        # diff
        outputs.append(grp_df[feat].diff(lag).add_prefix(f"diff_{lag}_"))

    # rolling
    windows = [3] + [i * 24 for i in range(1, 3)]
    # windows = [3] + [i * 24 for i in range(1, 10)]
    for window in windows:
        tmp_df = grp_df[feat].rolling(window, min_periods=1)
        tmp_df = tmp_df.mean().add_prefix(f"rolling_{window}_mean_")
        outputs.append(tmp_df.reset_index(drop=True))

    _df = pd.concat(outputs, axis=1)
    return _df


def temperature_decompose(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    from statsmodels.tsa.seasonal import STL

    _df = df.copy()
    np.random.seed(Config.seed)
    # for col in ["temperature_tokyo", "temperature_utsunomiya", "temperature_chiba"]:
    for col in cols:
        np.random.seed(Config.seed)
        res = STL(_df[col], period=24).fit()
        _df[f"{col}_decompose_trend"] = res.trend
        _df[f"{col}_decompose_seasonal"] = res.seasonal
        _df[f"{col}_decompose_resid"] = res.resid
        _df[f"{col}_decompose_minus_trend"] = _df[col] - _df[f"{col}_decompose_trend"]

    return _df


def add_sekisan_ondo2(
    df: pd.DataFrame,
    cols: List[str] = [
        "temperature_utsunomiya",
        "temperature_tokyo",
        "temperature_chiba",
    ],
) -> pd.DataFrame:
    # https://hp.otenki.com/401/
    # 1/1からの毎日の温度の平均を和を算出していく

    dfs = []
    for year in [2017, 2018, 2019, 2020]:
        _df = df[df.year == year].reset_index(drop=True).copy()
        # _df["ymd"] = _df["datetime_dt"].apply(lambda x: x.strftime("%Y%m%d"))

        for col in cols:
            _df.loc[_df[col] < 0, col] = 0
            s = _df[col].sum()
            _df[f"sekisan_ondo_{col}_2"] = _df[col].cumsum()
            # _df[f"sekisan_ondo_{col}_2_ratio"] = _df[f"sekisan_ondo_{col}_2"] / s
        dfs.append(_df)

    _df = pd.concat(dfs)

    return _df


def add_wind_direction_to_cos_sin(
    df: pd.DataFrame,
    cols: List[str] = [
        "winddirection_utsunomiya",
        "winddirection_tokyo",
        "winddirection_chiba",
    ],
) -> pd.DataFrame:
    """
    {col} が 0：静穏 の場合 {col}_cos, {col}_sinには、欠損になる
    1：北北東
    2：北東
    3：東北東
    4：東
    5：東南東
    6：南東
    7：南南東
    8：南
    9：南南西
    10：南西
    11：西南西
    12：西
    13：西北西
    14：北西
    15：北北西
    16：北

    """
    df_origin = df.copy()

    direction_map = {
        1: 13,
        2: 14,
        3: 15,
        4: 0,
        5: 1,
        6: 2,
        7: 3,
        8: 4,
        9: 5,
        10: 6,
        11: 7,
        12: 8,
        13: 9,
        14: 10,
        15: 11,
        16: 12,
    }

    pref_list = ["utsunomiya", "tokyo", "chiba"]

    for col in cols:

        _df = df_origin[[col]].copy()
        _df = _df.reset_index()
        _df = _df[_df[col] != 0].reset_index(drop=True)  # 0：静穏 は風の方向ではないので消す
        # _df[col] = _df[col] - 1  # 1 ~ 16 なので 0 ~ 15に変換
        _df[col] = _df[col].apply(lambda x: direction_map[x])

        _df[f"{col}_cos"] = np.cos(2 * np.pi * _df[col] / (_df[col].max() + 1))
        _df[f"{col}_sin"] = np.sin(2 * np.pi * _df[col] / (_df[col].max() + 1))
        _df = _df.drop(col, axis=1)

        df_origin = df_origin.reset_index()
        df_origin = df_origin.merge(_df, on="index", how="left")
        df_origin = df_origin.drop("index", axis=1)

        pref = col.split("_")[-1]
        df_origin[f"{col}_cos_mult_window_speed"] = (
            df_origin[f"windspeed_{pref}"] * df_origin[f"{col}_cos"]
        )  # .fillna(0)
        df_origin[f"{col}_sin_mult_window_speed"] = (
            df_origin[f"windspeed_{pref}"] * df_origin[f"{col}_sin"]
        )

    return df_origin


def add_wind_direction_one_hot(
    df: pd.DataFrame,
    cols=[
        "winddirection_utsunomiya",
        "winddirection_tokyo",
        "winddirection_chiba",
    ],
) -> pd.DataFrame:

    num_wind_direction = 17

    dfs = []
    for col in cols:
        data = {f"{col}_one_hot_{i}": [] for i in range(num_wind_direction)}
        for x in df[col].astype(int):
            for i in range(num_wind_direction):
                if x == i:
                    data[f"{col}_one_hot_{i}"].append(1)
                else:
                    data[f"{col}_one_hot_{i}"].append(0)

        _df = pd.DataFrame(data)
        dfs.append(_df)

    return pd.concat([df] + dfs, axis=1)


def make_feature_ut(df: pd.DataFrame) -> pd.DataFrame:
    _df = df.copy()
    _df = time_feat(_df)

    # 降水量カウント
    _df = add_precipitation_zero_count_feat(_df)
    # _df = remove_noise(
    #     _df,
    #     window_length=7,
    #     overwite=False,
    #     cols=[
    #         "precipitation_utsunomiya",
    #         "temperature_utsunomiya",
    #         "windspeed_utsunomiya",
    #     ],
    # )

    _df = add_lag_feat(
        _df,
        ["precipitation_utsunomiya", "temperature_utsunomiya"],
        "year",
    )

    _df = temperature_decompose(_df, ["temperature_utsunomiya"])

    # _df = add_wind_direction_to_cos_sin(
    #     _df,
    #     [
    #         # "winddirection_utsunomiya",
    #         # "winddirection_tokyo",
    #         # "winddirection_chiba"
    #     ],
    # )

    # _df = add_wind_direction_one_hot(_df, ["winddirection_utsunomiya"])

    # _df = add_sekisan_ondo2(_df, cols=["temperature_utsunomiya"])

    _df = _df.drop(
        [
            "winddirection_utsunomiya",
            "winddirection_tokyo",
            "winddirection_chiba",
            "temperature_tokyo",
            # "windspeed_tokyo",
            # "precipitation_tokyo",
            # "windspeed_chiba",
            # "temperature_chiba",
            # "precipitation_chiba",
        ],
        axis=1,
    )

    return _df


def make_feature_cb(df: pd.DataFrame) -> pd.DataFrame:
    _df = df.copy()
    _df = time_feat(_df)

    # 降水量カウント
    _df = add_precipitation_zero_count_feat(_df)

    _df = remove_noise(
        _df,
        window_length=7,
        overwite=False,
        cols=[
            "precipitation_chiba",
            "temperature_chiba",
            "windspeed_chiba",
            "precipitation_tokyo",
            "temperature_tokyo",
            "windspeed_tokyo",
        ],
    )

    # _df = add_lag_feat(
    #     _df,
    #     [
    #         # "precipitation_chiba",
    #         "temperature_chiba"
    #     ],
    #     "year",
    # )

    # _df = temperature_decompose(_df, ["temperature_tokyo"])

    # _df = add_wind_direction_to_cos_sin(
    #     _df,
    #     [
    #         "winddirection_utsunomiya",
    #         "winddirection_tokyo",
    #         "winddirection_chiba"
    #     ],
    # )

    # _df = add_wind_direction_one_hot(_df, ["winddirection_chiba"])

    # _df = add_sekisan_ondo2(_df, cols=["temperature_chiba"])

    _df = _df.drop(
        [
            "winddirection_utsunomiya",
            "winddirection_tokyo",
            "winddirection_chiba",
            # "temperature_utsunomiya",
            # "temperature_tokyo",
            # "precipitation_utsunomiya",
            # "windspeed_utsunomiya",
        ],
        axis=1,
    )

    return _df


def make_feature_tk(df: pd.DataFrame) -> pd.DataFrame:
    _df = df.copy()
    _df = time_feat(_df)

    # 降水量カウント
    _df = add_precipitation_zero_count_feat(_df)
    _df = remove_noise(
        _df,
        window_length=7,
        overwite=False,
        cols=[
            "precipitation_tokyo",
            "temperature_tokyo",
            "windspeed_tokyo",
            "precipitation_utsunomiya",
            # "temperature_utsunomiya",
            # "windspeed_utsunomiya",
            # "precipitation_chiba",
            # "temperature_chiba",
            # "windspeed_chiba",
        ],
    )

    _df = add_lag_feat(
        _df,
        [
            # "precipitation_tokyo",
            "temperature_tokyo",
            "windspeed_tokyo",
            "precipitation_utsunomiya",
            # "temperature_utsunomiya",
            "precipitation_chiba",
            # "temperature_chiba",
            # "windspeed_chiba",
            # "windspeed_utsunomiya",
        ],
        "year",
    )

    # _df = temperature_decompose(_df, ["temperature_tokyo"])

    # _df = add_wind_direction_to_cos_sin(
    #     _df,
    #     [
    #         "winddirection_utsunomiya",
    #         # "winddirection_tokyo",
    #         # "winddirection_chiba"
    #     ],
    # )

    _df = add_sekisan_ondo2(_df, cols=["temperature_tokyo"])

    _df = _df.drop(
        [
            "winddirection_utsunomiya",
            "winddirection_tokyo",
            "winddirection_chiba",
        ],
        axis=1,
    )

    return _df


def run_train(_df, _df_test, q, label, unused_feat=[]):

    _df_tr = _df[_df[label] >= 0].reset_index(drop=True)
    _df_tr = _df_tr[_df_tr[label] <= q].reset_index(drop=True)

    _df_tr = _df_tr[
        (_df_tr["month"] == 4) | (_df_tr["month"] == 5) | (_df_tr["month"] == 6)
    ].reset_index(drop=True)

    prediction, scores, df_imps, _val_scores = train_lightgbm_with_cv_log(
        _df_tr,
        _df_test,
        target_label=label,
        unused_label=["datetime", "datetime_dt"] + unused_feat,
    )

    # for i, score in enumerate(scores):
    #    print(f" fold_{i} mae: {score}")
    # print(f"mean: {np.mean(scores)}")

    return prediction, scores, df_imps, _val_scores


def make_feature(df_train, df_test, make_feature_func):
    df_test["pollen_utsunomiya"] = -1
    df_test["pollen_chiba"] = -1
    df_test["pollen_tokyo"] = -1
    _df_tmp = pd.concat([df_train, df_test]).reset_index(drop=True)

    _df_feat = make_feature_func(_df_tmp)

    _df = _df_feat[_df_feat["pollen_utsunomiya"] != -1].reset_index(drop=True)
    _df_test = _df_feat[_df_feat["pollen_utsunomiya"] == -1].reset_index(drop=True)
    _df_test = _df_test.drop(
        ["pollen_utsunomiya", "pollen_chiba", "pollen_tokyo"], axis=1
    )

    return _df, _df_test


def main():
    print(f"read_data")
    df_train = pd.read_csv(Config.train_path)
    df_test = pd.read_csv(Config.test_path)

    q_ut, q_tk, q_cb = 20, 20, 20

    print("make feat")
    df_train = preprocessing(df_train)
    _df_ut, _df_test_ut = make_feature(df_train, df_test, make_feature_ut)
    _df_tk, _df_test_tk = make_feature(df_train, df_test, make_feature_tk)
    _df_cb, _df_test_cb = make_feature(df_train, df_test, make_feature_cb)

    print(f"train")
    prediction_ut, scores_ut, df_imps_ut, _val_scores_ut = run_train(
        _df_ut,
        _df_test_ut,
        q_ut,
        "pollen_utsunomiya",
    )

    # print(df_imps_ut)

    prediction_tk, scores_tk, df_imps_tk, _val_scores_tk = run_train(
        _df_tk, _df_test_tk, q_tk, "pollen_tokyo"
    )

    prediction_cb, scores_cb, df_imps_cb, _val_scores_cb = run_train(
        _df_cb, _df_test_cb, q_cb, "pollen_chiba"
    )

    print("=============score")
    print(
        f"ut = {np.mean(scores_ut)}",
        f"tk = {np.mean(scores_tk)}",
        f"cb = {np.mean(scores_cb)}",
        f"mean = {np.mean([np.mean(scores_ut), np.mean(scores_tk), np.mean(scores_cb)])}",
    )
    print("=============score")

    # submission
    df_sub = _df_test_ut[["datetime"]]
    df_sub.loc[:, "pollen_utsunomiya"] = prediction_ut
    df_sub.loc[:, "pollen_chiba"] = prediction_cb
    df_sub.loc[:, "pollen_tokyo"] = prediction_tk
    df_sub.to_csv(f"sub.csv", index=None)


if __name__ == "__main__":
    main()
