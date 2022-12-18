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
from dateutil import tz
import re
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor

from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold, TimeSeriesSplit

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Normalizer, StandardScaler
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

import warnings
import seaborn as sns

from feature import (
    make_feature_tk,
    make_feature_tk_2,
    make_feature_ut,
    make_feature_ut_2,
    make_feature_cb,
    make_feature_cb_2,
)

warnings.simplefilter("ignore")


def preprocessing(df_train: pd.DataFrame, args):

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

    np.random.seed(args.seed)
    lgb_imp = IterativeImputer(
        estimator=LGBMRegressor(random_state=args.seed, n_estimators=1000),
        max_iter=10,
        initial_strategy="mean",
        imputation_order="ascending",
        verbose=-1,
        random_state=args.seed,
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


def getNearestValue(list, num):
    """
    https://qiita.com/icchi_h/items/fc0df3abb02b51f81657
    概要: リストからある値に最も近い値を返却する関数
    @param list: データ配列
    @param num: 対象値
    @return 対象値に最も近い値
    """

    # リスト要素と対象値の差分を計算し最小値のインデックスを取得
    idx = np.abs(np.asarray(list) - num).argmin()
    return list[idx]


def train_lightgbm_with_cv_log(
    _df: pd.DataFrame,  # 学習データ
    df_test: pd.DataFrame,  # テストデータ
    target_label: str,  # target label
    args,
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
    val_test_split: float = 0.8,
    q: int = -1,
    plot: bool = False,
    convert_pollen: bool = True,
) -> Union[np.array, List[np.float], pd.DataFrame, np.array]:

    # print(f"========={target_label}==========")

    _df = _df.copy()
    df_test = df_test.copy()
    label = target_label
    cols = [col for col in _df.columns if col not in label_cols + unused_label]

    folds = KFold(n_splits=args.n_splits, random_state=args.seed, shuffle=False)
    # folds = TimeSeriesSplit(n_splits=Config.n_splits)
    scores = []
    prediction = np.zeros(len(df_test))
    imps_list = []
    v_test_scores = []

    _val_scores = np.zeros(len(_df))
    pollen_list = list(set(_df[label].tolist()))

    for fold, (train_idx, val_idx) in enumerate(folds.split(_df)):
        print(train_idx)
        np.random.seed(args.seed)

        df_train = _df.iloc[train_idx].reset_index(drop=True)
        df_val = _df.iloc[val_idx].reset_index(drop=True)

        df_v_val = df_val.iloc[: int(len(df_val) * val_test_split)].reset_index(
            drop=True
        )
        df_v_test = df_val.iloc[int(len(df_val) * val_test_split) :].reset_index(
            drop=True
        )

        if q > 0:
            df_train = df_train[df_train[label] <= q].reset_index(drop=True)
            df_v_val = df_v_val[df_v_val[label] <= q].reset_index(drop=True)

        model = LGBMRegressor(random_state=args.seed, n_estimators=1000)

        model.fit(
            df_train[cols],
            np.log1p(df_train[label]),
            # np.log1p(df_train[label] / 4),
            # eval_set=(df_val[cols], np.log1p(df_val[label])),
            # eval_set=(df_val[cols], np.log1p(df_val[label] / 4)),
            eval_set=(df_v_val[cols], np.log1p(df_v_val[label])),
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
        # val_pred = np.expm1(val_pred).round() * 4
        # val_pred = np.expm1(val_pred).round()
        val_pred = np.expm1(val_pred)
        val_pred = np.array([getNearestValue(pollen_list, v) for v in val_pred])

        val_score = mean_absolute_error(df_val[label], val_pred)
        scores.append(val_score)

        _val_scores[val_idx] = val_pred

        # test data (not used in train)
        v_test_pred = model.predict(
            df_v_test[cols], num_iteration=model.best_iteration_
        )
        # v_test_pred = np.expm1(v_test_pred).round() * 4
        # v_test_pred = np.expm1(v_test_pred).round()
        v_test_pred = np.expm1(v_test_pred)
        v_test_pred = np.array([getNearestValue(pollen_list, v) for v in v_test_pred])
        v_test_score = mean_absolute_error(df_v_test[label], v_test_pred)
        v_test_scores.append(v_test_score)

        _pred = model.predict(df_test[cols], num_iteration=model.best_iteration_)
        # _pred = np.expm1(_pred).round() * 4
        _pred = np.expm1(_pred)
        prediction += _pred / args.n_splits
        # prediction = prediction.round()

        imps = model.feature_importances_
        imps_list.append(imps)

        if plot:
            _df_v_test = df_v_test.copy()
            _df_v_test["pred"] = v_test_pred
            plt.figure(figsize=(20, 6))
            ax = sns.lineplot(data=_df_v_test, x="datetime_dt", y="pred", label="pred")
            ax = sns.lineplot(data=_df_v_test, x="datetime_dt", y=label, label=label)

    prediction = np.where(prediction < 0, 0, prediction)
    # prediction = np.array([getNearestValue(pollen_list, v) for v in prediction])
    if convert_pollen:
        prediction = np.array([getNearestValue(pollen_list, v) for v in prediction])

    imps = np.mean(imps_list, axis=0)
    df_imps = pd.DataFrame({"columns": _df[cols].columns.tolist(), "feat_imp": imps})
    df_imps = df_imps.sort_values("feat_imp", ascending=False).reset_index(drop=True)

    return prediction, scores, df_imps, _val_scores, v_test_scores


def train_skmodel_with_cv_log(
    _df: pd.DataFrame,  # 学習データ
    df_test: pd.DataFrame,  # テストデータ
    target_label: str,  # target label
    args,
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
    val_test_split: float = 0.8,
    q: int = -1,
    plot: bool = False,
) -> Union[np.array, List[np.float], pd.DataFrame, np.array]:

    # print(f"========={target_label}==========")

    _df = _df.copy()
    df_test = df_test.copy()
    label = target_label
    cols = [col for col in _df.columns if col not in label_cols + unused_label]

    folds = KFold(n_splits=args.n_splits, random_state=args.seed, shuffle=False)
    # folds = TimeSeriesSplit(n_splits=Config.n_splits)
    scores = []
    prediction = np.zeros(len(df_test))
    imps_list = []
    v_test_scores = []

    _val_scores = np.zeros(len(_df))
    pollen_list = list(set(_df[label].tolist()))

    for fold, (train_idx, val_idx) in enumerate(folds.split(_df)):
        print(train_idx)
        np.random.seed(args.seed)

        df_train = _df.iloc[train_idx].reset_index(drop=True)
        df_val = _df.iloc[val_idx].reset_index(drop=True)

        df_v_val = df_val.iloc[: int(len(df_val) * val_test_split)].reset_index(
            drop=True
        )
        df_v_test = df_val.iloc[int(len(df_val) * val_test_split) :].reset_index(
            drop=True
        )

        if q > 0:
            df_train = df_train[df_train[label] <= q].reset_index(drop=True)
            df_v_val = df_v_val[df_v_val[label] <= q].reset_index(drop=True)

        # model = LGBMRegressor(random_state=Config.seed, n_estimators=1000)

        if args.model == "svr":
            normalizer = StandardScaler().fit(df_train[cols])
            model = SVR()

        elif args.model == "mlp":
            normalizer = Normalizer().fit(df_train[cols])

            model = MLPRegressor(
                hidden_layer_sizes=(300, 200),
                random_state=args.seed,
                activation="relu",
                solver="adam",
                learning_rate="adaptive",
            )

        model.fit(normalizer.transform(df_train[cols]), np.log1p(df_train[label]))

        # validation
        val_pred = model.predict(normalizer.transform(df_val[cols]))
        val_pred = np.expm1(val_pred)
        val_pred = np.array([getNearestValue(pollen_list, v) for v in val_pred])

        val_score = mean_absolute_error(df_val[label], val_pred)
        scores.append(val_score)

        _val_scores[val_idx] = val_pred

        # test data (not used in train)
        v_test_pred = model.predict(normalizer.transform(df_v_test[cols]))

        v_test_pred = np.expm1(v_test_pred)
        v_test_pred = np.array([getNearestValue(pollen_list, v) for v in v_test_pred])
        v_test_score = mean_absolute_error(df_v_test[label], v_test_pred)
        v_test_scores.append(v_test_score)

        _pred = model.predict(normalizer.transform(df_test[cols]))
        # _pred = np.expm1(_pred).round() * 4
        _pred = np.expm1(_pred)
        prediction += _pred / args.n_splits
        # prediction = prediction.round()

        if plot:
            _df_v_test = df_v_test.copy()
            _df_v_test["pred"] = v_test_pred
            plt.figure(figsize=(20, 6))
            ax = sns.lineplot(data=_df_v_test, x="datetime_dt", y="pred", label="pred")
            ax = sns.lineplot(data=_df_v_test, x="datetime_dt", y=label, label=label)

    prediction = np.where(prediction < 0, 0, prediction)
    prediction = np.array([getNearestValue(pollen_list, v) for v in prediction])

    return prediction, scores, None, _val_scores, v_test_scores


def convert_pollen_total_2020(
    df: pd.DataFrame, target_cols: List[str], total_pollen_2020: List[float]
):
    dfs = []
    for year in [2017, 2018, 2019, 2020]:

        _df = df[df.year == year].reset_index(drop=True)
        for i, col in enumerate(target_cols):
            pollen_list = df[col].unique().tolist()
            if year == 2020:
                continue
            # _df.loc[_df[col] < 0 , col] = 0
            _df = _df[_df[col] >= 0].reset_index(drop=True)
            _df[col] = _df[col] * (total_pollen_2020[i] / _df[col].sum())
            _df[col] = _df[col].apply(lambda x: getNearestValue(pollen_list, x))

        dfs.append(_df)

    df = pd.concat(dfs).reset_index(drop=True)
    return df


def run_train(
    _df,
    _df_test,
    q,
    label,
    args,
    unused_feat=[],
    label_cols: List[str] = [
        "pollen_utsunomiya",
        "pollen_chiba",
        "pollen_tokyo",
    ],
    convert_pollen: bool = True,
):

    _df_tr = _df[_df[label] >= 0].reset_index(drop=True)
    _df_tr = _df_tr.dropna(how="any").reset_index(drop=True)

    if args.model == "lgbm":

        (
            prediction,
            scores,
            df_imps,
            _val_scores,
            v_test_scores,
        ) = train_lightgbm_with_cv_log(
            _df_tr,
            _df_test,
            target_label=label,
            args=args,
            unused_label=["datetime", "datetime_dt"] + unused_feat,
            label_cols=label_cols,
            q=q,
            convert_pollen=convert_pollen,
        )

    elif args.model == "svr" or args.model == "mlp":
        (
            prediction,
            scores,
            df_imps,
            _val_scores,
            v_test_scores,
        ) = train_skmodel_with_cv_log(
            _df_tr,
            _df_test,
            target_label=label,
            args=args,
            unused_label=["datetime", "datetime_dt"] + unused_feat,
            label_cols=label_cols,
            q=q,
        )

    return prediction, scores, df_imps, _val_scores, v_test_scores


def make_feature(
    df_train,
    df_test,
    make_feature_func,
    args,
    cols=["pollen_utsunomiya", "pollen_chiba", "pollen_tokyo"],
):
    for col in cols:
        df_test[col] = -1

    _df_tmp = pd.concat([df_train, df_test]).reset_index(drop=True)

    _df_feat = make_feature_func(_df_tmp, args)

    _df = _df_feat[_df_feat[cols[0]] != -1].reset_index(drop=True)
    _df_test = _df_feat[_df_feat[cols[0]] == -1].reset_index(drop=True)
    _df_test = _df_test.drop(cols, axis=1)

    return _df, _df_test


# class Config:
#     train_path = "../input/train_v2.csv"
#     test_path = "../input/test_v2.csv"
#     sample_submission_path = "../input/sample_submission.csv"
#     output_path = "submission/"
#     seed = 42
#     n_splits = 4


def parser():
    import argparse

    parser = argparse.ArgumentParser(description="Kafun solution")
    parser.add_argument("--train_path", default="../input/train_v2.csv", type=str)
    parser.add_argument("--test_path", default="../input/test_v2.csv", type=str)
    parser.add_argument(
        "--sample_submission_path", default="../input/sample_submission.csv", type=str
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--n_splits", default=4, type=int)
    parser.add_argument("--hosei", action="store_true")
    parser.add_argument("--q50", action="store_true")
    parser.add_argument("--no_q", action="store_true")

    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--model", type=str, default="lgbm", choices=["lgbm", "svr", "mlp"]
    )

    args = parser.parse_args()

    print("==== params =====")
    for key, value in vars(args).items():
        print(f"{key}={value}")
    print("==== params =====")

    return args


def main():
    args = parser()

    # seed
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)

    # const
    total_pollen_2020_ut = 47848.946232729206
    total_pollen_2020_tk = 28232.168622513003
    total_pollen_2020_cb = 32978.780406463025

    df_train = pd.read_csv(args.train_path)
    df_test = pd.read_csv(args.test_path)

    df_train = preprocessing(df_train, args)

    _df_ut, _df_test_ut = make_feature(df_train, df_test, make_feature_ut, args)
    _df_tk, _df_test_tk = make_feature(df_train, df_test, make_feature_tk, args)
    _df_cb, _df_test_cb = make_feature(df_train, df_test, make_feature_cb, args)

    q_ut, q_tk, q_cb = 50, 30, 36

    if args.q50:
        q_ut, q_tk, q_cb = 50, 50, 50

    if args.no_q:
        q_ut, q_tk, q_cb = -1, -1, -1

    if args.hosei:
        q_ut, q_tk, q_cb = -1, -1, -1

        _df_ut = convert_pollen_total_2020(
            _df_ut, ["pollen_utsunomiya"], [total_pollen_2020_ut]
        )
        _df_tk = convert_pollen_total_2020(
            _df_tk, ["pollen_tokyo"], [total_pollen_2020_tk]
        )
        _df_cb = convert_pollen_total_2020(
            _df_cb, ["pollen_chiba"], [total_pollen_2020_cb]
        )

    # train 1
    (
        prediction_ut_1,
        scores_ut,
        df_imps_ut,
        _val_scores_ut,
        v_test_scores_ut,
    ) = run_train(
        _df_ut,
        _df_test_ut,
        q_ut,
        "pollen_utsunomiya",
        args,
    )

    (
        prediction_tk_1,
        scores_tk,
        df_imps_tk,
        _val_scores_tk,
        v_test_scores_tk,
    ) = run_train(_df_tk, _df_test_tk, q_tk, "pollen_tokyo", args)

    (
        prediction_cb_1,
        scores_cb,
        df_imps_cb,
        _val_scores_cb,
        v_test_scores_cb,
    ) = run_train(_df_cb, _df_test_cb, q_cb, "pollen_chiba", args)

    # train 2
    _df_ut_2 = _df_ut.copy()
    _df_cb_2 = _df_cb.copy()
    _df_tk_2 = _df_tk.copy()

    _df_test_ut_2 = _df_test_ut.copy()
    _df_test_cb_2 = _df_test_cb.copy()
    _df_test_tk_2 = _df_test_tk.copy()

    _df_test_ut_2["pollen_tokyo"] = prediction_tk_1
    _df_test_ut_2["pollen_chiba"] = prediction_cb_1
    _df_test_ut_2["pollen_utsunomiya_pseudo"] = prediction_ut_1

    _df_ut_2["pollen_utsunomiya_pseudo"] = _df_ut["pollen_utsunomiya"]

    _df_test_tk_2["pollen_utsunomiya"] = prediction_ut_1
    _df_test_tk_2["pollen_chiba"] = prediction_cb_1
    _df_test_tk_2["pollen_tokyo_pseudo"] = prediction_tk_1

    _df_tk_2["pollen_tokyo_pseudo"] = _df_tk["pollen_tokyo"]

    _df_test_cb_2["pollen_utsunomiya"] = prediction_ut_1
    _df_test_cb_2["pollen_tokyo"] = prediction_tk_1
    _df_test_cb_2["pollen_chiba_pseudo"] = prediction_cb_1

    _df_cb_2["pollen_chiba_pseudo"] = _df_cb["pollen_chiba"]

    if args.hosei:
        _df_ut_2 = convert_pollen_total_2020(
            _df_ut_2,
            ["pollen_tokyo", "pollen_chiba"],
            [total_pollen_2020_tk, total_pollen_2020_cb],
        )

        _df_tk_2 = convert_pollen_total_2020(
            _df_tk_2,
            ["pollen_utsunomiya", "pollen_chiba"],
            [total_pollen_2020_ut, total_pollen_2020_cb],
        )

        _df_cb_2 = convert_pollen_total_2020(
            _df_cb_2,
            ["pollen_utsunomiya", "pollen_tokyo"],
            [total_pollen_2020_ut, total_pollen_2020_tk],
        )

    # ut
    _df_ut_2, _df_test_ut_2 = make_feature(
        _df_ut_2, _df_test_ut_2, make_feature_ut_2, args, cols=["pollen_utsunomiya"]
    )
    prediction_ut, scores_ut, df_imps_ut, _val_scores_ut, v_test_scores_ut = run_train(
        _df_ut_2,
        _df_test_ut_2,
        q_ut,
        "pollen_utsunomiya",
        args,
        label_cols=["pollen_utsunomiya"],
    )

    # tk
    _df_tk_2, _df_test_tk_2 = make_feature(
        _df_tk_2, _df_test_tk_2, make_feature_tk_2, args, cols=["pollen_tokyo"]
    )

    prediction_tk, scores_tk, df_imps_tk, _val_scores_tk, v_test_scores_tk = run_train(
        _df_tk_2, _df_test_tk_2, q_tk, "pollen_tokyo", args, label_cols=["pollen_tokyo"]
    )

    # cb

    _df_cb_2, _df_test_cb_2 = make_feature(
        _df_cb_2, _df_test_cb_2, make_feature_cb_2, args, cols=["pollen_chiba"]
    )

    prediction_cb, scores_cb, df_imps_cb, _val_scores_cb, v_test_scores_cb = run_train(
        _df_cb_2, _df_test_cb_2, q_cb, "pollen_chiba", args, label_cols=["pollen_chiba"]
    )

    print("=============score")
    print(f"seed {args.seed}")
    print(
        "val  score",
        f"ut = {np.mean(scores_ut):0.4}",
        f"tk = {np.mean(scores_tk):0.4}",
        f"cb = {np.mean(scores_cb):0.4}",
        f"mean = {np.mean([np.mean(scores_ut), np.mean(scores_tk), np.mean(scores_cb)]):0.4}",
    )

    print(
        "test score",
        f"ut = {np.mean(v_test_scores_ut):0.4}",
        f"tk = {np.mean(v_test_scores_tk):0.4}",
        f"cb = {np.mean(v_test_scores_cb):0.4}",
        f"mean = {np.mean([np.mean(v_test_scores_ut), np.mean(v_test_scores_tk), np.mean(v_test_scores_cb)]):0.4}",
    )

    df_sub = _df_test_ut[["datetime"]]
    df_sub.loc[:, "pollen_utsunomiya"] = prediction_ut
    df_sub.loc[:, "pollen_chiba"] = prediction_cb
    df_sub.loc[:, "pollen_tokyo"] = prediction_tk

    output_path = "/".join(args.output.split("/")[:-1])
    output_file = args.output.split("/")[-1]

    os.makedirs(output_path, exist_ok=True)

    df_sub.to_csv(f"{output_path}/{output_file}", index=None)


if __name__ == "__main__":
    main()
