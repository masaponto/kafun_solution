import pandas as pd
import numpy as np
import os
from main import getNearestValue


class Const:
    lgbm_score = 12.22388
    lgbm_hosei_score = 12.46269
    catboost_score = 12.36318
    lgbm_q_50_score = 12.26368
    lgbm_no_q_score = 15.25373  # same as lgbm_q_50k_score
    lgbm_q_20_score = 13.41791
    svr_score = 12.44279
    mlp_score = 12.71144
    lgbm_tel_score = 12.42289
    lgbm_cv5_score = 12.30348
    lgbm_hosei_cv5_score = 12.60199
    lgbm_param_score = 12.44279


def ensemble(col, score_list, dfs, pollen_list):
    s = sum([100 - x for x in score_list])
    weight_list = [(100 - score) / s for score in score_list]

    df = pd.DataFrame()
    df["datetime"] = dfs[0]["datetime"]
    df[col] = 0

    for weight, _df in zip(weight_list, dfs):
        df[col] = df[col] + (weight * _df[col])

    df[col] = np.array([getNearestValue(pollen_list, v) for v in df[col]])

    return df


def ensemble_6():
    df_train = pd.read_csv("../input/train_v2.csv", index_col=None)
    pollen_list_ut = list(set(df_train["pollen_utsunomiya"].tolist()))
    pollen_list_tk = list(set(df_train["pollen_tokyo"].tolist()))
    pollen_list_cb = list(set(df_train["pollen_chiba"].tolist()))

    df_lgbm = pd.read_csv("submission/sub_42_4.csv", index_col=None)
    df_lgbm_hosei = pd.read_csv("submission/sub_42_4_hosei.csv", index_col=None)
    df_lgbm_q_50 = pd.read_csv("submission/sub_q-50-50-50.csv", index_col=None)
    df_lgbm_no_q = pd.read_csv(
        "submission/sub_42_4_no_q.csv", index_col=None
    )  # same as lgbm_q_50k
    df_svr = pd.read_csv("submission/sub_svr_42_4.csv", index_col=None)
    df_mlp = pd.read_csv("submission/sub_mlp_42_4.csv", index_col=None)

    # ut
    dfs_ut = [df_lgbm, df_lgbm_hosei, df_lgbm_q_50, df_svr, df_mlp]
    score_list_ut = [
        Const.lgbm_score,
        Const.lgbm_hosei_score,
        Const.lgbm_q_50_score,
        Const.svr_score,
        Const.mlp_score,
    ]

    # tk
    dfs_tk = [df_lgbm_hosei, df_lgbm_no_q]
    score_list_tk = [Const.lgbm_hosei_score, Const.lgbm_no_q_score]

    # cb
    dfs_cb = [df_lgbm_hosei, df_lgbm_no_q]
    score_list_cb = [Const.lgbm_hosei_score, Const.lgbm_no_q_score]

    df_ut = ensemble("pollen_utsunomiya", score_list_ut, dfs_ut, pollen_list_ut)
    df_tk = ensemble("pollen_tokyo", score_list_tk, dfs_tk, pollen_list_tk)
    df_cb = ensemble("pollen_chiba", score_list_cb, dfs_cb, pollen_list_cb)

    #
    df = df_ut.merge(df_tk, on="datetime")
    df = df.merge(df_cb, on="datetime")

    df = df[["datetime", "pollen_utsunomiya", "pollen_chiba", "pollen_tokyo"]]
    return df


def ensemble_5():
    df_train = pd.read_csv("../input/train_v2.csv", index_col=None)
    pollen_list_ut = list(set(df_train["pollen_utsunomiya"].tolist()))
    pollen_list_tk = list(set(df_train["pollen_tokyo"].tolist()))
    pollen_list_cb = list(set(df_train["pollen_chiba"].tolist()))

    df_lgbm = pd.read_csv("submission/sub_42_4.csv", index_col=None)
    df_lgbm_hosei = pd.read_csv("submission/sub_42_4_hosei.csv", index_col=None)
    df_lgbm_q_50 = pd.read_csv("submission/sub_q-50-50-50.csv", index_col=None)
    df_lgbm_no_q = pd.read_csv(
        "submission/sub_42_4_no_q.csv", index_col=None
    )  # same as lgbm_q_50k
    df_svr = pd.read_csv("submission/sub_svr_42_4.csv", index_col=None)

    # ut
    dfs_ut = [df_lgbm, df_lgbm_hosei, df_lgbm_q_50, df_svr]
    score_list_ut = [
        Const.lgbm_score,
        Const.lgbm_hosei_score,
        Const.lgbm_q_50_score,
        Const.svr_score,
    ]

    # tk
    dfs_tk = [df_lgbm_hosei, df_lgbm_no_q]
    score_list_tk = [Const.lgbm_hosei_score, Const.lgbm_no_q_score]

    # cb
    dfs_cb = [df_lgbm_hosei, df_lgbm_no_q]
    score_list_cb = [Const.lgbm_hosei_score, Const.lgbm_no_q_score]

    df_ut = ensemble("pollen_utsunomiya", score_list_ut, dfs_ut, pollen_list_ut)
    df_tk = ensemble("pollen_tokyo", score_list_tk, dfs_tk, pollen_list_tk)
    df_cb = ensemble("pollen_chiba", score_list_cb, dfs_cb, pollen_list_cb)

    #
    df = df_ut.merge(df_tk, on="datetime")
    df = df.merge(df_cb, on="datetime")

    df = df[["datetime", "pollen_utsunomiya", "pollen_chiba", "pollen_tokyo"]]
    return df


def ensemble_7():
    df_train = pd.read_csv("../input/train_v2.csv", index_col=None)
    pollen_list_ut = list(set(df_train["pollen_utsunomiya"].tolist()))
    pollen_list_tk = list(set(df_train["pollen_tokyo"].tolist()))
    pollen_list_cb = list(set(df_train["pollen_chiba"].tolist()))

    df_lgbm = pd.read_csv("submission/sub_42_4.csv", index_col=None)
    df_lgbm_hosei = pd.read_csv("submission/sub_42_4_hosei.csv", index_col=None)
    df_lgbm_q_50 = pd.read_csv("submission/sub_q-50-50-50.csv", index_col=None)
    df_lgbm_no_q = pd.read_csv(
        "submission/sub_42_4_no_q.csv", index_col=None
    )  # same as lgbm_q_50k
    df_svr = pd.read_csv("submission/sub_svr_42_4.csv", index_col=None)

    # ut
    dfs_ut = [df_lgbm, df_lgbm_hosei, df_lgbm_q_50, df_svr]
    score_list_ut = [
        Const.lgbm_score,
        Const.lgbm_hosei_score,
        Const.lgbm_q_50_score,
        Const.svr_score,
    ]

    # tk
    dfs_tk = [df_lgbm_hosei, df_lgbm_no_q]
    score_list_tk = [50, 50]

    # cb
    dfs_cb = [df_lgbm_hosei, df_lgbm_no_q]
    score_list_cb = [50, 50]

    df_ut = ensemble("pollen_utsunomiya", score_list_ut, dfs_ut, pollen_list_ut)
    df_tk = ensemble("pollen_tokyo", score_list_tk, dfs_tk, pollen_list_tk)
    df_cb = ensemble("pollen_chiba", score_list_cb, dfs_cb, pollen_list_cb)

    #
    df = df_ut.merge(df_tk, on="datetime")
    df = df.merge(df_cb, on="datetime")

    df = df[["datetime", "pollen_utsunomiya", "pollen_chiba", "pollen_tokyo"]]
    return df


def parser():
    import argparse

    parser = argparse.ArgumentParser(description="Ensemble")
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    print("==== params =====")
    for key, value in vars(args).items():
        print(f"{key}={value}")
    print("==== params =====")

    return args


def main():
    args = parser()

    # df = ensemble_6()
    df = ensemble_5()
    # df = ensemble_7()

    output_path = "/".join(args.output.split("/")[:-1])
    output_file = args.output.split("/")[-1]

    os.makedirs(output_path, exist_ok=True)

    df.to_csv(f"{output_path}/{output_file}", index=None)


if __name__ == "__main__":
    main()
