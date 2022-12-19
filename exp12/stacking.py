import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import Normalizer, StandardScaler
from main import getNearestValue
from sklearn.model_selection import GridSearchCV
import numpy as np
import os
from ensemble import Const, ensemble


def merge_data(dfs, target_col: str, df_train=None):

    df = dfs[0][["datetime", target_col]]
    for _df in dfs[1:]:
        df = df.merge(_df[["datetime", target_col]], on="datetime")

    if df_train is not None:
        df_label = df_train[["datetime", target_col]].rename(
            columns={target_col: "target"}
        )
        df = df.merge(df_label, on="datetime")

    return df


def stacking_ridge(df, df_test, pollen_list, target_col):
    """
    Ref https://zenn.dev/nishimoto/articles/4fb2a46cee0e43
    """

    cols = [col for col in df.columns if col not in ["datetime", "target"]]

    estimator = Ridge(normalize=False, random_state=0, alpha=0.001)

    param_grid = {
        "alpha": [0.001, 0.01, 0.1, 1, 10],
    }
    grid_search = GridSearchCV(estimator, param_grid)

    grid_search.fit(df[cols], np.log1p(df["target"]))
    estimator = grid_search.best_estimator_

    # normalizer = Normalizer().fit(df[cols])
    # estimator = LinearRegression()
    # estimator.fit(normalizer.transform(df[cols]), np.log1p(df["target"]))

    # print("===train")
    # print(df[cols])

    # print("===target")
    # print(df["target"])

    # print("===test")
    # print(df_test[cols])

    preds = estimator.predict(df_test[cols])
    preds = np.expm1(preds)

    # print("===pred")
    # print(preds)

    _df = df_test[["datetime"]].copy()
    _df.loc[:, target_col] = preds

    print("===coef")
    print(estimator.coef_)

    _df[target_col] = np.array(
        [getNearestValue(pollen_list, v) for v in _df[target_col]]
    )

    return _df


def stacking_1():
    df_train = pd.read_csv("../input/train_v2.csv", index_col=None)
    df_train.loc[df_train["pollen_utsunomiya"] == -9998, "pollen_utsunomiya"] = 0
    df_train.loc[df_train["pollen_chiba"] == -9998, "pollen_chiba"] = 0
    df_train.loc[df_train["pollen_tokyo"] == -9998, "pollen_tokyo"] = 0

    pollen_list_ut = list(set(df_train["pollen_utsunomiya"].tolist()))
    pollen_list_tk = list(set(df_train["pollen_tokyo"].tolist()))
    pollen_list_cb = list(set(df_train["pollen_chiba"].tolist()))

    # read data
    df_lgbm_tr = pd.read_csv("submission/sub_42_4_oof.csv", index_col=None)
    df_lgbm_te = pd.read_csv("submission/sub_42_4.csv", index_col=None)

    df_lgbm_hosei_tr = pd.read_csv("submission/sub_42_4_hosei_oof.csv", index_col=None)
    df_lgbm_hosei_te = pd.read_csv("submission/sub_42_4_hosei.csv", index_col=None)

    df_lgbm_q_50_tr = pd.read_csv("submission/sub_q-50-50-50_oof.csv", index_col=None)
    df_lgbm_q_50_te = pd.read_csv("submission/sub_q-50-50-50.csv", index_col=None)

    df_lgbm_no_q_tr = pd.read_csv(
        "submission/sub_42_4_no_q_oof.csv", index_col=None
    )  # same as lgbm_q_50k
    df_lgbm_no_q_te = pd.read_csv(
        "submission/sub_42_4_no_q.csv", index_col=None
    )  # same as lgbm_q_50k

    df_svr_tr = pd.read_csv("submission/sub_svr_42_4_oof.csv", index_col=None)
    df_svr_te = pd.read_csv("submission/sub_svr_42_4.csv", index_col=None)

    df_mlp_tr = pd.read_csv("submission/sub_mlp_42_4_oof.csv", index_col=None)
    df_mlp_te = pd.read_csv("submission/sub_mlp_42_4.csv", index_col=None)

    # preprocess

    df_train_ut = merge_data(
        [df_lgbm_tr, df_lgbm_hosei_tr, df_lgbm_q_50_tr, df_svr_tr, df_mlp_tr],
        "pollen_utsunomiya",
        df_train,
    )

    df_test_ut = merge_data(
        [df_lgbm_te, df_lgbm_hosei_te, df_lgbm_q_50_te, df_svr_te, df_mlp_te],
        "pollen_utsunomiya",
    )

    df_train_tk = merge_data(
        [df_lgbm_hosei_tr, df_lgbm_no_q_tr], "pollen_tokyo", df_train
    )

    df_test_tk = merge_data([df_lgbm_hosei_te, df_lgbm_no_q_te], "pollen_tokyo")

    df_train_cb = merge_data(
        [df_lgbm_hosei_tr, df_lgbm_no_q_tr], "pollen_chiba", df_train
    )
    df_test_cb = merge_data([df_lgbm_hosei_te, df_lgbm_no_q_te], "pollen_chiba")

    # stacking
    df_ut = stacking_ridge(df_train_ut, df_test_ut, pollen_list_ut, "pollen_utsunomiya")
    df_tk = stacking_ridge(df_train_tk, df_test_tk, pollen_list_tk, "pollen_tokyo")
    df_cb = stacking_ridge(df_train_cb, df_test_cb, pollen_list_cb, "pollen_chiba")

    # post processing
    df = df_ut.merge(df_tk, on="datetime")
    df = df.merge(df_cb, on="datetime")

    df = df[["datetime", "pollen_utsunomiya", "pollen_chiba", "pollen_tokyo"]]
    return df


def stacking_2():
    df_train = pd.read_csv("../input/train_v2.csv", index_col=None)
    df_train.loc[df_train["pollen_utsunomiya"] == -9998, "pollen_utsunomiya"] = 0
    df_train.loc[df_train["pollen_chiba"] == -9998, "pollen_chiba"] = 0
    df_train.loc[df_train["pollen_tokyo"] == -9998, "pollen_tokyo"] = 0

    pollen_list_ut = list(set(df_train["pollen_utsunomiya"].tolist()))
    pollen_list_tk = list(set(df_train["pollen_tokyo"].tolist()))
    pollen_list_cb = list(set(df_train["pollen_chiba"].tolist()))

    # read data
    df_lgbm_tr = pd.read_csv("submission/sub_42_4_oof.csv", index_col=None)
    df_lgbm_te = pd.read_csv("submission/sub_42_4.csv", index_col=None)

    df_lgbm_hosei_tr = pd.read_csv("submission/sub_42_4_hosei_oof.csv", index_col=None)
    df_lgbm_hosei_te = pd.read_csv("submission/sub_42_4_hosei.csv", index_col=None)

    df_lgbm_q_50_tr = pd.read_csv("submission/sub_q-50-50-50_oof.csv", index_col=None)
    df_lgbm_q_50_te = pd.read_csv("submission/sub_q-50-50-50.csv", index_col=None)

    df_lgbm_no_q_tr = pd.read_csv(
        "submission/sub_42_4_no_q_oof.csv", index_col=None
    )  # same as lgbm_q_50k
    df_lgbm_no_q_te = pd.read_csv(
        "submission/sub_42_4_no_q.csv", index_col=None
    )  # same as lgbm_q_50k

    df_svr_tr = pd.read_csv("submission/sub_svr_42_4_oof.csv", index_col=None)
    df_svr_te = pd.read_csv("submission/sub_svr_42_4.csv", index_col=None)

    df_mlp_tr = pd.read_csv("submission/sub_mlp_42_4_oof.csv", index_col=None)
    df_mlp_te = pd.read_csv("submission/sub_mlp_42_4.csv", index_col=None)

    # preprocess

    df_train_ut = merge_data(
        [df_lgbm_tr, df_lgbm_hosei_tr, df_lgbm_q_50_tr, df_svr_tr, df_mlp_tr],
        "pollen_utsunomiya",
        df_train,
    )

    df_train_ut = df_train_ut[df_train_ut["target"] <= 50].reset_index(drop=True)

    df_test_ut = merge_data(
        [df_lgbm_te, df_lgbm_hosei_te, df_lgbm_q_50_te, df_svr_te, df_mlp_te],
        "pollen_utsunomiya",
    )

    # tk
    dfs_tk = [df_lgbm_hosei_te, df_lgbm_no_q_te]
    score_list_tk = [Const.lgbm_hosei_score, Const.lgbm_no_q_score]

    # cb
    dfs_cb = [df_lgbm_hosei_te, df_lgbm_no_q_te]
    score_list_cb = [Const.lgbm_hosei_score, Const.lgbm_no_q_score]

    df_ut = stacking_ridge(df_train_ut, df_test_ut, pollen_list_ut, "pollen_utsunomiya")
    df_tk = ensemble("pollen_tokyo", score_list_tk, dfs_tk, pollen_list_tk)
    df_cb = ensemble("pollen_chiba", score_list_cb, dfs_cb, pollen_list_cb)

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

    df = stacking_2()

    output_path = "/".join(args.output.split("/")[:-1])
    output_file = args.output.split("/")[-1]

    os.makedirs(output_path, exist_ok=True)

    df.to_csv(f"{output_path}/{output_file}", index=None)


if __name__ == "__main__":
    main()
