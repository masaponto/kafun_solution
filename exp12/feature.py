import pandas as pd
import numpy as np

from typing import List, Tuple, Union, Dict

from scipy.signal import savgol_filter
import geocoder
import datetime
from dateutil import tz
import re
import ephem
import tqdm


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
    args,
    polyorder: int = 2,
    overwite: bool = False,
    cols=[],
) -> pd.DataFrame:
    _df = df.copy()
    for col in cols:
        np.random.seed(args.seed)
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


# ラグ特徴/ローリング特徴量
def add_lag_feat_presudo(df: pd.DataFrame, feat: List[str], group: str) -> pd.DataFrame:
    outputs = [df]

    grp_df = df.groupby(group)  # year ごとにシフトする。 各年1~6月期間しかないのでこのようにする。

    # for lag in [1, 2, 3, 4, 5]:
    # for lag in list(range(1, 25)):
    #     # shift
    #     outputs.append(grp_df[feat].shift(lag).add_prefix(f"shift_{lag}_"))
    #     # diff
    #     outputs.append(grp_df[feat].diff(lag).add_prefix(f"diff_{lag}_"))

    # rolling
    windows = [i * 24 for i in range(1, 3)]
    # windows = [3] + [i * 24 for i in range(1, 10)]
    for window in windows:
        tmp_df = grp_df[feat].rolling(window, min_periods=1)
        tmp_df = tmp_df.mean().add_prefix(f"rolling_{window}_mean_")
        outputs.append(tmp_df.reset_index(drop=True).copy())

        # tmp_df = grp_df[feat].rolling(window, min_periods=1)
        # tmp_df = tmp_df.max().add_prefix(f"rolling_{window}_max_")
        # outputs.append(tmp_df.reset_index(drop=True).copy())

        # tmp_df = grp_df[feat].rolling(window, min_periods=1)
        # tmp_df = tmp_df.min().add_prefix(f"rolling_{window}_min_")
        # outputs.append(tmp_df.reset_index(drop=True).copy())

    _df = pd.concat(outputs, axis=1)
    return _df


def temperature_decompose(df: pd.DataFrame, cols: List[str], args) -> pd.DataFrame:
    from statsmodels.tsa.seasonal import STL

    _df = df.copy()
    np.random.seed(args.seed)
    # for col in ["temperature_tokyo", "temperature_utsunomiya", "temperature_chiba"]:
    for col in cols:
        np.random.seed(args.seed)
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
        df_origin[f"{col}_cos"] = df_origin[f"{col}_cos"].fillna(0)
        df_origin[f"{col}_sin"] = df_origin[f"{col}_sin"].fillna(0)

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


def get_latlng():

    loc_list = ["宇都宮市中央生涯学習センター", "千葉県環境研究センター", "東京都多摩小平保健所"]
    key_list = ["utsunomiya", "chiba", "tokyo"]

    LATLNG_DICT = {}

    try:

        for i, key in zip(loc_list, key_list):
            loc = geocoder.osm(i, timeout=5.0)
            LATLNG_DICT[key] = tuple(loc.latlng)

    except Exception as e:
        print(e)
        print("APIでのデータ取得に失敗したため、成功時に取得したデータを使います。")
        LATLNG_DICT = {
            "utsunomiya": (36.5594462, 139.88265145),
            "chiba": (35.633642, 140.077749),
            "tokyo": (35.7298652, 139.51664115548698),
        }

    return LATLNG_DICT


def add_rising_setting(df: pd.DataFrame) -> pd.DataFrame:
    _df = df.copy()

    latlng_dict = get_latlng()
    sun = ephem.Sun()

    ut = ephem.Observer()
    ut.lat = latlng_dict["utsunomiya"][0]
    ut.lon = latlng_dict["utsunomiya"][1]

    tk = ephem.Observer()
    tk.lat = latlng_dict["tokyo"][0]
    tk.lon = latlng_dict["tokyo"][1]

    cb = ephem.Observer()
    cb.lat = latlng_dict["chiba"][0]
    cb.lon = latlng_dict["chiba"][1]

    def get_rising_time(
        observer: ephem.Observer, sun: ephem.Sun, datetime: datetime.datetime
    ):
        observer.date = datetime
        return ephem.localtime(observer.next_rising(sun))

    def get_setting_time(
        observer: ephem.Observer, sun: ephem.Sun, datetime: datetime.datetime
    ):
        observer.date = datetime
        return ephem.localtime(observer.next_setting(sun))

    _df["ymd"] = _df["datetime_dt"].dt.strftime("%Y%m%d")
    dfs = []
    JST = tz.gettz("Asia/Tokyo")
    UTC = tz.gettz("UTC")

    for i, __df in tqdm.tqdm(
        _df[["ymd", "datetime_dt", "hour_sin", "hour_cos"]].groupby("ymd")
    ):
        # datetime_min = __df["datetime_dt"].median().values.tz_localize("Asia/Tokyo")
        datetime_min = __df["datetime_dt"].min().to_pydatetime()
        # print(datetime_min, type(datetime_min))
        datetime_min = (
            datetime_min.replace(tzinfo=JST).astimezone(UTC).replace(tzinfo=None)
        )
        # print(datetime_min)
        datetime_rising_ut = (
            get_rising_time(ut, sun, datetime_min).astimezone(JST).replace(tzinfo=None)
        )
        datetime_setting_ut = (
            get_setting_time(ut, sun, datetime_min).astimezone(JST).replace(tzinfo=None)
        )

        datetime_rising_tk = (
            get_rising_time(tk, sun, datetime_min).astimezone(JST).replace(tzinfo=None)
        )
        datetime_setting_tk = (
            get_setting_time(tk, sun, datetime_min).astimezone(JST).replace(tzinfo=None)
        )

        datetime_rising_cb = (
            get_rising_time(cb, sun, datetime_min).astimezone(JST).replace(tzinfo=None)
        )
        datetime_setting_cb = (
            get_setting_time(cb, sun, datetime_min).astimezone(JST).replace(tzinfo=None)
        )

        __df["sun_rising_rate_ut"] = (__df["datetime_dt"] - datetime_rising_ut).apply(
            lambda x: x.total_seconds() / 60
        ) / ((datetime_setting_ut - datetime_rising_ut).total_seconds() / 60)

        __df.loc[__df["sun_rising_rate_ut"] < 0, "sun_rising_rate_ut"] = 0
        __df.loc[__df["sun_rising_rate_ut"] >= 1, "sun_rising_rate_ut"] = 0

        __df["sun_rising_rate_tk"] = (__df["datetime_dt"] - datetime_rising_tk).apply(
            lambda x: x.total_seconds() / 60
        ) / ((datetime_setting_tk - datetime_rising_tk).total_seconds() / 60)

        __df.loc[__df["sun_rising_rate_tk"] < 0, "sun_rising_rate_tk"] = 0
        __df.loc[__df["sun_rising_rate_tk"] >= 1, "sun_rising_rate_tk"] = 0

        __df["sun_rising_rate_cb"] = (__df["datetime_dt"] - datetime_rising_cb).apply(
            lambda x: x.total_seconds() / 60
        ) / ((datetime_setting_cb - datetime_rising_cb).total_seconds() / 60)

        __df.loc[__df["sun_rising_rate_cb"] < 0, "sun_rising_rate_cb"] = 0
        __df.loc[__df["sun_rising_rate_cb"] >= 1, "sun_rising_rate_cb"] = 0

        __df["diff_rising_tk"] = (__df["datetime_dt"] - datetime_rising_tk).apply(
            lambda x: x.total_seconds() / 60
        )
        __df["diff_setting_tk"] = (datetime_setting_tk - __df["datetime_dt"]).apply(
            lambda x: x.total_seconds() / 60
        )

        # __df["rising_tk"] = ((datetime_setting_tk - datetime_rising_tk).total_seconds() / 60)

        __df["diff_rising_tk"] = (__df["datetime_dt"] - datetime_rising_tk).apply(
            lambda x: x.total_seconds() / 60
        )
        __df["diff_setting_tk"] = (datetime_setting_tk - __df["datetime_dt"]).apply(
            lambda x: x.total_seconds() / 60
        )

        __df["diff_rising_cb"] = (__df["datetime_dt"] - datetime_rising_cb).apply(
            lambda x: x.total_seconds() / 60
        )
        __df["diff_setting_cb"] = (datetime_setting_cb - __df["datetime_dt"]).apply(
            lambda x: x.total_seconds() / 60
        )

        __df["hour_sin_mult_sun_rate_tk"] = (
            __df["sun_rising_rate_tk"] * __df["hour_sin"]
        )
        __df["hour_sin_mult_sun_rate_cb"] = (
            __df["sun_rising_rate_cb"] * __df["hour_sin"]
        )
        __df["hour_sin_mult_sun_rate_ut"] = (
            __df["sun_rising_rate_ut"] * __df["hour_sin"]
        )

        __df["hour_cos_mult_sun_rate_tk"] = (
            __df["sun_rising_rate_tk"] * __df["hour_cos"]
        )
        __df["hour_cos_mult_sun_rate_cb"] = (
            __df["sun_rising_rate_cb"] * __df["hour_cos"]
        )
        __df["hour_cos_mult_sun_rate_ut"] = (
            __df["sun_rising_rate_ut"] * __df["hour_cos"]
        )

        dfs.append(__df)

    df_rising_setting = pd.concat(dfs)
    # print(df_rising_setting.columns.tolist())

    # _df = _df.merge(df_rising_setting, on="datetime_dt", how="left")
    _df = pd.concat(
        [
            _df.drop("ymd", axis=1),
            df_rising_setting.drop(
                ["ymd", "datetime_dt", "hour_sin", "hour_cos"], axis=1
            ),
        ],
        axis=1,
    )

    return _df


def make_feature_ut(df: pd.DataFrame, args) -> pd.DataFrame:
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

    _df = temperature_decompose(_df, ["temperature_utsunomiya"], args)

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

    _df = add_rising_setting(_df)

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
            # "sun_rising_rate_tk",
            # "sun_rising_rate_cb",
            # "diff_setting_tk",
            # "diff_rising_cb",
            # "diff_setting_cb",
        ],
        axis=1,
    )

    return _df


def make_feature_ut_2(df: pd.DataFrame, args) -> pd.DataFrame:
    _df = df.copy()
    _df = remove_noise(
        _df,
        window_length=7,
        args=args,
        overwite=False,
        cols=[
            "pollen_tokyo",
            "pollen_chiba",
        ],
    )
    _df = add_lag_feat(
        _df,
        ["pollen_tokyo", "pollen_chiba"],
        "year",
    )

    _df = add_lag_feat_presudo(
        _df,
        ["pollen_utsunomiya_pseudo"],
        "year",
    )

    _df = _df.drop("pollen_utsunomiya_pseudo", axis=1)

    return _df


def make_feature_tk(df: pd.DataFrame, args) -> pd.DataFrame:
    _df = df.copy()
    _df = time_feat(_df)

    # 降水量カウント
    _df = add_precipitation_zero_count_feat(_df)
    _df = remove_noise(
        _df,
        window_length=7,
        args=args,
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
            "temperature_tokyo",
            "windspeed_tokyo",
            "precipitation_utsunomiya",
            "precipitation_chiba",
            #####
            "precipitation_tokyo",
            # "temperature_utsunomiya",
            # "temperature_chiba",
            # "windspeed_chiba",
            # "windspeed_utsunomiya",
            #####
        ],
        "year",
    )

    _df = temperature_decompose(_df, ["temperature_tokyo"], args)

    _df = add_wind_direction_to_cos_sin(
        _df,
        [
            # "winddirection_utsunomiya",
            "winddirection_tokyo",
            # "winddirection_chiba"
        ],
    )

    _df = add_sekisan_ondo2(_df, cols=["temperature_tokyo"])

    _df = add_rising_setting(_df)

    _df = _df.drop(
        [
            "winddirection_utsunomiya",
            "winddirection_tokyo",
            "winddirection_chiba",
        ],
        axis=1,
    )

    return _df


def make_feature_tk_2(df: pd.DataFrame, args) -> pd.DataFrame:
    _df = df.copy()
    # _df = remove_noise(
    #     _df,
    #     window_length=7,
    #     overwite=False,
    #     cols=[
    #         "pollen_utsunomiya",
    #         "pollen_chiba",
    #     ],
    # )
    _df = add_lag_feat(
        _df,
        ["pollen_utsunomiya", "pollen_chiba"],
        "year",
    )

    _df = add_lag_feat_presudo(
        _df,
        ["pollen_tokyo_pseudo"],
        "year",
    )

    # _df["windspeed_mult_pollen_chiba"] = _df["windspeed_chiba"] * _df["pollen_chiba"]
    # _df["windspeed_mult_pollen_utsunomiya_cos"] = (
    #    _df["pollen_utsunomiya"] * _df["windspeed_utsunomiya_cos_mult_window_speed"]
    # )

    _df = _df.drop("pollen_tokyo_pseudo", axis=1)

    return _df


def make_feature_cb(df: pd.DataFrame, args) -> pd.DataFrame:
    _df = df.copy()
    _df = time_feat(_df)

    # 降水量カウント
    _df = add_precipitation_zero_count_feat(_df)

    _df = remove_noise(
        _df,
        window_length=7,
        args=args,
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

    _df = add_rising_setting(_df)

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


def make_feature_cb_2(df: pd.DataFrame, args) -> pd.DataFrame:
    _df = df.copy()
    _df = remove_noise(
        _df,
        window_length=7,
        args=args,
        overwite=False,
        cols=[
            "pollen_utsunomiya",
            "pollen_tokyo",
        ],
    )
    _df = add_lag_feat(
        _df,
        ["pollen_utsunomiya", "pollen_tokyo"],
        "year",
    )

    _df = add_lag_feat_presudo(
        _df,
        ["pollen_chiba_pseudo"],
        "year",
    )

    # _df["windspeed_mult_pollen_tokyo"] = _df["windspeed_tokyo"] * _df["pollen_tokyo"]
    _df = _df.drop("pollen_chiba_pseudo", axis=1)

    return _df
