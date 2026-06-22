"""ML 분류기 — 룰 기반의 비교 기준선.

핵심 rigor 장치: **시간 기반 train/test 분할**. (계좌, 윈도우) 표본을 윈도우 순서로
정렬해 앞쪽 구간으로 학습하고 뒤쪽 구간으로 평가한다. 무작위 분할 대신 시간 분할을
쓰는 이유는 RA-Testbed의 lookahead bias 방지와 같다 — 미래 정보로 과거를 맞히는
누설을 막고, "과거로 학습해 미래의 조작을 탐지"하는 실제 운영 상황을 모사한다.
조작 에피소드는 짧아 한 윈도우에 들어가므로 분할선을 넘어 새지 않는다.

라벨 누설 차단: 학습 피처는 행동 피처만 사용하고 account_id/window/episode_id는
입력에서 제외한다(계좌 식별자로 외우는 것을 방지).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ..generator.events import Label

# 모델 입력 피처(행동 기반만). account_id/window는 식별자라 제외.
FEATURE_COLS: List[str] = [
    "num_new",
    "num_cancel",
    "num_trade",
    "self_trade_count",
    "max_new_qty",
    "distinct_price_levels",
    "cancel_ratio",
    "order_to_trade_ratio",
    "qty_zscore",
]


@dataclass
class MLConfig:
    train_fraction: float = 0.7      # 앞쪽 70% 윈도우로 학습, 뒤쪽 30%로 평가
    n_estimators: int = 200
    random_state: int = 0


def build_labeled_frame(features: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """피처에 정답을 붙여 학습용 테이블을 만든다. true_label(다중클래스)과
    is_manip(이진 타깃)을 포함."""
    df = features.merge(truth, on=["account_id", "window"], how="left")
    df["true_label"] = df["true_label"].fillna(Label.NORMAL.value)
    df["is_manip"] = (df["true_label"] != Label.NORMAL.value).astype(int)
    return df


def time_split(
    labeled: pd.DataFrame, train_fraction: float = 0.7
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """윈도우 시간 순서로 train/test 분할(lookahead 누설 방지)."""
    windows = np.sort(labeled["window"].unique())
    if len(windows) == 0:
        return labeled.iloc[:0], labeled.iloc[:0]
    cut_idx = int(len(windows) * train_fraction)
    cut_window = windows[min(cut_idx, len(windows) - 1)]
    train = labeled[labeled["window"] < cut_window].copy()
    test = labeled[labeled["window"] >= cut_window].copy()
    return train, test


@dataclass
class MLResult:
    model: RandomForestClassifier
    test: pd.DataFrame          # score(조작 확률), ml_pred 컬럼 포함
    train_size: int
    test_size: int
    train_positives: int
    test_positives: int


def train_and_score(
    train: pd.DataFrame, test: pd.DataFrame, config: MLConfig | None = None
) -> MLResult:
    """이진 분류기(조작 vs 정상)를 학습하고 test에 조작 확률(score)을 매긴다."""
    cfg = config or MLConfig()
    clf = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        random_state=cfg.random_state,
        class_weight="balanced",   # 불균형 보정
    )
    clf.fit(train[FEATURE_COLS], train["is_manip"])

    test = test.copy()
    test["score"] = clf.predict_proba(test[FEATURE_COLS])[:, 1]
    test["ml_pred"] = clf.predict(test[FEATURE_COLS])
    return MLResult(
        model=clf,
        test=test,
        train_size=len(train),
        test_size=len(test),
        train_positives=int(train["is_manip"].sum()),
        test_positives=int(test["is_manip"].sum()),
    )


def feature_importances(model: RandomForestClassifier) -> pd.Series:
    """피처 중요도(어떤 신호가 탐지에 기여했는지)."""
    return pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(
        ascending=False
    )
