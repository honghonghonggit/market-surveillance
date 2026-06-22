"""ROC/PR 커브 + 임계값 트레이드오프 시각화 및 이진 지표.

불균형 데이터에서 임계값을 어디에 두느냐에 따라 정밀도와 재현율이 어떻게 맞교환되는지
보여주는 것이 목적이다. ROC는 전체 분별력을, PR 커브는 (불균형에서 더 정직한) 양성
탐지 성능을 나타낸다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

import matplotlib

matplotlib.use("Agg")  # 헤드리스 저장용 백엔드
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


@dataclass
class BinaryMetrics:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> BinaryMetrics:
    """이진 예측의 정밀도/재현율/F1/혼동행렬(룰 vs ML 공정 비교에 사용)."""
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return BinaryMetrics(precision, recall, f1, tp, fp, fn, tn)


def curve_summary(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """ROC AUC와 평균정밀도(PR AUC). 양성/음성이 모두 있어야 의미가 있다."""
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return {"roc_auc": float("nan"), "average_precision": float("nan")}
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
    }


def save_curves(
    y_true: np.ndarray, scores: np.ndarray, outdir: str = "artifacts"
) -> Dict[str, str]:
    """ROC, PR, 임계값-정밀도/재현율 트레이드오프 3종 플롯을 PNG로 저장."""
    os.makedirs(outdir, exist_ok=True)
    y_true = np.asarray(y_true).astype(int)
    paths: Dict[str, str] = {}

    # 1) ROC
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    paths["roc"] = os.path.join(outdir, "roc_curve.png")
    fig.savefig(paths["roc"], dpi=120)
    plt.close(fig)

    # 2) PR
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    baseline = y_true.mean()  # 무작위 분류기의 PR 기준선 = 양성 비율
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"PR (AP={ap:.3f})")
    ax.axhline(baseline, ls="--", color="gray", linewidth=1,
               label=f"baseline={baseline:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    paths["pr"] = os.path.join(outdir, "pr_curve.png")
    fig.savefig(paths["pr"], dpi=120)
    plt.close(fig)

    # 3) 임계값별 정밀도/재현율 트레이드오프
    thresholds = np.linspace(0.0, 1.0, 101)
    precs, recs = [], []
    for t in thresholds:
        m = binary_metrics(y_true, scores >= t)
        precs.append(m.precision)
        recs.append(m.recall)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(thresholds, precs, label="Precision")
    ax.plot(thresholds, recs, label="Recall")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall vs Threshold")
    ax.legend(loc="lower center")
    fig.tight_layout()
    paths["tradeoff"] = os.path.join(outdir, "threshold_tradeoff.png")
    fig.savefig(paths["tradeoff"], dpi=120)
    plt.close(fig)

    return paths
