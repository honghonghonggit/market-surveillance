"""평가 — ground truth를 (계좌, 윈도우) 단위로 집계해 예측과 정합 비교.

이 프로젝트의 핵심: 합성 데이터에 패턴을 주입해 정답을 알고 있으므로 정밀도·
재현율·혼동행렬이 진짜 의미를 갖는다.

왜 accuracy 하나로 평가하면 안 되는가:
조작은 전체의 극소수(불균형)다. "전부 정상"이라고만 예측해도 accuracy는 99%를
넘지만 조작은 하나도 못 잡는다(recall 0). 그래서 정밀도(오탐 억제)와 재현율
(미탐 억제)을 함께 본다.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from ..features.windows import DEFAULT_WINDOW_MS, window_index
from ..generator.events import EventType, Label, OrderEvent, GroundTruthLabel

Key = Tuple[str, int]


def build_truth(
    events: Iterable[OrderEvent],
    labels: Iterable[GroundTruthLabel],
    window_ms: int = DEFAULT_WINDOW_MS,
) -> pd.DataFrame:
    """order_id 기준 라벨을 (계좌, 윈도우) 단위 정답 라벨로 집계.

    한 (계좌, 윈도우)는 그 계좌의 조작 주문 이벤트가 하나라도 들어 있으면
    조작으로 본다. 피처 집계와 동일한 window_index를 써서 단위를 일치시킨다.
    """
    oid_label: Dict[int, Label] = {l.order_id: l.label for l in labels}
    truth: Dict[Key, set] = defaultdict(set)

    for e in events:
        w = window_index(e.ts, window_ms)
        if e.event_type is EventType.TRADE:
            sides = [
                (e.maker_order_id, e.maker_account),
                (e.taker_order_id, e.taker_account),
            ]
        else:
            sides = [(e.order_id, e.account_id)]
        for oid, account in sides:
            if oid in oid_label and account is not None:
                truth[(account, w)].add(oid_label[oid])

    records = []
    for (account_id, w), lbls in truth.items():
        # 한 윈도우에 두 패턴이 겹치면 워시를 우선(자기체결이 더 확정적)
        if Label.WASH_TRADING in lbls:
            true_label = Label.WASH_TRADING.value
        elif Label.SPOOFING in lbls:
            true_label = Label.SPOOFING.value
        else:
            true_label = next(iter(lbls)).value
        records.append({"account_id": account_id, "window": w, "true_label": true_label})

    return pd.DataFrame.from_records(
        records, columns=["account_id", "window", "true_label"]
    )


@dataclass
class EvalReport:
    n_samples: int
    n_positives: int          # 실제 조작 (계좌,윈도우) 수
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    per_class: Dict[str, Dict[str, float]]

    def format(self) -> str:
        lines = [
            "=" * 56,
            "탐지 평가 리포트  (단위: 계좌 × 시간윈도우)",
            "=" * 56,
            f"표본 수: {self.n_samples}   |   실제 조작: {self.n_positives} "
            f"({self.n_positives / self.n_samples:.2%})  ← 불균형",
            "",
            "혼동행렬 (조작 탐지 = positive)",
            f"               예측:조작   예측:정상",
            f"  실제:조작    TP={self.tp:<6}   FN={self.fn:<6}",
            f"  실제:정상    FP={self.fp:<6}   TN={self.tn:<6}",
            "",
            f"  정밀도(Precision): {self.precision:.3f}   (탐지한 것 중 진짜 조작 비율 → 오탐 억제)",
            f"  재현율(Recall):    {self.recall:.3f}   (진짜 조작 중 탐지 비율 → 미탐 억제)",
            f"  F1:                {self.f1:.3f}",
            f"  Accuracy:          {self.accuracy:.3f}   ← 불균형이라 단독 지표로 부적절",
            "",
            "패턴별 정밀도/재현율",
        ]
        for cls, m in self.per_class.items():
            lines.append(
                f"  {cls:<14} P={m['precision']:.3f}  R={m['recall']:.3f}  "
                f"(support={int(m['support'])})"
            )
        lines.append("=" * 56)
        return "\n".join(lines)


def evaluate(predictions: pd.DataFrame, truth: pd.DataFrame) -> EvalReport:
    """예측(predicted_label/is_alert)과 정답(true_label)을 (계좌,윈도우)로 정합 비교."""
    df = predictions.merge(truth, on=["account_id", "window"], how="left")
    df["true_label"] = df["true_label"].fillna(Label.NORMAL.value)

    true_manip = df["true_label"] != Label.NORMAL.value
    pred_manip = df["is_alert"]

    tp = int((true_manip & pred_manip).sum())
    fp = int((~true_manip & pred_manip).sum())
    fn = int((true_manip & ~pred_manip).sum())
    tn = int((~true_manip & ~pred_manip).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(df) if len(df) else 0.0

    per_class: Dict[str, Dict[str, float]] = {}
    for cls in (Label.SPOOFING.value, Label.WASH_TRADING.value):
        is_true = df["true_label"] == cls
        is_pred = df["predicted_label"] == cls
        c_tp = int((is_true & is_pred).sum())
        c_fp = int((~is_true & is_pred).sum())
        c_fn = int((is_true & ~is_pred).sum())
        per_class[cls] = {
            "precision": c_tp / (c_tp + c_fp) if (c_tp + c_fp) else 0.0,
            "recall": c_tp / (c_tp + c_fn) if (c_tp + c_fn) else 0.0,
            "support": int(is_true.sum()),
        }

    return EvalReport(
        n_samples=len(df),
        n_positives=int(true_manip.sum()),
        tp=tp, fp=fp, fn=fn, tn=tn,
        precision=precision, recall=recall, f1=f1, accuracy=accuracy,
        per_class=per_class,
    )
