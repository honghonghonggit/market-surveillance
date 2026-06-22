"""Phase 3 분석 — 패턴별 탐지 정확도 분해, 탐지 지연, 오탐 케이스 분석.

평가가 "전체 정밀도/재현율 한 줄"에서 멈추지 않도록, 정답(ground truth)을 알고 있다는
강점을 끝까지 활용한다:
- 어떤 패턴을, 어떤 강도에서 놓치는가 (패턴 × 강도 recall)
- 조작 발생부터 탐지까지 얼마나 걸리는가 (탐지 지연)
- 임계값을 낮출 때 무엇을 오탐하는가 (정밀도 비용)
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from ..features.windows import DEFAULT_WINDOW_MS, window_index
from ..generator.events import EventType, GroundTruthLabel, OrderEvent


def build_episode_table(
    events: Iterable[OrderEvent],
    labels: Iterable[GroundTruthLabel],
    window_ms: int = DEFAULT_WINDOW_MS,
) -> pd.DataFrame:
    """주입된 조작 에피소드 1건당 1행. 패턴·계좌·시작시각·강도·점유 윈도우를 집계.

    강도(intensity)는 패턴별 핵심 신호로 정의한다:
    스푸핑=최대 주문량, 레이어링=서로 다른 레벨 수, 워시=자기체결 수.
    """
    oid_info: Dict[int, Tuple[str, str]] = {
        l.order_id: (l.label.value, l.episode_id) for l in labels
    }

    def _blank():
        return {
            "pattern": None, "account": None,
            "ts_min": math.inf, "ts_max": -math.inf,
            "n_orders": 0, "max_qty": 0, "self_trades": 0,
            "prices": set(), "windows": set(),
        }

    rec: Dict[str, dict] = defaultdict(_blank)

    for e in events:
        if e.event_type is EventType.TRADE:
            hits = [(e.maker_order_id, e.maker_account), (e.taker_order_id, e.taker_account)]
            hits = [(o, a) for (o, a) in hits if o in oid_info]
        else:
            hits = [(e.order_id, e.account_id)] if e.order_id in oid_info else []
        if not hits:
            continue

        for oid, account in hits:
            _, eid = oid_info[oid]
            r = rec[eid]
            r["pattern"] = oid_info[oid][0]
            r["account"] = account
            r["ts_min"] = min(r["ts_min"], e.ts)
            r["ts_max"] = max(r["ts_max"], e.ts)
            r["windows"].add(window_index(e.ts, window_ms))
            if e.event_type is EventType.NEW:
                r["n_orders"] += 1
                r["max_qty"] = max(r["max_qty"], e.quantity)
                r["prices"].add(e.price)

        if e.event_type is EventType.TRADE and e.is_self_trade and e.maker_order_id in oid_info:
            rec[oid_info[e.maker_order_id][1]]["self_trades"] += 1

    rows: List[dict] = []
    for eid, r in rec.items():
        pattern = r["pattern"]
        distinct_levels = len(r["prices"])
        intensity = (
            r["max_qty"] if pattern == "SPOOFING"
            else distinct_levels if pattern == "LAYERING"
            else r["self_trades"]
        )
        rows.append({
            "episode_id": eid, "pattern": pattern, "account_id": r["account"],
            "start_ts": int(r["ts_min"]), "end_ts": int(r["ts_max"]),
            "n_orders": r["n_orders"], "max_qty": r["max_qty"],
            "distinct_levels": distinct_levels, "self_trades": r["self_trades"],
            "intensity": intensity, "windows": sorted(r["windows"]),
        })
    return pd.DataFrame(rows).sort_values("start_ts").reset_index(drop=True)


def episode_detection(
    episodes: pd.DataFrame,
    predictions: pd.DataFrame,
    window_ms: int = DEFAULT_WINDOW_MS,
) -> pd.DataFrame:
    """에피소드별 탐지 여부/예측 패턴/탐지 지연을 계산.

    탐지 지연 = (최초로 알림이 난 윈도우의 마감 시각) - (에피소드 시작 시각).
    윈도우 단위 집계라 탐지는 윈도우 마감 시점에 가능하므로 마감 시각을 쓴다.
    """
    alert_map: Dict[Tuple[str, int], Tuple[bool, str]] = {
        (row.account_id, row.window): (bool(row.is_alert), row.predicted_label)
        for row in predictions.itertuples()
    }

    out = episodes.copy()
    detected, pred_label, latency, correct = [], [], [], []
    for ep in episodes.itertuples():
        first_w = None
        plabel = None
        for w in ep.windows:
            a = alert_map.get((ep.account_id, w))
            if a is not None and a[0]:
                first_w, plabel = w, a[1]
                break
        is_det = first_w is not None
        detected.append(is_det)
        pred_label.append(plabel)
        latency.append((first_w + 1) * window_ms - ep.start_ts if is_det else None)
        correct.append(bool(is_det and plabel == ep.pattern))

    out["detected"] = detected
    out["pred_label"] = pred_label
    out["latency_ms"] = latency
    out["correct_pattern"] = correct
    return out


def recall_by_pattern(episode_det: pd.DataFrame) -> pd.DataFrame:
    """패턴별 탐지율(recall)과 패턴까지 정확히 맞춘 비율."""
    g = episode_det.groupby("pattern")
    return pd.DataFrame({
        "episodes": g.size(),
        "recall": g["detected"].mean(),
        "correct_pattern_rate": g["correct_pattern"].mean(),
        "avg_latency_ms": g["latency_ms"].mean(),
    }).round(3)


def recall_by_intensity(episode_det: pd.DataFrame) -> pd.DataFrame:
    """패턴 내에서 강도를 약/강(중앙값 분할)으로 나눠 recall을 본다.
    약한 에피소드를 더 많이 놓친다는 점(고정 임계값의 한계)을 정량화."""
    rows = []
    for pattern, grp in episode_det.groupby("pattern"):
        if len(grp) < 2:
            rows.append({"pattern": pattern, "intensity_bucket": "all",
                         "episodes": len(grp), "recall": grp["detected"].mean()})
            continue
        med = grp["intensity"].median()
        for bucket, mask in (("약함(≤중앙값)", grp["intensity"] <= med),
                             ("강함(>중앙값)", grp["intensity"] > med)):
            sub = grp[mask]
            if len(sub):
                rows.append({"pattern": pattern, "intensity_bucket": bucket,
                             "episodes": len(sub), "recall": round(sub["detected"].mean(), 3)})
    return pd.DataFrame(rows)


def false_positive_profile(
    predictions: pd.DataFrame, truth: pd.DataFrame, feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    """오탐(정상인데 알림) 케이스의 발화 룰 분포와 평균 피처 프로파일을 반환.
    임계값을 낮췄을 때 '무엇을' 오탐하는지 = 정밀도 비용을 설명한다."""
    m = predictions.merge(truth, on=["account_id", "window"], how="left")
    m["true_label"] = m["true_label"].fillna("NORMAL")
    fp = m[(m["is_alert"]) & (m["true_label"] == "NORMAL")]
    by_rule = fp["predicted_label"].value_counts()
    profile = fp[feature_cols].mean() if len(fp) else pd.Series(dtype=float)
    return by_rule.to_frame("count"), profile
