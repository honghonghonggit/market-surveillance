"""Streamlit 대시보드 — 합성 주문흐름 + 탐지 알림 + 평가 지표.

실행:  streamlit run src/surveillance/app.py

100% 합성 데이터 기반이며, 실제 종목·거래소·투자자와 무관하다. 코스콤 CAMS를
'그대로 재현'한 것이 아니라 시세조종 탐지의 핵심 원리를 구현한 교육·포트폴리오용이다.
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run`에서도 surveillance 패키지를 찾도록 src를 경로에 추가.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
# 차트 한글 깨짐 방지. 로컬(Windows)=맑은 고딕, 배포(Linux/Streamlit Cloud)=나눔고딕
# (packages.txt의 fonts-nanum). matplotlib는 설치된 첫 폰트를 쓰므로 양쪽을 모두 둔다.
matplotlib.rcParams["font.family"] = ["Malgun Gothic", "NanumGothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from surveillance.detection.curves import binary_metrics, curve_summary, save_curves  # noqa: E402
from surveillance.detection.evaluate import build_truth, evaluate  # noqa: E402
from surveillance.detection.ml import (  # noqa: E402
    build_labeled_frame,
    feature_importances,
    time_split,
    train_and_score,
)
from surveillance.detection.rules import RuleConfig, predict  # noqa: E402
from surveillance.features.account_features import build_features  # noqa: E402
from surveillance.features.windows import DEFAULT_WINDOW_MS  # noqa: E402
from surveillance.generator.dataset import DatasetConfig, generate_dataset  # noqa: E402
from surveillance.generator.events import EventType  # noqa: E402
from surveillance.generator.injectors import InjectionConfig  # noqa: E402

st.set_page_config(page_title="이상거래 탐지(시장감시) 엔진", layout="wide")


@st.cache_data(show_spinner="합성 데이터 생성 중...")
def _generate(seed: int, duration: int, hard: bool):
    inj = (
        InjectionConfig(randomize_intensity=True, camouflage=True,
                        spoof_qty_range=(10, 90), layering_levels_range=(2, 8))
        if hard else InjectionConfig()
    )
    n = 12 if hard else 8
    cfg = DatasetConfig(
        seed=seed, duration=duration,
        num_spoofing_episodes=n, num_wash_episodes=n, num_layering_episodes=n,
        injection=inj,
    )
    result = generate_dataset(cfg)
    # 캐시 친화적으로 이벤트를 DataFrame으로 변환
    ev = pd.DataFrame([{
        "ts": e.ts, "event_type": e.event_type.value, "account_id": e.account_id,
        "side": e.side.value, "price": e.price, "quantity": e.quantity,
    } for e in result.events])
    features = build_features(result.events)
    truth = build_truth(result.events, result.labels)
    return ev, features, truth


def main() -> None:
    st.title("이상거래 탐지(시장감시) 엔진")
    st.caption(
        "⚠️ 100% 합성 데이터입니다. 실제 종목·거래소·투자자와 무관하며, "
        "시세조종 탐지의 핵심 원리를 구현한 교육·포트폴리오 목적의 데모입니다."
    )

    sb = st.sidebar
    sb.header("설정")
    seed = sb.number_input("시드(seed)", value=42, step=1)
    duration = sb.select_slider("타임라인 길이(틱)", options=[10_000, 20_000, 30_000, 40_000], value=30_000)
    hard = sb.toggle("위장(camouflage) 난이도 모드", value=True,
                     help="조작 계좌가 정상 거래를 섞고 강도를 약화 → 고정 룰의 재현율이 떨어진다.")
    sb.divider()
    sb.subheader("룰 임계값")
    rule_cfg = RuleConfig(
        cancel_ratio_threshold=sb.slider("취소율 임계값", 0.0, 1.0, 0.7, 0.05),
        qty_zscore_threshold=sb.slider("주문량 z-score 임계값(스푸핑)", 0.0, 10.0, 5.0, 0.5),
        distinct_levels_threshold=sb.slider("distinct 레벨 수 임계값(레이어링)", 2, 15, 5, 1),
        self_trade_threshold=sb.slider("자기체결 수 임계값(워시)", 1, 10, 3, 1),
    )

    ev, features, truth = _generate(int(seed), int(duration), bool(hard))
    window_ms = DEFAULT_WINDOW_MS

    # ── 탐지(룰) ──
    rule_pred = predict(features, rule_cfg)
    rule_rep = evaluate(rule_pred, truth)

    # ── ML(시간 분할) ──
    labeled = build_labeled_frame(features, truth)
    train, test = time_split(labeled, train_fraction=0.7)
    ml = train_and_score(train, test)
    ml_m = binary_metrics(ml.test["is_manip"], ml.test["ml_pred"])
    ml_summary = curve_summary(ml.test["is_manip"], ml.test["score"])

    # ── 상단 지표 ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("전체 이벤트", f"{len(ev):,}")
    c2.metric("조작 윈도우(정답)", f"{rule_rep.n_positives}",
              f"{rule_rep.n_positives / rule_rep.n_samples:.2%} (불균형)")
    c3.metric("룰 재현율 / 정밀도", f"{rule_rep.recall:.2f} / {rule_rep.precision:.2f}")
    c4.metric("ML 재현율 / 정밀도", f"{ml_m.recall:.2f} / {ml_m.precision:.2f}")

    tab_flow, tab_alert, tab_eval, tab_ml = st.tabs(
        ["📈 주문 흐름", "🚨 탐지 알림", "📊 평가 지표", "🤖 룰 vs ML"]
    )

    # ── 주문 흐름 ──
    with tab_flow:
        ev = ev.assign(window=ev["ts"] // window_ms)
        trades = ev[ev["event_type"] == "TRADE"]
        price_by_w = trades.groupby("window")["price"].last()
        manip_windows = set(truth["window"].unique())

        fig, ax = plt.subplots(figsize=(11, 3.5))
        ax.plot(price_by_w.index, price_by_w.values, linewidth=1, label="체결가(윈도우별 마지막)")
        for w in manip_windows:
            ax.axvspan(w - 0.5, w + 0.5, color="red", alpha=0.12)
        ax.set_xlabel("window"); ax.set_ylabel("price(tick)")
        ax.set_title("체결가 추이 — 빨간 구간 = 조작 주입(ground truth)")
        ax.legend(loc="upper right")
        st.pyplot(fig)
        plt.close(fig)

        counts = ev.groupby(["window", "event_type"]).size().unstack(fill_value=0)
        st.caption("윈도우별 이벤트 수 (NEW / CANCEL / TRADE)")
        st.bar_chart(counts)

    # ── 탐지 알림 ──
    with tab_alert:
        merged = rule_pred.merge(truth, on=["account_id", "window"], how="left")
        merged["true_label"] = merged["true_label"].fillna("NORMAL")
        alerts = merged[merged["is_alert"]].copy()
        alerts["결과"] = alerts.apply(
            lambda r: "✅ TP" if r["true_label"] != "NORMAL" else "❌ FP(오탐)", axis=1
        )
        st.caption(f"룰이 발생시킨 알림 {len(alerts)}건 (정답 라벨과 대조)")
        st.dataframe(
            alerts[["account_id", "window", "predicted_label", "true_label", "결과",
                    "cancel_ratio", "distinct_price_levels", "self_trade_count",
                    "qty_zscore", "num_trade"]]
            .sort_values("window").reset_index(drop=True),
            width="stretch", height=380,
        )
        missed = merged[(merged["true_label"] != "NORMAL") & (~merged["is_alert"])]
        if len(missed):
            st.warning(f"놓친 조작(미탐, FN) {len(missed)}건 — 위장 모드에서 고정 룰의 한계.")

    # ── 평가 지표 ──
    with tab_eval:
        st.caption("왜 accuracy가 아니라 정밀도/재현율인가: 조작은 극소수(불균형)라 "
                   "'전부 정상'만 외쳐도 accuracy는 높지만 조작은 못 잡는다.")
        cm = pd.DataFrame(
            [[rule_rep.tp, rule_rep.fn], [rule_rep.fp, rule_rep.tn]],
            index=["실제:조작", "실제:정상"], columns=["예측:조작", "예측:정상"],
        )
        cc1, cc2 = st.columns([1, 1])
        cc1.write("**룰 기반 혼동행렬**"); cc1.table(cm)
        pc = pd.DataFrame(rule_rep.per_class).T[["precision", "recall", "support"]]
        cc2.write("**패턴별 정밀도/재현율 (룰)**"); cc2.table(pc.round(3))

    # ── 룰 vs ML + 커브 ──
    with tab_ml:
        st.caption("시간 분할(과거로 학습 → 미래 탐지)로 lookahead 누설을 차단한 비교.")
        comp = pd.DataFrame({
            "정밀도": [rule_rep.precision, ml_m.precision],
            "재현율": [rule_rep.recall, ml_m.recall],
            "F1": [rule_rep.f1, ml_m.f1],
        }, index=["룰 기반", "ML"]).round(3)
        st.table(comp)
        st.write(f"ML ROC AUC = **{ml_summary['roc_auc']:.3f}**, "
                 f"평균정밀도(PR AUC) = **{ml_summary['average_precision']:.3f}**")

        if test["is_manip"].nunique() > 1:
            paths = save_curves(ml.test["is_manip"], ml.test["score"], outdir="artifacts")
            i1, i2, i3 = st.columns(3)
            i1.image(paths["roc"]); i2.image(paths["pr"]); i3.image(paths["tradeoff"])

        st.write("**피처 중요도**")
        st.bar_chart(feature_importances(ml.model))


if __name__ == "__main__":
    main()
