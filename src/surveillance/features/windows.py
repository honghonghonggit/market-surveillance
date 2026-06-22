"""시간 윈도우 헬퍼.

피처 집계와 ground truth 집계가 *동일한* 윈도우 정의를 공유해야 예측-정답 단위가
정확히 일치한다. 그래서 윈도우 함수를 한 곳에 둔다.
"""

from __future__ import annotations

# 평가의 기본 단위인 텀블링 윈도우 길이(ms). 라벨 집계도 이 값을 공유한다.
DEFAULT_WINDOW_MS = 500


def window_index(ts: int, window_ms: int = DEFAULT_WINDOW_MS) -> int:
    """타임스탬프(ms)를 고정 길이 텀블링 윈도우 인덱스로 변환."""
    return ts // window_ms
