from __future__ import annotations

from collections import Counter, deque


class MajorityVoteSmoother:
    def __init__(self, maxlen: int = 8) -> None:
        self._buf: deque[str] = deque(maxlen=maxlen)

    def push(self, label: str) -> str:
        self._buf.append(label)
        counts = Counter(self._buf)
        return counts.most_common(1)[0][0]

    def set_maxlen(self, maxlen: int) -> None:
        if maxlen <= 0:
            return
        if self._buf.maxlen == maxlen:
            return
        old = list(self._buf)
        self._buf = deque(old[-maxlen:], maxlen=maxlen)
