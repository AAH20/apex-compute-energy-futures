"""High-Throughput, Zero-Allocation Circular Ring Buffer Tick Cache.

Engineered for sub-microsecond price tick caches and eBPF kernel telemetry streams:
- Pre-allocated memory slots (zero heap re-allocation during operation)
- Circular modular pointer arithmetic (O(1) push and indexed lookup)
- O(1) rolling mean, variance, standard deviation, and Exponential Moving Average (EMA)
- Windowed statistics computed without list slice allocations
- Dual layout: ScalarRingBuffer for continuous floats and TelemetryRingBuffer for eBPF ticks
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterator, List, Optional, Tuple


class RingBuffer:
    """Pre-allocated circular ring buffer for sub-microsecond continuous metric feeds.
    
    Provides O(1) push, O(1) indexing, and O(1) running statistics (mean, variance, EMA)
    without triggering heap allocations or garbage collection passes.
    """

    __slots__ = (
        "_capacity",
        "_buffer",
        "_head",
        "_count",
        "_is_full",
        "_sum",
        "_sum_sq",
        "_ema",
        "_ema_alpha",
    )

    def __init__(self, capacity: int, ema_alpha: Optional[float] = None) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
            
        self._capacity: int = int(capacity)
        # Pre-allocate contiguous array of floats
        self._buffer: List[float] = [0.0] * self._capacity
        self._head: int = 0
        self._count: int = 0
        self._is_full: bool = False
        
        # Incremental statistics accumulators
        self._sum: float = 0.0
        self._sum_sq: float = 0.0
        self._ema: float = 0.0
        # Default alpha = 2 / (capacity + 1)
        self._ema_alpha: float = (
            float(ema_alpha) if ema_alpha is not None else (2.0 / (self._capacity + 1.0))
        )

    @property
    def capacity(self) -> int:
        """Maximum number of elements before overwriting oldest."""
        return self._capacity

    @property
    def is_full(self) -> bool:
        """Returns True if buffer has filled its capacity."""
        return self._is_full

    @property
    def total_pushed(self) -> int:
        """Monotonically increasing count of total elements inserted."""
        return self._count

    def __len__(self) -> int:
        """Current number of active valid elements in the buffer."""
        return self._capacity if self._is_full else self._count

    def append(self, value: float) -> None:
        """Appends value in O(1) time and updates rolling accumulators with zero allocations."""
        val = float(value)
        val_sq = val * val
        head = self._head

        if self._is_full:
            old_val = self._buffer[head]
            self._sum += (val - old_val)
            self._sum_sq += (val_sq - old_val * old_val)
        else:
            self._sum += val
            self._sum_sq += val_sq
            if self._count + 1 >= self._capacity:
                self._is_full = True

        self._buffer[head] = val
        self._head = (head + 1) % self._capacity
        self._count += 1

        # Exponential Moving Average update
        if self._count == 1:
            self._ema = val
        else:
            self._ema = self._ema_alpha * val + (1.0 - self._ema_alpha) * self._ema

    # Alias for HFT nomenclature
    push = append

    def extend(self, values: Any) -> None:
        """Appends multiple values consecutively."""
        for v in values:
            self.append(v)

    def mean(self) -> float:
        """Computes current rolling mean in O(1) time."""
        size = len(self)
        if size == 0:
            return 0.0
        return self._sum / size

    def variance(self, ddof: int = 0) -> float:
        """Computes current rolling variance in O(1) time.
        
        Args:
            ddof: Delta Degrees of Freedom (0 for population variance, 1 for sample).
        """
        size = len(self)
        if size <= ddof or size == 0:
            return 0.0
            
        mu = self._sum / size
        raw_var = (self._sum_sq / size) - (mu * mu)
        if raw_var < 0.0:
            raw_var = 0.0  # Guard against floating point epsilon cancellation
            
        if ddof == 1:
            raw_var = raw_var * size / (size - 1)
            
        return raw_var

    def std(self, ddof: int = 0) -> float:
        """Computes current rolling standard deviation in O(1) time."""
        return math.sqrt(self.variance(ddof=ddof))

    def ema(self) -> float:
        """Returns current exponential moving average in O(1) time."""
        return self._ema

    def window_mean(self, window_size: int) -> float:
        """Calculates rolling mean over the last W items in O(W) without list slicing."""
        size = len(self)
        if size == 0 or window_size <= 0:
            return 0.0
            
        w = min(window_size, size)
        if w == size:
            return self.mean()
            
        acc = 0.0
        head = self._head
        cap = self._capacity
        buf = self._buffer
        for i in range(1, w + 1):
            idx = (head - i) % cap
            acc += buf[idx]
        return acc / w

    def window_variance(self, window_size: int, ddof: int = 0) -> float:
        """Calculates rolling variance over the last W items without list slicing."""
        size = len(self)
        if size == 0 or window_size <= 0:
            return 0.0
            
        w = min(window_size, size)
        if w <= ddof:
            return 0.0
        if w == size:
            return self.variance(ddof=ddof)
            
        acc_s = 0.0
        acc_sq = 0.0
        head = self._head
        cap = self._capacity
        buf = self._buffer
        for i in range(1, w + 1):
            idx = (head - i) % cap
            v = buf[idx]
            acc_s += v
            acc_sq += v * v
            
        mu = acc_s / w
        var = (acc_sq / w) - (mu * mu)
        if var < 0.0:
            var = 0.0
        if ddof == 1:
            var = var * w / (w - 1)
        return var

    def window_std(self, window_size: int, ddof: int = 0) -> float:
        """Calculates rolling standard deviation over the last W items without list slicing."""
        return math.sqrt(self.window_variance(window_size, ddof=ddof))

    def latest(self) -> float:
        """Returns the most recently pushed value in O(1) time."""
        if self._count == 0:
            raise IndexError("RingBuffer is empty.")
        idx = (self._head - 1) % self._capacity
        return self._buffer[idx]

    def oldest(self) -> float:
        """Returns the oldest active value currently retained in O(1) time."""
        size = len(self)
        if size == 0:
            raise IndexError("RingBuffer is empty.")
        idx = (self._head - size) % self._capacity
        return self._buffer[idx]

    def __getitem__(self, index: int) -> float:
        """Retrieves item by logical index (0 = oldest, -1 = newest)."""
        size = len(self)
        if size == 0:
            raise IndexError("RingBuffer is empty.")
            
        if index < 0:
            if index < -size:
                raise IndexError(f"Index {index} out of range for buffer size {size}.")
            # -1 refers to head - 1, -size refers to head - size
            phys_idx = (self._head + index) % self._capacity
            return self._buffer[phys_idx]
            
        if index >= size:
            raise IndexError(f"Index {index} out of range for buffer size {size}.")
            
        # 0 refers to oldest = (head - size) % capacity
        phys_idx = (self._head - size + index) % self._capacity
        return self._buffer[phys_idx]

    def __iter__(self) -> Iterator[float]:
        """Iterates from oldest to newest without allocating a temporary list."""
        size = len(self)
        if size == 0:
            return
        head = self._head
        cap = self._capacity
        buf = self._buffer
        start_idx = (head - size) % cap
        for i in range(size):
            yield buf[(start_idx + i) % cap]

    def to_list(self) -> List[float]:
        """Copies active buffer contents into a list ordered from oldest to newest."""
        return list(self)

    def clear(self) -> None:
        """Resets pointers and accumulators without re-allocating buffer memory."""
        self._head = 0
        self._count = 0
        self._is_full = False
        self._sum = 0.0
        self._sum_sq = 0.0
        self._ema = 0.0


@dataclass(frozen=True)
class TelemetryRecord:
    """eBPF kernel telemetry snapshot for an AI server rack."""
    timestamp_ns: int
    p_compute_watts: float
    p_overhead_watts: float
    t_die_c: float
    clock_ghz: float
    queue_depth: int


class TelemetryRingBuffer:
    """Structure-of-Arrays (SoA) circular telemetry ring buffer for eBPF kernel streams.
    
    Maintains parallel pre-allocated channels to avoid Python object creation overhead
    during high-frequency telemetry ingestion.
    """

    __slots__ = (
        "_capacity",
        "_head",
        "_count",
        "_is_full",
        "_ts",
        "_p_comp",
        "_p_over",
        "_t_die",
        "_clock",
        "_queue",
        "_sum_p_comp",
        "_sum_t_die",
    )

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be positive.")
        self._capacity: int = int(capacity)
        self._head: int = 0
        self._count: int = 0
        self._is_full: bool = False
        
        # Parallel pre-allocated primitive arrays
        self._ts: List[int] = [0] * self._capacity
        self._p_comp: List[float] = [0.0] * self._capacity
        self._p_over: List[float] = [0.0] * self._capacity
        self._t_die: List[float] = [0.0] * self._capacity
        self._clock: List[float] = [0.0] * self._capacity
        self._queue: List[int] = [0] * self._capacity
        
        # Running sums for instant O(1) thermal/power checks
        self._sum_p_comp: float = 0.0
        self._sum_t_die: float = 0.0

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return self._capacity if self._is_full else self._count

    def push(
        self,
        timestamp_ns: int,
        p_compute_watts: float,
        p_overhead_watts: float,
        t_die_c: float,
        clock_ghz: float,
        queue_depth: int = 0
    ) -> None:
        """Ingests eBPF telemetry tick in O(1) zero-allocation time."""
        head = self._head
        if self._is_full:
            old_p = self._p_comp[head]
            old_t = self._t_die[head]
            self._sum_p_comp += (p_compute_watts - old_p)
            self._sum_t_die += (t_die_c - old_t)
        else:
            self._sum_p_comp += p_compute_watts
            self._sum_t_die += t_die_c
            if self._count + 1 >= self._capacity:
                self._is_full = True

        self._ts[head] = timestamp_ns
        self._p_comp[head] = p_compute_watts
        self._p_over[head] = p_overhead_watts
        self._t_die[head] = t_die_c
        self._clock[head] = clock_ghz
        self._queue[head] = queue_depth

        self._head = (head + 1) % self._capacity
        self._count += 1

    def average_compute_power(self) -> float:
        """Returns mean compute power across active window in O(1) time."""
        size = len(self)
        return (self._sum_p_comp / size) if size > 0 else 0.0

    def average_die_temp(self) -> float:
        """Returns mean silicon die temperature across active window in O(1) time."""
        size = len(self)
        return (self._sum_t_die / size) if size > 0 else 0.0

    def latest(self) -> TelemetryRecord:
        """Returns the most recent telemetry tick."""
        if self._count == 0:
            raise IndexError("TelemetryRingBuffer is empty.")
        idx = (self._head - 1) % self._capacity
        return TelemetryRecord(
            timestamp_ns=self._ts[idx],
            p_compute_watts=self._p_comp[idx],
            p_overhead_watts=self._p_over[idx],
            t_die_c=self._t_die[idx],
            clock_ghz=self._clock[idx],
            queue_depth=self._queue[idx]
        )
