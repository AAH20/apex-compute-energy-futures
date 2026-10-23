"""Core domain models and typed data structures for Apex Compute-Energy Futures."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class SLATier(str, Enum):
    BEST_EFFORT = "BEST_EFFORT"      # Retail spot, preemptible with zero grace
    FAULT_TOLERANT = "FAULT_TOLERANT"  # Batch jobs, checkpointing required
    MISSION_CRITICAL = "MISSION_CRITICAL"  # Enterprise inference / training, 99.999% SLA


class MarketVenue(str, Enum):
    VAST_AI = "vast.ai"
    RUNPOD = "runpod"
    AWS_SPOT = "aws_spot"
    COREWEAVE = "coreweave"
    NEO_CLOUD_PRIVATE = "neo_cloud_private"


@dataclass(frozen=True)
class ComputeContract:
    """Standardized forward compute contract definition."""
    contract_id: str
    region: str
    node_type: str                   # e.g., "H100-SXM5-80GB"
    gpu_count: int
    duration_hours: float
    forward_price_per_gpu_hr: float
    sla_tier: SLATier
    created_at_epoch: float
    expiration_epoch: float


@dataclass(frozen=True)
class EnergySlot:
    """Locational marginal electricity pricing and grid stability slot."""
    region: str                      # e.g., "ERCOT_HOUSTON"
    timestamp_epoch: float
    lmp_price_per_mwh: float         # Locational Marginal Price ($/MWh)
    grid_frequency_hz: float         # Target: 60.0 Hz
    carbon_intensity_g_per_kwh: float
    rocof_hz_per_sec: float = 0.0    # Rate of Change of Frequency


@dataclass(frozen=True)
class ArbitrageQuote:
    """Real-time arbitrage opportunity between spot order book and forward curve."""
    opportunity_id: str
    venue: MarketVenue
    node_type: str
    gpu_count: int
    spot_ask_rate: float            # Spot ask price ($/GPU-hr)
    forward_clearance_price: float   # Clearance threshold ($/GPU-hr)
    gross_margin_bps: float          # Margin in basis points
    estimated_slippage_bps: float
    net_edge_bps: float
    latency_microseconds: float


@dataclass
class ActionGateDecision:
    """Deterministic fast-path risk and gating evaluation output."""
    approved: bool
    reason: str
    opportunity_id: str
    evaluated_latency_us: float
    target_host_ip: str
    allocated_quota_usd: float
    formal_invariants_satisfied: bool
    audit_hash: str


@dataclass
class HostNode:
    """Physical or virtual GPU host instance telemetry."""
    host_id: str
    provider: MarketVenue
    ip_address: str
    gpu_model: str
    gpu_count: int
    ping_latency_ms: float
    packet_loss_pct: float
    jitter_ms: float
    is_preemptible: bool
    last_health_check_epoch: float
    is_healthy: bool = True
