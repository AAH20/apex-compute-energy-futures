"""Apex Compute-Energy Futures, Predictive ML, and Microsecond Arbitrage Engine."""

from apex_compute_futures.core.action_gate import ActionGate
from apex_compute_futures.core.invariants import FormalInvariantVerifier
from apex_compute_futures.core.models import (
    ActionGateDecision,
    ArbitrageQuote,
    ComputeContract,
    EnergySlot,
    HostNode,
    MarketVenue,
    SLATier,
)
from apex_compute_futures.evolution.chromosome import AgentChromosome, ChromosomeEngine
from apex_compute_futures.evolution.fitness import SystemFitnessCalculator
from apex_compute_futures.futures.arbitrage_engine import ComputeArbitrageEngine
from apex_compute_futures.futures.forward_curve import ComputeForwardCurve, ForwardCurvePoint
from apex_compute_futures.futures.synthetic_inertia_amm import SyntheticInertiaAMM
from apex_compute_futures.integrations.registry import EcosystemIntegrationRegistry
from apex_compute_futures.predictive.fractional_diff import FractionalDifferentiator
from apex_compute_futures.predictive.information_bars import CusumInformationSampler, InformationBar
from apex_compute_futures.predictive.triple_barrier import BarrierEvent, TripleBarrierLabeler

__version__ = "0.1.0"
__all__ = [
    "ActionGate",
    "FormalInvariantVerifier",
    "ComputeContract",
    "EnergySlot",
    "ArbitrageQuote",
    "ActionGateDecision",
    "HostNode",
    "SLATier",
    "MarketVenue",
    "FractionalDifferentiator",
    "CusumInformationSampler",
    "InformationBar",
    "TripleBarrierLabeler",
    "BarrierEvent",
    "ComputeForwardCurve",
    "ForwardCurvePoint",
    "SyntheticInertiaAMM",
    "ComputeArbitrageEngine",
    "SystemFitnessCalculator",
    "AgentChromosome",
    "ChromosomeEngine",
    "EcosystemIntegrationRegistry",
]
