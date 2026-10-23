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
from apex_compute_futures.core.ring_buffer import (
    RingBuffer,
    TelemetryRecord,
    TelemetryRingBuffer,
)
from apex_compute_futures.core.settlement_prover import (
    ASTNode,
    Add,
    And,
    AuditBlock,
    Constant,
    CryptographicAuditLedger,
    Div,
    Ed25519Signer,
    Eq,
    FormalSettlementProver,
    Ge,
    Le,
    MerkleTree,
    MerkleTreeNode,
    Mul,
    Not,
    ProofCertificate,
    ProofStatus,
    SMTConstraintSolver,
    Sub,
    Symbol,
)
from apex_compute_futures.evolution.chromosome import AgentChromosome, ChromosomeEngine
from apex_compute_futures.evolution.fitness import SystemFitnessCalculator
from apex_compute_futures.evolution.pareto_tournament import (
    ArchetypeIndividual,
    MultiObjectiveFitness,
    ParetoEvolutionEngine,
)
from apex_compute_futures.futures.almgren_chriss import (
    AlmgrenChrissConfig,
    AlmgrenChrissEngine,
    OptimalExecutionSchedule,
    TrajectoryStep,
)
from apex_compute_futures.futures.arbitrage_engine import ComputeArbitrageEngine
from apex_compute_futures.futures.forward_curve import ComputeForwardCurve, ForwardCurvePoint
from apex_compute_futures.futures.hjb_storage_solver import (
    BatteryOperatingMode,
    ComputeThrottlingMode,
    HJBBatteryConfig,
    HJBPolicyAction,
    HJBState,
    HJBStorageSolver,
)
from apex_compute_futures.futures.smart_order_router import (
    ComputeVenue,
    LiquidityKnapsackSolver,
    RoutingSolution,
    SmartOrderRouter,
    VenueAllocation,
    VenueTier,
)
from apex_compute_futures.futures.synthetic_inertia_amm import SyntheticInertiaAMM
from apex_compute_futures.futures.thermal_pue import (
    PUEBreakdown,
    ThermalConfig,
    ThermalHeadroom,
    ThermalPUEModel,
)
from apex_compute_futures.integrations.registry import EcosystemIntegrationRegistry
from apex_compute_futures.predictive.adaptive_kalman import (
    AdaptiveKalmanConfig,
    DualAdaptiveKalmanFilter,
    KalmanEstimate,
)
from apex_compute_futures.predictive.fractional_diff import FractionalDifferentiator
from apex_compute_futures.predictive.hawkes_process import (
    HawkesParameters,
    HawkesProcess,
    PreemptionShockEvent,
)
from apex_compute_futures.predictive.information_bars import CusumInformationSampler, InformationBar
from apex_compute_futures.predictive.ou_jump_diffusion import (
    OUJumpDiffusionModel,
    OUParameters,
    SpreadSignal,
)
from apex_compute_futures.predictive.regime_hmm import (
    ContinuousTimeHMM,
    HamiltonFilter,
    HamiltonFilterState,
    MarketRegime,
    RegimeState,
)
from apex_compute_futures.predictive.triple_barrier import BarrierEvent, TripleBarrierLabeler
from apex_compute_futures.simulation.stress_tester import (
    InstitutionalStressTestingEngine,
    PathTrajectory,
    StressScenario,
    StressScenarioType,
    StressTestResult,
)

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
    "RingBuffer",
    "TelemetryRingBuffer",
    "TelemetryRecord",
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
    "ArchetypeIndividual",
    "MultiObjectiveFitness",
    "ParetoEvolutionEngine",
    "EcosystemIntegrationRegistry",
    "ThermalPUEModel",
    "ThermalConfig",
    "ThermalHeadroom",
    "PUEBreakdown",
    "AlmgrenChrissEngine",
    "AlmgrenChrissConfig",
    "OptimalExecutionSchedule",
    "TrajectoryStep",
    "HJBStorageSolver",
    "HJBBatteryConfig",
    "HJBState",
    "HJBPolicyAction",
    "BatteryOperatingMode",
    "ComputeThrottlingMode",
    "SmartOrderRouter",
    "ComputeVenue",
    "VenueTier",
    "VenueAllocation",
    "RoutingSolution",
    "LiquidityKnapsackSolver",
    "OUJumpDiffusionModel",
    "OUParameters",
    "SpreadSignal",
    "ContinuousTimeHMM",
    "HamiltonFilter",
    "HamiltonFilterState",
    "MarketRegime",
    "RegimeState",
    "DualAdaptiveKalmanFilter",
    "AdaptiveKalmanConfig",
    "KalmanEstimate",
    "HawkesProcess",
    "HawkesParameters",
    "PreemptionShockEvent",
    "InstitutionalStressTestingEngine",
    "StressScenario",
    "StressScenarioType",
    "StressTestResult",
    "PathTrajectory",
    "FormalSettlementProver",
    "SMTConstraintSolver",
    "ProofCertificate",
    "ProofStatus",
    "MerkleTree",
    "MerkleTreeNode",
    "CryptographicAuditLedger",
    "AuditBlock",
    "Ed25519Signer",
    "ASTNode",
    "Symbol",
    "Constant",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Le",
    "Ge",
    "Eq",
    "And",
    "Not",
]
