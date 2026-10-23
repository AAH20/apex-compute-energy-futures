"""Dynamic On-Demand Integration Registry for Apex Sovereign Ecosystem Repositories.

Allows apex-compute-energy-futures to detect, bind, and orchestrate external
peer kernels and repos dynamically without hard compile-time dependencies.
"""

import importlib
import logging
import os
import sys
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("apex_compute_futures.integrations")


class EcosystemIntegrationRegistry:
    """Discovers, verifies, and provides lazy-loaded adapters to ecosystem projects."""

    _ADAPTERS: Dict[str, Dict[str, Any]] = {
        "datacenter_np_hard_kernel": {
            "name": "Data Center NP-Hard Optimization Kernel",
            "repo_url": "https://github.com/AAH20/datacenter-np-hard-kernel",
            "description": "Multi-dimensional vector bin packing, OCS All-Reduce topology, and BGP fast reroute.",
            "adapter_module": "apex_compute_futures.integrations.dcc_adapter"
        },
        "gigawatt_ride_through_amm_kernel": {
            "name": "Gigawatt Ride-Through AMM Kernel",
            "repo_url": "https://github.com/AAH20/gigawatt-ride-through-amm-kernel",
            "description": "Multi-tier Ragone knapsack and sub-microsecond DVFS GPU compute shedding.",
            "adapter_module": "apex_compute_futures.integrations.gigawatt_adapter"
        },
        "hft_microstructure_kernel": {
            "name": "Quantitative HFT & Market Microstructure Kernel",
            "repo_url": "https://github.com/AAH20/hft-microstructure-kernel",
            "description": "471k orders/s LOB, Bellman-Ford negative cycle arbitrage, and Almgren-Chriss liquidation.",
            "adapter_module": "apex_compute_futures.integrations.hft_adapter"
        },
        "apex_quant_whale_kernel": {
            "name": "Apex Quant Whale Execution Kernel",
            "repo_url": "https://github.com/AAH20/apex-quant-whale-kernel",
            "description": "Hawkes process toxic flow defense, relativistic skywave routing, and dark pool knapsack.",
            "adapter_module": "apex_compute_futures.integrations.quant_whale_adapter"
        },
        "agentic_fintech_kernel": {
            "name": "Agentic FinTech & Formal Settlement OS",
            "repo_url": "https://github.com/AAH20/agentic-fintech-kernel",
            "description": "SMT/Z3 mathematical invariant verification, ZK PCI DSS 4.0 escrow, and multi-rail settlement.",
            "adapter_module": "apex_compute_futures.integrations.formal_guard_adapter"
        },
        "Apex_FDE_Matrix": {
            "name": "Apex FDE Matrix & Infrastructure Digital Twin",
            "repo_url": "https://github.com/AAH20/Apex_FDE_Matrix",
            "description": "0.12ms Causal Digital Twin, bounded blast-radius mutations, and native MCP server.",
            "adapter_module": "apex_compute_futures.integrations.fde_matrix_adapter"
        },
        "intraday_liquidity_orchestrator": {
            "name": "Intraday Liquidity Orchestrator",
            "repo_url": "https://github.com/AAH20/intraday-liquidity-orchestrator",
            "description": "24/7/365 real-time clearing rail treasury sweeper and ISO 20022 camt.050 engine.",
            "adapter_module": "apex_compute_futures.integrations.liquidity_sweeper_adapter"
        }
    }

    @classmethod
    def list_available_integrations(cls) -> Dict[str, Dict[str, str]]:
        """Returns metadata for all available on-demand ecosystem integration modules."""
        return {
            k: {
                "name": v["name"],
                "repo_url": v["repo_url"],
                "description": v["description"]
            }
            for k, v in cls._ADAPTERS.items()
        }

    @classmethod
    def load_adapter(cls, kernel_key: str) -> Optional[Any]:
        """Dynamically imports and returns the specialized adapter for the target kernel."""
        info = cls._ADAPTERS.get(kernel_key)
        if not info:
            logger.warning(f"Kernel '{kernel_key}' not found in ecosystem registry.")
            return None

        try:
            mod = importlib.import_module(info["adapter_module"])
            adapter_cls = getattr(mod, f"{kernel_key.title().replace('_', '')}Adapter", None)
            if adapter_cls:
                return adapter_cls()
            return mod
        except Exception as e:
            logger.info(f"Optional adapter for {kernel_key} operating in decoupled mock mode: {e}")
            return None
