"""Command Line Interface for Apex Compute-Energy Futures Platform."""

import argparse
import sys
from apex_compute_futures.futures.forward_curve import ComputeForwardCurve
from apex_compute_futures.futures.synthetic_inertia_amm import SyntheticInertiaAMM
from apex_compute_futures.integrations.registry import EcosystemIntegrationRegistry


def main():
    parser = argparse.ArgumentParser(
        description="Apex Compute-Energy Futures, Predictive ML & Microsecond Arbitrage Engine"
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available commands")

    subparsers.add_parser("demo", help="Run interactive master demonstration harness")
    subparsers.add_parser("integrations", help="List available ecosystem integrations")

    curve_parser = subparsers.add_parser("forward-curve", help="Calculate forward compute yield curve")
    curve_parser.add_argument("--spot", type=float, default=2.45, help="Base spot rate ($/GPU-hr)")
    curve_parser.add_argument("--region", type=str, default="US_EAST_VA", help="Target region")

    amm_parser = subparsers.add_parser("inertia-amm", help="Simulate Synthetic Inertia AMM response")
    amm_parser.add_argument("--freq", type=float, default=59.82, help="Observed grid frequency (Hz)")

    args = parser.parse_args()

    if args.subcommand == "demo" or args.subcommand is None:
        # Import and run master demo
        import demo
        demo.run_demonstration()
    elif args.subcommand == "integrations":
        print("\n=== APEX ECOSYSTEM ON-DEMAND INTEGRATIONS ===")
        integrations = EcosystemIntegrationRegistry.list_available_integrations()
        for k, v in integrations.items():
            print(f"• [{k}] {v['name']}")
            print(f"  URL  : {v['repo_url']}")
            print(f"  Role : {v['description']}\n")
    elif args.subcommand == "forward-curve":
        curve = ComputeForwardCurve(region=args.region, base_spot_rate=args.spot)
        print(f"\nForward Term Structure for {args.region} (Spot: ${args.spot:.2f}/GPU-hr):")
        print(f"{'Maturity':<12} {'Forward Rate':<16} {'Carry Cost':<14} {'Basis':<12}")
        print("-" * 54)
        for pt in curve.generate_full_term_structure():
            print(f"{pt.maturity_hours:<5.0f} hrs      ${pt.forward_rate_per_gpu_hr:<15.4f} {pt.cost_of_carry_pct:<13.2f}% ${pt.basis_spread_usd:<11.4f}")
        print()
    elif args.subcommand == "inertia-amm":
        amm = SyntheticInertiaAMM()
        quote = amm.quote_frequency_response(args.freq)
        print(f"\nSynthetic Inertia AMM Quote for Frequency Trip: {args.freq} Hz (Delta: {quote.frequency_deviation_hz} Hz):")
        print(f"• Compute Power to Shed  : {quote.required_power_mw:.2f} MW ({quote.gpu_shed_fraction*100:.1f}% DVFS throttle)")
        print(f"• Synthesized Inertia    : {quote.synthesized_inertia_mw_s:.2f} MW-s")
        print(f"• Ancillary Value Earned : ${quote.clearing_revenue_per_mwh:,.2f} per 30s event\n")


if __name__ == "__main__":
    main()
