from __future__ import annotations

import argparse

from dotenv import load_dotenv

from app.cli_eval import run_eval
from app.cli_interactive import run_interactive
from app.cli_common import _parse_bool
from app.cli_tune import run_tune

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="eRisk 2026 Conversational Depression Detection PoC")
    parser.add_argument("--mode", choices=["interactive", "eval", "tune"], default="interactive")
    parser.add_argument("--personas", type=int, default=10, help="Number of synthetic personas")
    parser.add_argument("--seed", type=int, default=42, help="Seed for synthetic persona generation")
    parser.add_argument(
        "--eval_mode",
        choices=["mixed_holdout", "synthetic_only"],
        default="mixed_holdout",
    )
    parser.add_argument("--prompt_version", default="v1")
    parser.add_argument("--save_diagnostics", default="true")
    parser.add_argument("--max_api_calls", type=int, default=180)
    parser.add_argument("--trace_level", choices=["compact", "off"], default="compact")
    parser.add_argument("--fit_calibrator", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--tune_personas", type=int, default=30)
    parser.add_argument("--tune_seed", type=int, default=42)
    parser.add_argument("--tune_max_api_calls", type=int, default=800)
    parser.add_argument("--tune_save_diagnostics", default="false")
    parser.add_argument("--tune_trace_level", choices=["compact", "off"], default="off")
    parser.add_argument("--tune_prompt_version", default="")
    parser.add_argument("--tune_top_k", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "interactive":
        run_interactive()
    elif args.mode == "eval":
        run_eval(
            persona_count=args.personas,
            seed=args.seed,
            eval_mode=args.eval_mode,
            prompt_version=args.prompt_version,
            save_diagnostics=_parse_bool(args.save_diagnostics),
            max_api_calls=args.max_api_calls,
            trace_level=args.trace_level,
            fit_calibrator_policy=args.fit_calibrator,
        )
    else:
        tune_prompt_version = args.tune_prompt_version.strip() or args.prompt_version
        run_tune(
            tune_personas=args.tune_personas,
            tune_seed=args.tune_seed,
            tune_max_api_calls=args.tune_max_api_calls,
            tune_save_diagnostics=_parse_bool(args.tune_save_diagnostics),
            tune_trace_level=args.tune_trace_level,
            tune_prompt_version=tune_prompt_version,
            tune_top_k=args.tune_top_k,
            fit_calibrator_policy=args.fit_calibrator,
        )


if __name__ == "__main__":
    main()
