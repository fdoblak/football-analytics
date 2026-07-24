"""Self-contained acceptance package."""

from football_analytics.acceptance.self_contained.generator import generate_scenario
from football_analytics.acceptance.self_contained.release import (
    build_technical_preview_report,
    render_technical_preview_png,
    write_report,
)
from football_analytics.acceptance.self_contained.runner import (
    run_self_contained_acceptance,
    validate_two_run_determinism,
)
from football_analytics.acceptance.self_contained.scenario import ScenarioConfig

__all__ = [
    "ScenarioConfig",
    "build_technical_preview_report",
    "generate_scenario",
    "render_technical_preview_png",
    "run_self_contained_acceptance",
    "validate_two_run_determinism",
    "write_report",
]
