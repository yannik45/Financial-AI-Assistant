from collections.abc import Sequence
from dataclasses import dataclass

from financial_ai.schemas import (
    AnalyticsResponse,
    PortfolioRiskScore,
    RiskComponent,
    RiskDimension,
    RiskDriver,
)

METHODOLOGY_VERSION = "portfolio-risk-score-v2"


@dataclass(frozen=True)
class ComponentDefinition:
    key: str
    label: str
    weight: float


COMPONENTS = {
    "volatility": ComponentDefinition("volatility", "Historical portfolio volatility", 0.55),
    "drawdown": ComponentDefinition("drawdown", "Historical drawdown", 0.35),
    "asset_mix": ComponentDefinition("asset_mix", "Structural exposure", 0.10),
}

# The volatility anchors follow the historical UCITS SRRI bucket boundaries only as
# transparent reference points. This score is not an SRRI or PRIIPs risk class.
VOLATILITY_ANCHORS = (
    (0, 0),
    (0.5, 5),
    (2, 18),
    (5, 32),
    (10, 48),
    (15, 65),
    (25, 82),
    (40, 95),
    (60, 100),
)
DRAWDOWN_ANCHORS = ((0, 0), (5, 12), (10, 25), (20, 45), (30, 65), (50, 88), (70, 100))
LARGEST_POSITION_ANCHORS = (
    (0, 0),
    (5, 5),
    (10, 12),
    (20, 32),
    (30, 52),
    (50, 78),
    (75, 95),
    (100, 100),
)
HHI_ANCHORS = ((0, 0), (0.10, 10), (0.15, 25), (0.25, 50), (0.40, 75), (0.60, 90), (1, 100))
FX_ANCHORS = ((0, 0), (10, 10), (25, 30), (50, 60), (75, 85), (100, 100))
CASH_RISK_ANCHORS = ((-20, 100), (0, 100), (5, 80), (10, 60), (20, 30), (30, 10), (50, 0), (100, 0))


def calculate_risk_score(
    analytics: AnalyticsResponse,
    *,
    base_currency: str,
    cash_value_eur: float,
) -> PortfolioRiskScore:
    invested_value = float(analytics.market_value_eur)
    total_equity = invested_value + cash_value_eur
    denominator = total_equity if total_equity > 0 else max(invested_value, 1.0)
    cash_ratio_percent = cash_value_eur / denominator * 100

    volatility = analytics.annualized_volatility_percent
    volatility_score = interpolate(volatility, VOLATILITY_ANCHORS)
    drawdown = abs(min(analytics.max_drawdown_percent, 0))
    drawdown_score = interpolate(drawdown, DRAWDOWN_ANCHORS)

    largest_weight_percent = analytics.largest_position_weight * 100
    largest_score = interpolate(largest_weight_percent, LARGEST_POSITION_ANCHORS)
    hhi_score = interpolate(analytics.concentration_hhi, HHI_ANCHORS)
    largest_sector_label, largest_sector_percent = _largest_allocation(analytics, "sector")
    sector_score = interpolate(largest_sector_percent, LARGEST_POSITION_ANCHORS)
    raw_concentration_risk = 0.50 * largest_score + 0.30 * hhi_score + 0.20 * sector_score
    broad_market_percent = _allocation_weight(analytics, "sector", "broad market")
    fixed_income_percent = _allocation_weight(analytics, "sector", "fixed income")
    diversified_vehicle_percent = min(broad_market_percent + fixed_income_percent, 100)
    # A broad fund is legally one position but economically represents many underlying
    # holdings. Without full holdings data this transparent credit avoids treating it like
    # a single stock while remaining more conservative than true fund look-through.
    look_through_credit = min(diversified_vehicle_percent / 100 * 0.75, 0.75)
    concentration_risk = raw_concentration_risk * (1 - look_through_credit)
    diversification_quality = 100 - concentration_risk

    asset_mix_score, asset_details = _asset_mix_score(analytics)
    foreign_currency_percent = _foreign_currency_exposure(analytics, base_currency) * 100
    liquidity_resilience = 100 - interpolate(cash_ratio_percent, CASH_RISK_ANCHORS)

    components = [
        _component(
            "volatility",
            volatility_score,
            volatility,
            "% annualized",
            f"Annualized historical portfolio volatility is {volatility:.1f}%.",
        ),
        _component(
            "drawdown",
            drawdown_score,
            drawdown,
            "% loss",
            f"The reconstructed series' maximum drawdown is {drawdown:.1f}%.",
        ),
        _component(
            "asset_mix",
            asset_mix_score,
            asset_mix_score,
            "structural score",
            "The structural score applies explicit risk factors to current asset-class weights.",
            asset_details,
        ),
    ]
    total_score = round(sum(item.contribution for item in components), 1)
    drivers = [
        RiskDriver(
            component=item.key,
            contribution=item.contribution,
            explanation=item.summary,
        )
        for item in sorted(components, key=lambda value: value.contribution, reverse=True)[:3]
    ]
    return PortfolioRiskScore(
        score=total_score,
        level=_risk_level(total_score),
        methodology_version=METHODOLOGY_VERSION,
        as_of=analytics.as_of,
        components=components,
        main_drivers=drivers,
        diversification=RiskDimension(
            key="diversification",
            label="Diversification quality",
            score=round(diversification_quality, 1),
            level=_quality_level(diversification_quality),
            summary=(
                f"{analytics.largest_position_symbol} is {largest_weight_percent:.1f}% of "
                f"invested assets; the largest sector is {largest_sector_label} at "
                f"{largest_sector_percent:.1f}%. Broad-market exposure receives a limited "
                "look-through credit."
            ),
            details={
                "largest_position_percent": round(largest_weight_percent, 2),
                "hhi": round(analytics.concentration_hhi, 4),
                "largest_sector_percent": round(largest_sector_percent, 2),
                "broad_market_percent": round(broad_market_percent, 2),
                "fixed_income_percent": round(fixed_income_percent, 2),
                "foreign_currency_denomination_percent": round(foreign_currency_percent, 2),
            },
        ),
        liquidity_resilience=RiskDimension(
            key="liquidity_resilience",
            label="Liquidity resilience",
            score=round(liquidity_resilience, 1),
            level=_quality_level(liquidity_resilience),
            summary=(
                f"Brokerage cash is {cash_ratio_percent:.1f}% of total equity. This measures "
                "immediate liquidity, not long-term inflation protection."
            ),
            details={"cash_percent": round(cash_ratio_percent, 2)},
        ),
        interpretation=_interpretation(total_score, diversification_quality),
        disclaimer=(
            "Heuristic market-risk indicator for educational comparison; it is not a "
            "regulatory risk class, loss probability, suitability assessment, or investment advice."
        ),
        limitations=[
            "Historical measures reconstruct current quantities backwards and are not actual "
            "account performance.",
            "The score uses historical observations and cannot predict future losses.",
            "Broad-market funds receive a limited diversification credit. External "
            "listing data does not include fund holdings, so only explicitly curated "
            "broad-market instruments receive this treatment.",
            "Currency denomination is shown as context and is not treated as market risk; "
            "hedging and economic revenue exposure are unknown.",
            "Investment horizon changes a person's capacity to tolerate losses, not the "
            "observed volatility or drawdown reported here.",
            "Thresholds are documented portfolio heuristics and are not fitted to real "
            "customer outcomes.",
        ],
    )


def interpolate(value: float, anchors: Sequence[tuple[float, float]]) -> float:
    if value <= anchors[0][0]:
        return float(anchors[0][1])
    for (left_x, left_y), (right_x, right_y) in zip(anchors, anchors[1:], strict=False):
        if value <= right_x:
            position = (value - left_x) / (right_x - left_x)
            return round(left_y + position * (right_y - left_y), 4)
    return float(anchors[-1][1])


def _component(
    key: str,
    score: float,
    raw_value: float,
    raw_unit: str,
    summary: str,
    details: dict[str, float] | None = None,
) -> RiskComponent:
    definition = COMPONENTS[key]
    bounded_score = min(max(score, 0), 100)
    return RiskComponent(
        key=key,
        label=definition.label,
        score=round(bounded_score, 1),
        weight=definition.weight,
        contribution=round(bounded_score * definition.weight, 2),
        raw_value=round(raw_value, 4),
        raw_unit=raw_unit,
        summary=summary,
        details=details or {},
    )


def _asset_mix_score(analytics: AnalyticsResponse) -> tuple[float, dict[str, float]]:
    details: dict[str, float] = {}
    weighted_score = 0.0
    for allocation in analytics.allocations.get("asset_class", []):
        factor = _asset_risk_factor(allocation.label)
        exposure = allocation.weight * 100
        details[allocation.label] = round(exposure, 2)
        weighted_score += allocation.weight * factor * 100
    return round(weighted_score, 4), details


def _asset_risk_factor(label: str) -> float:
    normalized = label.casefold()
    if "cash" in normalized or "money market" in normalized:
        return 0.05
    if "bond" in normalized or "fixed income" in normalized:
        return 0.30
    if "crypto" in normalized:
        return 1.00
    if "commodity" in normalized or "gold" in normalized:
        return 0.55
    if "real estate" in normalized or "reit" in normalized:
        return 0.75
    if "equity etf" in normalized or "broad" in normalized:
        return 0.60
    if "equity" in normalized or "stock" in normalized:
        return 0.75
    return 0.75


def _foreign_currency_exposure(analytics: AnalyticsResponse, base_currency: str) -> float:
    return sum(
        allocation.weight
        for allocation in analytics.allocations.get("currency", [])
        if allocation.label.upper() != base_currency.upper()
    )


def _largest_allocation(analytics: AnalyticsResponse, dimension: str) -> tuple[str, float]:
    allocations = analytics.allocations.get(dimension, [])
    if not allocations:
        return "Unknown", 0.0
    largest = max(allocations, key=lambda item: item.weight)
    return largest.label, largest.weight * 100


def _allocation_weight(analytics: AnalyticsResponse, dimension: str, label: str) -> float:
    return sum(
        item.weight * 100
        for item in analytics.allocations.get(dimension, [])
        if item.label.casefold() == label.casefold()
    )


def _risk_level(score: float) -> str:
    if score < 25:
        return "low"
    if score < 50:
        return "moderate"
    if score < 75:
        return "elevated"
    return "high"


def _quality_level(score: float) -> str:
    if score >= 75:
        return "strong"
    if score >= 50:
        return "adequate"
    return "weak"


def _interpretation(market_risk: float, diversification: float) -> str:
    risk = _risk_level(market_risk).replace("_", " ")
    quality = _quality_level(diversification)
    if diversification >= 75:
        return (
            f"The portfolio has {risk} measured market risk and {quality} diversification. "
            "Strong diversification reduces concentration risk but does not remove normal "
            "market drawdowns."
        )
    return (
        f"The portfolio has {risk} measured market risk and {quality} diversification. "
        "Its risk depends materially on a limited set of exposures."
    )
