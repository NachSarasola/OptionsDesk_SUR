"""Tests del motor de tasa: lanzamiento cubierto y venta de puts."""
from datetime import date, timedelta

import pytest

from optionsdesk.config.costs import CostModel
from optionsdesk.core.instruments import OptionContract, OptionType
from optionsdesk.core.rates import compute_covered_call, compute_short_put

# Costos cero para aislar la matemática de la tasa
ZERO = CostModel(
    stock_commission_pct=0.0, option_commission_pct=0.0,
    stock_market_fee_pct=0.0, option_market_fee_pct=0.0,
    exercise_fee_pct=0.0, exercise_market_fee_pct=0.0, iva_rate=0.0,
)
REAL = CostModel()  # costos configurados del ALyC


def _call(strike: float, days: int = 30) -> OptionContract:
    return OptionContract(
        symbol=f"GFGC{strike:.0f}T",
        underlying="GGAL",
        option_type=OptionType.CALL,
        strike=strike,
        expiration=date.today() + timedelta(days=days),
        expiry_code="T",
    )


def _put(strike: float, days: int = 30) -> OptionContract:
    return OptionContract(
        symbol=f"GFGV{strike:.0f}T",
        underlying="GGAL",
        option_type=OptionType.PUT,
        strike=strike,
        expiration=date.today() + timedelta(days=days),
        expiry_code="T",
    )


class TestCoveredCall:
    def test_positive_tna_for_itm_call(self):
        # Spot=100, Strike=90, prima=11.5: desembolso=88.5, ingresos=90 → tasa>0
        result = compute_covered_call(
            _call(90.0, 30),
            spot_bid=99.0, spot_ask=100.0,
            call_bid=11.0, call_ask=12.0,
            cost_model=ZERO, price_mode="mid",
        )
        assert result is not None
        assert result.tna_pct > 0.0
        assert result.net_proceeds > result.net_outlay

    def test_tea_greater_than_tna_for_positive_rate(self):
        # Para tasas positivas y corto plazo: TEA > TNA siempre
        result = compute_covered_call(
            _call(95.0, 15),
            spot_bid=99.0, spot_ask=100.0,
            call_bid=5.5, call_ask=6.5,
            cost_model=ZERO,
        )
        assert result is not None
        assert result.tea_pct > result.tna_pct

    def test_returns_none_for_expired_contract(self):
        expired = OptionContract(
            symbol="EXPIRED",
            underlying="GGAL",
            option_type=OptionType.CALL,
            strike=100.0,
            expiration=date.today(),
            expiry_code="X",
        )
        assert compute_covered_call(
            expired, spot_bid=99.0, spot_ask=100.0,
            call_bid=2.0, call_ask=3.0, cost_model=ZERO,
        ) is None

    def test_returns_none_for_put_contract(self):
        assert compute_covered_call(
            _put(90.0, 30),
            spot_bid=99.0, spot_ask=100.0,
            call_bid=2.0, call_ask=3.0, cost_model=ZERO,
        ) is None

    def test_itm_call_has_positive_cushion(self):
        result = compute_covered_call(
            _call(90.0, 30),
            spot_bid=99.0, spot_ask=100.0,
            call_bid=11.0, call_ask=12.0,
            cost_model=ZERO,
        )
        assert result is not None
        assert result.cushion_pct > 0.0

    def test_otm_call_has_smaller_cushion_than_itm(self):
        # Un call OTM tiene colchón = prima/spot (pequeño pero positivo).
        # Un call ITM tiene colchón mayor (= prima ITM / spot, mucho más grande).
        result_otm = compute_covered_call(
            _call(110.0, 30),
            spot_bid=99.0, spot_ask=100.0,
            call_bid=0.5, call_ask=1.5,
            cost_model=ZERO,
        )
        result_itm = compute_covered_call(
            _call(90.0, 30),
            spot_bid=99.0, spot_ask=100.0,
            call_bid=11.0, call_ask=12.0,
            cost_model=ZERO,
        )
        assert result_otm is not None
        assert result_itm is not None
        assert result_otm.cushion_pct < result_itm.cushion_pct

    def test_real_costs_reduce_tna(self):
        kwargs = dict(
            contract=_call(90.0, 30),
            spot_bid=99.0, spot_ask=100.0,
            call_bid=11.0, call_ask=12.0,
        )
        no_cost = compute_covered_call(**kwargs, cost_model=ZERO)
        with_cost = compute_covered_call(**kwargs, cost_model=REAL)
        assert no_cost.tna_pct > with_cost.tna_pct

    def test_market_rights_include_iva(self):
        costs = CostModel(
            stock_commission_pct=0.0,
            option_commission_pct=0.0,
            stock_market_fee_pct=0.08,
            option_market_fee_pct=0.20,
            exercise_fee_pct=0.0,
            exercise_market_fee_pct=0.08,
            iva_rate=0.21,
        )
        assert costs.gross_cost(100_000.0, "option_buy") == pytest.approx(242.0)
        assert costs.exercise_cost(100_000.0) == pytest.approx(96.8)

    def test_dividends_increase_proceeds(self):
        kwargs = dict(
            contract=_call(90.0, 30),
            spot_bid=99.0, spot_ask=100.0,
            call_bid=11.0, call_ask=12.0,
            cost_model=ZERO,
        )
        no_div = compute_covered_call(**kwargs, dividends=0.0)
        with_div = compute_covered_call(**kwargs, dividends=5.0)
        assert with_div.tna_pct > no_div.tna_pct

    def test_period_tna_tea_consistency(self):
        result = compute_covered_call(
            _call(90.0, 30),
            spot_bid=99.0, spot_ask=100.0,
            call_bid=11.0, call_ask=12.0,
            cost_model=ZERO,
        )
        # TNA = period_rate * 365/days
        expected_tna = result.period_rate_pct * 365.0 / result.days
        assert abs(result.tna_pct - expected_tna) < 0.01

    @pytest.mark.parametrize("days", [7, 15, 30, 60, 90])
    def test_various_maturities(self, days: int):
        result = compute_covered_call(
            _call(90.0, days),
            spot_bid=99.0, spot_ask=100.0,
            call_bid=11.0, call_ask=12.0,
            cost_model=ZERO,
        )
        assert result is not None
        assert result.tna_pct > 0.0


class TestShortPut:
    def test_positive_tna(self):
        result = compute_short_put(
            _put(90.0, 30),
            put_bid=1.5, put_ask=2.5, spot=100.0,
            cost_model=ZERO,
        )
        assert result is not None
        assert result.tna_pct > 0.0

    def test_net_capital_gives_higher_tna_than_gross(self):
        kwargs = dict(
            contract=_put(90.0, 30),
            put_bid=1.5, put_ask=2.5, spot=100.0,
            cost_model=ZERO,
        )
        net = compute_short_put(**kwargs, capital_mode="net")
        gross = compute_short_put(**kwargs, capital_mode="gross")
        assert net.tna_pct >= gross.tna_pct

    def test_returns_none_for_call_contract(self):
        assert compute_short_put(
            _call(90.0, 30),
            put_bid=1.5, put_ask=2.5, spot=100.0,
            cost_model=ZERO,
        ) is None

    def test_returns_none_for_expired(self):
        expired = OptionContract(
            symbol="EXP", underlying="GGAL",
            option_type=OptionType.PUT, strike=90.0,
            expiration=date.today(), expiry_code="X",
        )
        assert compute_short_put(
            expired, put_bid=1.5, put_ask=2.5, spot=100.0, cost_model=ZERO
        ) is None

    def test_higher_premium_higher_tna(self):
        r1 = compute_short_put(_put(90.0, 30), put_bid=1.0, put_ask=2.0, spot=100.0, cost_model=ZERO)
        r2 = compute_short_put(_put(90.0, 30), put_bid=3.0, put_ask=4.0, spot=100.0, cost_model=ZERO)
        assert r2.tna_pct > r1.tna_pct

    def test_spread_set_correctly(self):
        result = compute_short_put(
            _put(90.0, 30),
            put_bid=1.5, put_ask=2.5, spot=100.0,
            cost_model=ZERO,
        )
        result.set_spread(100.0)
        assert abs(result.spread_vs_caucion_pct - (result.tna_pct - 100.0)) < 0.001

    def test_assignment_cost_worsens_short_put_breakeven(self):
        costs = CostModel(
            stock_commission_pct=0.0,
            option_commission_pct=0.0,
            stock_market_fee_pct=0.0,
            option_market_fee_pct=0.0,
            exercise_fee_pct=1.0,
            exercise_market_fee_pct=0.0,
            iva_rate=0.0,
        )
        result = compute_short_put(
            _put(90.0, 30),
            put_bid=1.5,
            put_ask=2.5,
            spot=100.0,
            cost_model=costs,
        )

        assert result is not None
        breakeven = 100.0 * (1.0 - result.cushion_pct / 100.0)
        assert breakeven == pytest.approx(88.9)
