"""Tests del parser de tickers de opciones GGAL."""
from datetime import date

import pytest

from optionsdesk.core.instruments import OptionContract, OptionType, parse_option_symbol

CALENDAR = {
    "E": date(2026, 5, 15),
    "F": date(2026, 6, 19),
    "G": date(2026, 7, 17),
}


class TestParseOptionSymbol:
    def test_parse_call(self):
        c = parse_option_symbol("GFGC7000F", CALENDAR)
        assert c is not None
        assert c.option_type == OptionType.CALL
        assert c.strike == 7000.0
        assert c.expiration == CALENDAR["F"]
        assert c.expiry_code == "F"
        assert c.underlying == "GGAL"

    def test_parse_put(self):
        p = parse_option_symbol("GFGV8500G", CALENDAR)
        assert p is not None
        assert p.option_type == OptionType.PUT
        assert p.strike == 8500.0
        assert p.expiration == CALENDAR["G"]

    def test_lowercase_input_normalized(self):
        c = parse_option_symbol("gfgc7000f", CALENDAR)
        assert c is not None
        assert c.symbol == "GFGC7000F"
        assert c.option_type == OptionType.CALL

    def test_unknown_expiry_returns_none(self):
        assert parse_option_symbol("GFGC7000Z", CALENDAR) is None

    def test_invalid_prefix_returns_none(self):
        assert parse_option_symbol("MSFT200C", CALENDAR) is None

    def test_empty_string_returns_none(self):
        assert parse_option_symbol("", CALENDAR) is None

    def test_whitespace_stripped(self):
        c = parse_option_symbol("  GFGC7000F  ", CALENDAR)
        assert c is not None

    def test_days_to_expiry_type(self):
        c = parse_option_symbol("GFGC7000E", CALENDAR)
        assert c is not None
        assert isinstance(c.days_to_expiry, int)
        assert c.days_to_expiry >= 0

    def test_years_to_expiry_consistent(self):
        c = parse_option_symbol("GFGC7000E", CALENDAR)
        assert c is not None
        assert abs(c.years_to_expiry - c.days_to_expiry / 365.0) < 1e-10

    @pytest.mark.parametrize("strike", [6000, 7500, 8500, 10000, 12000])
    def test_various_strikes(self, strike: int):
        c = parse_option_symbol(f"GFGC{strike}F", CALENDAR)
        assert c is not None
        assert c.strike == float(strike)


class TestMoneyness:
    def test_itm_call(self):
        c = OptionContract("SYM", "GGAL", OptionType.CALL, 90.0,
                           date(2026, 6, 19), "F")
        assert c.moneyness(100.0) == "ITM"

    def test_otm_call(self):
        c = OptionContract("SYM", "GGAL", OptionType.CALL, 110.0,
                           date(2026, 6, 19), "F")
        assert c.moneyness(100.0) == "OTM"

    def test_itm_put(self):
        p = OptionContract("SYM", "GGAL", OptionType.PUT, 110.0,
                           date(2026, 6, 19), "F")
        assert p.moneyness(100.0) == "ITM"

    def test_otm_put(self):
        p = OptionContract("SYM", "GGAL", OptionType.PUT, 90.0,
                           date(2026, 6, 19), "F")
        assert p.moneyness(100.0) == "OTM"

    def test_atm_within_threshold(self):
        c = OptionContract("SYM", "GGAL", OptionType.CALL, 101.0,
                           date(2026, 6, 19), "F")
        assert c.moneyness(100.0, atm_threshold_pct=2.0) == "ATM"
