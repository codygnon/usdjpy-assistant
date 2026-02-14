"""
Unit tests for pips and R-multiple calculation consistency.

Verifies that buy wins (exit > entry) and sell wins (exit < entry) yield positive pips
and positive R when stop loss is set. Convention used across api/main.py, core/trade_sync.py,
close_trade.py, and adapters: buy pips = (exit - entry)/pip_size, sell pips = (entry - exit)/pip_size.

Tests use local implementations of the formulas so the suite runs without project deps;
these formulas must stay in sync with close_trade.compute_pips and compute_r_multiple.
"""
import unittest


def _compute_pips(side: str, entry: float, exit_price: float, pip_size: float) -> float:
    if side == "buy":
        return (exit_price - entry) / pip_size
    if side == "sell":
        return (entry - exit_price) / pip_size
    raise ValueError("side must be 'buy' or 'sell'")


def _compute_r_multiple(
    pips: float, entry: float, stop: float | None, pip_size: float
) -> tuple[float | None, float | None]:
    if stop is None:
        return None, None
    risk_pips = abs(entry - stop) / pip_size
    if risk_pips == 0:
        return float(risk_pips), None
    return float(risk_pips), float(pips / risk_pips)


class TestPipsAndRMultiple(unittest.TestCase):
    pip_size = 0.01

    def test_buy_win_positive_pips(self):
        """Buy: exit > entry => winning trade => positive pips."""
        entry, exit_price = 150.0, 150.5
        pips = _compute_pips("buy", entry, exit_price, self.pip_size)
        self.assertGreater(pips, 0, "buy win should have positive pips")
        self.assertAlmostEqual(pips, 50.0, places=2)

    def test_sell_win_positive_pips(self):
        """Sell: exit < entry => winning trade => positive pips."""
        entry, exit_price = 150.5, 150.0
        pips = _compute_pips("sell", entry, exit_price, self.pip_size)
        self.assertGreater(pips, 0, "sell win should have positive pips")
        self.assertAlmostEqual(pips, 50.0, places=2)

    def test_buy_win_positive_r(self):
        """Buy win with stop => positive R-multiple."""
        entry, exit_price, stop = 150.0, 150.5, 149.5
        pips = _compute_pips("buy", entry, exit_price, self.pip_size)
        risk_pips, r_mult = _compute_r_multiple(pips, entry, stop, self.pip_size)
        self.assertIsNotNone(r_mult, "R should be computed when stop is set")
        self.assertGreater(r_mult, 0, "buy win should have positive R")
        self.assertAlmostEqual(r_mult, 1.0, places=2)

    def test_sell_win_positive_r(self):
        """Sell win with stop => positive R-multiple."""
        entry, exit_price, stop = 150.5, 150.0, 151.0
        pips = _compute_pips("sell", entry, exit_price, self.pip_size)
        risk_pips, r_mult = _compute_r_multiple(pips, entry, stop, self.pip_size)
        self.assertIsNotNone(r_mult, "R should be computed when stop is set")
        self.assertGreater(r_mult, 0, "sell win should have positive R")
        self.assertAlmostEqual(r_mult, 1.0, places=2)

    def test_buy_loss_negative_pips(self):
        """Buy: exit < entry => losing trade => negative pips."""
        entry, exit_price = 150.0, 149.5
        pips = _compute_pips("buy", entry, exit_price, self.pip_size)
        self.assertLess(pips, 0)

    def test_sell_loss_negative_pips(self):
        """Sell: exit > entry => losing trade => negative pips."""
        entry, exit_price = 150.0, 150.5
        pips = _compute_pips("sell", entry, exit_price, self.pip_size)
        self.assertLess(pips, 0)


if __name__ == "__main__":
    unittest.main()
