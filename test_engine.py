import numpy as np

from engine import MMParams, run_simulation
from models import MarketModel


def test_matched_model_earns_the_spread():
    """GBM world + correct vol: no model error, hedging residual ~ zero-mean noise."""
    res = run_simulation(MarketModel("gbm"), MMParams(), n_runs=400, seed=1)
    np.testing.assert_allclose(res.model_error, 0.0, atol=1e-9)
    assert abs(res.hedge_residual.mean()) < 0.5
    assert res.final_pnl.mean() > 0.8 * res.spread_capture.mean()
    assert (res.final_pnl < 0).mean() < 0.02


def test_attribution_adds_up():
    res = run_simulation(MarketModel("bates"), MMParams(), n_runs=20, seed=2)
    total = res.spread_capture + res.model_error + res.hedge_residual
    np.testing.assert_allclose(total, res.final_pnl, atol=1e-9)


def test_book_is_flat_at_the_end():
    res = run_simulation(MarketModel("merton"), MMParams(), n_runs=20, seed=3)
    assert np.all(res.inventory[:, -1] == 0)
    np.testing.assert_allclose(res.hedge[:, -1], 0.0, atol=1e-9)


def test_wrong_model_gets_picked_off():
    """BS at the diffusion vol under-prices options in a jump world; informed flow buys them."""
    res = run_simulation(MarketModel("merton"), MMParams(informed_share=0.5), n_runs=300, seed=4)
    assert res.model_error.mean() < 0
    matched = run_simulation(MarketModel("gbm"), MMParams(informed_share=0.5), n_runs=300, seed=4)
    assert res.final_pnl.std() > 3 * matched.final_pnl.std()


def test_calibrated_vol_removes_initial_mispricing():
    p = MMParams(mm_vol_mode="calibrated")
    res = run_simulation(MarketModel("merton"), p, n_runs=1, seed=5)
    np.testing.assert_allclose(res.fair[0, 0], res.true_val[0, 0], rtol=1e-6)


def test_put_steamroller_has_fat_left_tail():
    m = MarketModel("merton", lam=0.4, jump_mean=-0.15, jump_std=0.05)
    p = MMParams(strike=95, is_call=False, informed_share=0.05, p_buy=0.8)
    pnl = run_simulation(m, p, n_runs=600, seed=6).final_pnl
    skew = ((pnl - pnl.mean()) ** 3).mean() / pnl.std() ** 3
    assert np.median(pnl) > pnl.mean()
    assert skew < -1
