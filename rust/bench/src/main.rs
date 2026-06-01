use monte_carlo_simulation_using_black_scholes_for_stock_price_in_python_core::gbm_terminal_prices;

fn main() {
    for _ in 0..500 {
        let _ = gbm_terminal_prices(100.0, 0.0002, 0.01, 252, 1000, 42);
    }
}
