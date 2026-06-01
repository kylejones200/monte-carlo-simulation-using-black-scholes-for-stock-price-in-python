use monte_carlo_simulation_using_black_scholes_for_stock_price_in_python_core::gbm_terminal_prices;
use numpy::{PyArray1, IntoPyArray};
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (s0, log_drift, log_vol, n_steps, n_paths, seed=42))]
fn gbm_terminal_prices_py<'py>(
    py: Python<'py>,
    s0: f64,
    log_drift: f64,
    log_vol: f64,
    n_steps: usize,
    n_paths: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(gbm_terminal_prices(s0, log_drift, log_vol, n_steps, n_paths, seed).into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (s0, log_drift, log_vol, n_steps, n_paths, seed=42, iterations=100))]
fn bench_kernel_py(
    s0: f64,
    log_drift: f64,
    log_vol: f64,
    n_steps: usize,
    n_paths: usize,
    seed: u64,
    iterations: usize,
) -> PyResult<f64> {
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = gbm_terminal_prices(s0, log_drift, log_vol, n_steps, n_paths, seed);
    }
    Ok(start.elapsed().as_secs_f64())
}

#[pymodule]
fn monte_carlo_simulation_using_black_scholes_for_stock_price_in_python_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gbm_terminal_prices_py, m)?)?;
    m.add_function(wrap_pyfunction!(bench_kernel_py, m)?)?;
    Ok(())
}
