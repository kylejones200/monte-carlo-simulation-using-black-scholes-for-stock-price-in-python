//! GBM terminal price simulation (log-normal multiplicative steps).

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 33) as f64 / (1u64 << 31) as f64
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Simulate terminal prices after `n_steps` GBM steps for `n_paths` paths.
pub fn gbm_terminal_prices(
    s0: f64,
    log_drift: f64,
    log_vol: f64,
    n_steps: usize,
    n_paths: usize,
    seed: u64,
) -> Vec<f64> {
    let mut rng = Lcg::new(seed);
    let mut out = Vec::with_capacity(n_paths);
    for _ in 0..n_paths {
        let mut price = s0;
        for _ in 0..n_steps {
            let z = rng.normal();
            price *= (log_drift + log_vol * z).exp();
        }
        out.push(price);
    }
    out
}
