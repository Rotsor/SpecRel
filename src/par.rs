//! Data parallelism for the scene rebuild: on rayon's thread pool natively, on the calling
//! thread on wasm (which has no threads here). Either way the results are the same.

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

/// Run `a` and `b`, potentially in parallel.
#[cfg(not(target_arch = "wasm32"))]
pub fn join<A: Send, B: Send>(
    a: impl FnOnce() -> A + Send,
    b: impl FnOnce() -> B + Send,
) -> (A, B) {
    rayon::join(a, b)
}

/// Run `a` and `b`, potentially in parallel.
#[cfg(target_arch = "wasm32")]
pub fn join<A, B>(a: impl FnOnce() -> A, b: impl FnOnce() -> B) -> (A, B) {
    (a(), b())
}

/// `xs.iter().map(f).collect()`, potentially in parallel.
#[cfg(not(target_arch = "wasm32"))]
pub fn map<T: Sync, U: Send>(xs: &[T], f: impl Fn(&T) -> U + Sync + Send) -> Vec<U> {
    xs.par_iter().with_min_len(1024).map(f).collect()
}

/// `xs.iter().map(f).collect()`, potentially in parallel.
#[cfg(target_arch = "wasm32")]
pub fn map<T, U>(xs: &[T], f: impl Fn(&T) -> U) -> Vec<U> {
    xs.iter().map(f).collect()
}
