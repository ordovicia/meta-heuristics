//! Firefly Algorithm.
//!
//! # Example
//! ```
//! extern crate meta_heuristics;
//! extern crate rand;
//!
//! use meta_heuristics::firefly::{self, Firefly};
//!
//! #[derive(Clone, Copy)]
//! struct Particle {
//!     pos: f64,
//! }
//!
//! fn eval_func(x: f64) -> f64 {
//!     1.0 - ((x - 3.0) * x + 2.0) * x * x
//! }
//!
//! impl firefly::Firefly for Particle {
//!     type Pos = f64;
//!     type Eval = f64;
//!
//!     fn new_random() -> Self {
//!         use rand::{random, Closed01};
//!
//!        let Closed01(x) = random::<Closed01<f64>>();
//!        let x = 4.0 * x - 1.5;
//!        Self { pos: x }
//!     }
//!
//!     fn eval(&self) -> Self::Eval {
//!         eval_func(self.pos)
//!     }
//!
//!     fn distance(&self, rhs: &Self) -> f64 {
//!         (self.pos - rhs.pos).abs()
//!     }
//!
//!     fn pos(&self) -> Self::Pos {
//!         self.pos
//!     }
//!     fn pos_mut(&mut self) -> &mut Self::Pos {
//!         &mut self.pos
//!     }
//! }
//!
//! fn main() {
//!     let mut ff: firefly::FireflyAlg<Particle> = firefly::FireflyAlg::new(16, 0.5, 0.2);
//!
//!     for i in 0..30 {
//!         ff.update();
//!         print!("{} ", i);
//!         for &(p, e) in ff.fireflies() {
//!             print!("{:.3} {:.3} ", p.pos(), e);
//!         }
//!         println!();
//!     }
//! }
//! ```

use std::{ops, mem};

pub trait Firefly {
    type Pos: Copy + ops::Add<Output = Self::Pos> + ops::Sub<Output = Self::Pos> + ops::Mul<f64, Output = Self::Pos>;
    type Eval: Copy + PartialOrd;

    fn new_random() -> Self;
    fn eval(&self) -> Self::Eval;
    fn distance(&self, rhs: &Self) -> f64;

    fn pos(&self) -> Self::Pos;
    fn pos_mut(&mut self) -> &mut Self::Pos;
}

pub struct FireflyAlg<T: Firefly + Clone> {
    fireflies: Vec<(T, T::Eval)>,
    beta: f64,
    absorption: f64,
}

impl<T: Firefly + Clone> FireflyAlg<T> {
    pub fn new(fireflies_num: usize, beta: f64, absorption: f64) -> Self {
        let mut fireflies = Vec::with_capacity(fireflies_num);
        for _ in 0..fireflies_num {
            let ff = T::new_random();
            let e = ff.eval();
            fireflies.push((ff, e));
        }

        Self {
            fireflies,
            beta,
            absorption,
        }
    }

    pub fn update(&mut self) {
        let mut new_fireflies = self.fireflies.clone();
        let fireflies_num = self.fireflies.len();

        for i in 0..fireflies_num {
            for j in 0..fireflies_num {
                let ff_i = &self.fireflies[i];
                let ff_j = &self.fireflies[j];
                if ff_j.1 > ff_i.1 {
                    let dist = ff_i.0.distance(&ff_j.0);
                    let pos_diff = (ff_j.0.pos() - ff_i.0.pos()) * self.beta *
                                   (-dist * dist * self.absorption).exp();
                    let new_pos = ff_i.0.pos() + pos_diff;
                    *new_fireflies[i].0.pos_mut() = new_pos;
                    let new_e = new_fireflies[i].0.eval();
                    new_fireflies[i].1 = new_e;
                }
            }
        }

        mem::swap(&mut self.fireflies, &mut new_fireflies);
    }

    pub fn fireflies(&self) -> &Vec<(T, T::Eval)> {
        &self.fireflies
    }
}
