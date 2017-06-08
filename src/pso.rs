//! Particle Swarm Optimization.
//!
//! # Example
//! ```
//! extern crate meta_heuristics;
//! extern crate rand;
//!
//! use meta_heuristics::pso;
//! use std::cmp;
//!
//! #[derive(Clone, Copy)]
//! struct Particle {
//!     pos: f64,
//!     vel: f64,
//!     best: (f64, f64),
//! }
//!
//! fn eval_func(x: f64) -> f64 {
//!     1.0 - ((x - 3.0) * x + 2.0) * x * x
//! }
//!
//! impl PartialEq for Particle {
//!     fn eq(&self, rhs: &Self) -> bool {
//!         self.pos == rhs.pos
//!     }
//! }
//!
//! impl Eq for Particle {}
//!
//! impl PartialOrd for Particle {
//!     fn partial_cmp(&self, rhs: &Self) -> Option<cmp::Ordering> {
//!         Some(self.cmp(rhs))
//!     }
//! }
//!
//! impl Ord for Particle {
//!     fn cmp(&self, rhs: &Self) -> cmp::Ordering {
//!         let self_eval = eval_func(self.pos);
//!         let rhs_eval = eval_func(rhs.pos);
//!         self_eval.partial_cmp(&rhs_eval).unwrap()
//!     }
//! }
//!
//! impl pso::Particle for Particle {
//!     type Pos = f64;
//!     type Eval = f64;
//!
//!     fn new_random() -> Self {
//!         use rand::{random, Closed01};
//!
//!         let Closed01(x) = random::<Closed01<f64>>();
//!         let x = 4.0 * x - 1.0;
//!         Self {
//!             pos: x,
//!             vel: 0.0,
//!             best: (x, eval_func(x)),
//!         }
//!     }
//!
//!     fn eval(&self) -> Self::Eval {
//!         eval_func(self.pos)
//!     }
//!
//!     fn pos(&self) -> Self::Pos {
//!         self.pos
//!     }
//!     fn vel(&self) -> Self::Pos {
//!         self.vel
//!     }
//!     fn best(&self) -> (Self::Pos, Self::Eval) {
//!         self.best
//!     }
//!     fn pos_mut(&mut self) -> &mut Self::Pos {
//!         &mut self.pos
//!     }
//!     fn vel_mut(&mut self) -> &mut Self::Pos {
//!         &mut self.vel
//!     }
//!     fn best_mut(&mut self) -> &mut (Self::Pos, Self::Eval) {
//!         &mut self.best
//!     }
//! }
//!
//! fn main() {
//!     let mut pso: pso::PSO<Particle> = pso::PSO::new(8, 0.9, 0.9, 0.9);
//!
//!     for i in 0..10 {
//!         pso.update();
//!         let (Particle { pos: x, .. }, e) = pso.best();
//!         println!("{} {:.3} {:.3}", i, x, e);
//!     }
//!
//!     assert!(pso.best().1 > 1.5);
//! }
//! ```

use std::ops;

pub trait Particle {
    type Pos: Copy + ops::Add<Output = Self::Pos> + ops::Sub<Output = Self::Pos> + ops::Mul<f64, Output = Self::Pos>;
    type Eval: Copy + PartialOrd;

    fn new_random() -> Self;
    fn eval(&self) -> Self::Eval;

    fn pos(&self) -> Self::Pos;
    fn vel(&self) -> Self::Pos;
    fn best(&self) -> (Self::Pos, Self::Eval);
    fn pos_mut(&mut self) -> &mut Self::Pos;
    fn vel_mut(&mut self) -> &mut Self::Pos;
    fn best_mut(&mut self) -> &mut (Self::Pos, Self::Eval);
}

pub struct PSO<T: Particle> {
    particles: Vec<T>,
    inetia: f64,
    c_local: f64,
    c_global: f64,
    best: (T, T::Eval),
}

impl<T> PSO<T>
    where T: Particle + Ord + Copy
{
    pub fn new(particles_num: usize, inetia: f64, c_local: f64, c_global: f64) -> Self {
        let mut particles = Vec::with_capacity(particles_num);
        for _ in 0..particles_num {
            particles.push(T::new_random());
        }

        let best = Self::calc_best(&particles);

        Self {
            particles,
            inetia,
            c_local,
            c_global,
            best,
        }
    }

    fn calc_best(particles: &[T]) -> (T, T::Eval) {
        let best = particles.iter().max().unwrap();
        (*best, best.eval())
    }

    pub fn update(&mut self) {
        for p in &mut self.particles {
            let new_pos = p.pos() + p.vel();
            *p.pos_mut() = new_pos;
        }

        for mut p in &mut self.particles {
            let new_vel = p.vel() * self.inetia +
                          (p.best().0 - p.pos()) * self.c_local * Self::rand_01() +
                          (self.best.0.pos() - p.pos()) * self.c_global * Self::rand_01();
            *p.vel_mut() = new_vel;
        }

        for mut p in &mut self.particles {
            let e = p.eval();
            if e > p.best().1 {
                *p.best_mut() = (p.pos(), e);
            }
        }

        self.best = Self::calc_best(&self.particles);
    }

    pub fn best(&self) -> (T, T::Eval) {
        self.best
    }

    fn rand_01() -> f64 {
        use rand::{random, Closed01};

        let Closed01(val) = random::<Closed01<_>>();
        val
    }
}
