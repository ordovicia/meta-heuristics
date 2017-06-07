//! Particle Swarm Optimization.
//!
//! # Example
//! ```
//! extern crate meta_heuristics;
//! extern crate rand;
//!
//! use meta_heuristics::pso;
//! use std::{ops, cmp};
//!
//! #[derive(Default, Clone, Copy)]
//! struct Pos {
//!     x: f64,
//!     y: f64,
//! }
//!
//! impl ops::Add for Pos {
//!     type Output = Self;
//!     fn add(self, rhs: Self) -> Self {
//!         Self {
//!             x: self.x + rhs.x,
//!             y: self.y + rhs.y,
//!         }
//!     }
//! }
//!
//! impl ops::Sub for Pos {
//!     type Output = Self;
//!     fn sub(self, rhs: Self) -> Self {
//!         Self {
//!             x: self.x - rhs.x,
//!             y: self.y - rhs.y,
//!         }
//!     }
//! }
//!
//! impl ops::Mul<f64> for Pos {
//!     type Output = Self;
//!     fn mul(self, rhs: f64) -> Self {
//!         Self {
//!             x: self.x * rhs,
//!             y: self.y * rhs,
//!         }
//!     }
//! }
//!
//! impl Pos {
//!     fn eval(&self) -> f64 {
//!         -(self.x * self.x + self.y * self.y)
//!     }
//! }
//!
//! #[derive(Clone, Copy)]
//! struct Particle {
//!     pos: Pos,
//!     vel: Pos,
//!     best: (Pos, f64),
//! }
//!
//! impl PartialEq for Particle {
//!     fn eq(&self, rhs: &Self) -> bool {
//!         self.pos.eval() == rhs.pos.eval()
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
//!         let self_eval = self.pos.eval();
//!         let rhs_eval = rhs.pos.eval();
//!         if self_eval < rhs_eval {
//!             cmp::Ordering::Greater
//!         } else if self_eval > rhs_eval {
//!             cmp::Ordering::Less
//!         } else {
//!             cmp::Ordering::Equal
//!         }
//!     }
//! }
//!
//! impl pso::Particle for Particle {
//!     type Pos = Pos;
//!     type Eval = f64;
//!
//!     fn new_random() -> Self {
//!         use rand::{random, Closed01};
//!
//!         let Closed01(x) = random::<Closed01<_>>();
//!         let Closed01(y) = random::<Closed01<_>>();
//!         let pos = Pos { x, y };
//!         Self { pos: pos, vel: Pos::default(), best: (pos, pos.eval()) }
//!     }
//!
//!     fn eval(&self) -> Self::Eval {
//!         self.pos.eval()
//!     }
//!
//!     fn pos(&self) -> Self::Pos { self.pos }
//!     fn vel(&self) -> Self::Pos { self.vel }
//!     fn best(&self) -> (Self::Pos, Self::Eval) { self.best }
//!     fn pos_mut(&mut self) -> &mut Self::Pos { &mut self.pos }
//!     fn vel_mut(&mut self) -> &mut Self::Pos { &mut self.vel }
//!     fn best_mut(&mut self) -> &mut (Self::Pos, Self::Eval) { &mut self.best }
//! }
//!
//! fn main() {
//!     let mut pso: pso::PSO<Particle> = pso::PSO::new(8, 0.9, 0.9, 0.9);
//!     let mut loop_cnt = 0;
//!
//!     loop {
//!         if loop_cnt >= 30 {
//!             break;
//!         }
//!         if pso.update() {
//!             break;
//!         }
//!
//!         loop_cnt += 1;
//!         let (Particle { pos: Pos { x, y }, .. } , e) = pso.best();
//!         println!("({:.2}, {:.2}) {:.2}", x, y, e);
//!     }
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

    pub fn update(&mut self) -> bool {
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

        let best = Self::calc_best(&self.particles);
        if best.1 > self.best.1 {
            self.best = best;
            false
        } else {
            true
        }
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
