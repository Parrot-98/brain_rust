const MIN_WEIGHT: f32 = 0.0;
const MAX_WEIGHT: f32 = 2.0;

const MIN_ELIGIBILITY: f32 = -5.0;
const MAX_ELIGIBILITY: f32 = 5.0;

#[derive(Debug, Clone, Copy)]
pub struct Connection {
    pub target_id: usize,
    pub weight: f32,
    pub delay: f32,
    pub eligibility: f32,
}

impl Connection {
    /// Creates a connection to `target_id`, clamping the starting weight into the
    /// allowed range and starting with zero eligibility (no pending learning).
    pub fn new(target_id: usize, weight: f32, delay: f32) -> Self {
        Self {
            target_id,
            weight: weight.clamp(MIN_WEIGHT, MAX_WEIGHT),
            delay,
            eligibility: 0.0,
        }
    }

    /// Nudges the synaptic weight by `delta`, keeping it within [MIN_WEIGHT, MAX_WEIGHT].
    /// Called when a reward is applied to turn pending eligibility into a real weight change.
    pub fn adjust_weight(&mut self, delta: f32) {
        self.weight = (self.weight + delta).clamp(MIN_WEIGHT, MAX_WEIGHT);
    }

    /// Adds to the eligibility trace (the "this synapse recently did something
    /// learnable" marker left by STDP), clamped so it can't grow without bound.
    /// The trace is later multiplied by a reward to decide the actual weight change.
    pub fn accumulate_eligibility(&mut self, delta: f32) {
        self.eligibility = (self.eligibility + delta).clamp(MIN_ELIGIBILITY, MAX_ELIGIBILITY);
    }
}
