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
    // creates a connection with weight clamped to the allowed range and zero eligibility
    pub fn new(target_id: usize, weight: f32, delay: f32) -> Self {
        Self {
            target_id,
            weight: weight.clamp(MIN_WEIGHT, MAX_WEIGHT),
            delay,
            eligibility: 0.0,
        }
    }

    // nudges the weight by delta, clamped to the allowed range
    pub fn adjust_weight(&mut self, delta: f32) {
        self.weight = (self.weight + delta).clamp(MIN_WEIGHT, MAX_WEIGHT);
    }

    // adds to the eligibility trace left by STDP, clamped to the allowed range
    pub fn accumulate_eligibility(&mut self, delta: f32) {
        self.eligibility = (self.eligibility + delta).clamp(MIN_ELIGIBILITY, MAX_ELIGIBILITY);
    }
}
