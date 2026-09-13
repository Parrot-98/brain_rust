use crate::connection::Connection;
use rand::RngExt;

// how fast a signal travels along a connection
const CONDUCTION_SPEED: f32 = 50.0;
const MIN_DELAY: f32 = 0.5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    Input,
    Hidden,
    Output,
}

#[derive(Debug, Clone, Copy)]
pub struct Position {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Position {
    /// Straight-line (Euclidean) distance between two points in 3D space.
    /// Used to decide connection probability and signal conduction delay.
    pub fn distance_to(&self, other: &Position) -> f32 {
        ((self.x - other.x).powi(2)
            + (self.y - other.y).powi(2)
            + (self.z - other.z).powi(2))
        .sqrt()
    }
}

#[derive(Debug, Clone)]
pub struct Neuron {
    // identification
    pub id: usize,
    pub neuron_type: Type,

    // electrical
    pub voltage: f32,
    pub resting_voltage: f32,
    pub threshold: f32, // how mach the neuron can be stimulated before it fires
    pub leak_rate: f32, // how fast the voltage drops back to resting_voltage

    // timing
    pub last_update: f32, // the last time the voltage was brought up to date
    pub last_fired: Option<f32>,
    pub refractory_period: f32, // how long the neuron sleeps after firing
    pub refractory_until: f32,

    // geo
    pub position: Position,
    pub effective_radius: f32,

    // social
    pub connections: Vec<Connection>,
}

impl Neuron {
    // creates a new neuron
    pub fn new(id: usize, neuron_type: Type) -> Self {
        let mut rng = rand::rng();

        let resting_voltage = rng.random_range(-75.0..=-65.0); // each neuaon in an acually brain runs on -70 millivolts 
        let threshold = rng.random_range(-58.0..=-50.0);

        Self {
            id,
            neuron_type,

            voltage: resting_voltage,
            resting_voltage,
            threshold,
            leak_rate: 0.1,

            last_update: 0.0,
            last_fired: None,
            refractory_period: 2.0,
            refractory_until: 0.0,

            position: Position {
                x: rng.random_range(-25.0..=25.0),
                y: rng.random_range(-25.0..=25.0),
                z: rng.random_range(-25.0..=25.0),
            },
            effective_radius: rng.random_range(5.0..=20.0),

            connections: Vec::new(),
        }
    }

    /// Adds an outgoing synapse from this neuron to `target_id`. Skips it if one
    /// already exists (no duplicates). The conduction delay is derived from the
    /// physical distance between the two neurons, floored at MIN_DELAY.
    pub fn connect_to(&mut self, target_id: usize, target_position: Position, weight: f32) {
        if !self
            .connections
            .iter()
            .any(|connection| connection.target_id == target_id)
        {
            let distance = self.position.distance_to(&target_position);
            let delay = (distance / CONDUCTION_SPEED).max(MIN_DELAY);

            self.connections
                .push(Connection::new(target_id, weight, delay));
        }
    }

    /// Lazily advances the neuron's state to `time`: since the last update the
    /// membrane voltage has been leaking exponentially back toward resting_voltage,
    /// so this applies that decay in one step instead of simulating every instant.
    fn catch_up(&mut self, time: f32) {
        let elapsed = (time - self.last_update).max(0.0);

        if elapsed > 0.0 {
            let decay = 1.0 - (-self.leak_rate * elapsed).exp();
            self.voltage += (self.resting_voltage - self.voltage) * decay;
        }

        self.last_update = time;
    }

    /// Delivers an incoming signal: first decays the voltage up to `time`, then
    /// adds `influence` (positive = excitatory, negative = inhibitory) to it.
    pub fn receive(&mut self, influence: f32, time: f32) {
        self.catch_up(time);
        self.voltage += influence;
    }

    /// Checks whether the neuron spikes at `time`. Returns false if still in its
    /// refractory (recovery) window. Otherwise, if the voltage has reached the
    /// threshold it fires: resets to resting voltage, records the fire time, and
    /// starts a new refractory period. Returns whether it fired.
    pub fn fire(&mut self, time: f32) -> bool {
        self.catch_up(time);

        if time < self.refractory_until {
            return false; // still recovering from its last spike
        }

        if self.voltage >= self.threshold {
            self.voltage = self.resting_voltage;
            self.last_fired = Some(time);
            self.refractory_until = time + self.refractory_period;
            true
        } else {
            false
        }
    }
}
