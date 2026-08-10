use rand::RngExt;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

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
    pub fn distance_to(&self, other: &Position) -> f32 {
        ((self.x - other.x).powi(2)
            + (self.y - other.y).powi(2)
            + (self.z - other.z).powi(2))
        .sqrt()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Connection {
    pub target_id: usize,
    pub weight: f32,
    pub delay: f32,
}

impl Connection {
    pub fn new(target_id: usize, weight: f32, delay: f32) -> Self {
        Self {
            target_id,
            weight: weight.clamp(MIN_WEIGHT, MAX_WEIGHT),
            delay,
        }
    }

    pub fn adjust_weight(&mut self, delta: f32) {
        self.weight = (self.weight + delta).clamp(MIN_WEIGHT, MAX_WEIGHT);
    }
}

// how fast the signel travels
const CONDUCTION_SPEED: f32 = 50.0;
const MIN_DELAY: f32 = 0.5;

const SPIKE_AMPLITUDE: f32 = 20.0;

const MIN_WEIGHT: f32 = 0.0;
const MAX_WEIGHT: f32 = 2.0;

#[derive(Debug, Clone)]
pub struct Neuron {
    // identification
    pub id: usize,
    pub neuron_type: Type,

    // electrical
    pub voltage: f32,
    pub resting_voltage: f32,
    pub threshold: f32,
    pub leak_rate: f32, // how fast the voltage drops back to the resting_voltage

    // timing
    pub last_update: f32,       // the last time the voltage was used
    pub last_fired: Option<f32>,
    pub refractory_period: f32, // how long the neuron has to sleep before fireing
    pub refractory_until: f32,

    // geo
    pub position: Position,
    pub effective_radius: f32,

    // social
    pub connections: Vec<Connection>,
}

impl Neuron {
    pub fn new(id: usize, neuron_type: Type) -> Self {
        let mut rng = rand::rng();

        let resting_voltage = rng.random_range(-75.0..=-65.0);
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

    // brings the voltage up to date
    fn catch_up(&mut self, time: f32) {
        let elapsed = (time - self.last_update).max(0.0);

        if elapsed > 0.0 {
            let decay = 1.0 - (-self.leak_rate * elapsed).exp();
            self.voltage += (self.resting_voltage - self.voltage) * decay;
        }

        self.last_update = time;
    }

    pub fn receive(&mut self, influence: f32, time: f32) {
        self.catch_up(time);
        self.voltage += influence;
    }

    // check is the neuron fired on time
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

// stores all the secondary stuff of the signel
#[derive(Debug, Clone, Copy)]
struct Event {
    time: f32, // at what time it should arrive
    target_id: usize,
    influence: f32,
    source_id: Option<usize>,
}

impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        self.time == other.time
    }
}
impl Eq for Event {}

impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Event {
    fn cmp(&self, other: &Self) -> Ordering {
        // reversed so the minimum time is popped first
        other
            .time
            .partial_cmp(&self.time)
            .unwrap_or(Ordering::Equal)
    }
}

#[derive(Debug)]
pub struct Network {
    pub neurons: HashMap<usize, Neuron>,
    pub time: f32,
    event_queue: BinaryHeap<Event>,
}

impl Network {
    pub fn new() -> Self {
        Self {
            neurons: HashMap::new(),
            time: 0.0,
            event_queue: BinaryHeap::new(),
        }
    }

    // look up the neuron id if no neuron then it crete that neuron with that id
    pub fn get_neuron(&mut self, id: usize, neuron_type: Type) -> &mut Neuron {
        self.neurons
            .entry(id)
            .or_insert_with(|| Neuron::new(id, neuron_type))
    }

    pub fn neuron_exists(&self, id: usize) -> bool {
        self.neurons.contains_key(&id)
    }

    pub fn add_neuron(&mut self, neuron: Neuron) {
        self.neurons.entry(neuron.id).or_insert(neuron);
    }

    pub fn populate(&mut self, total: usize) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        let types = [Type::Input, Type::Hidden, Type::Output];
        let base = total / types.len();
        let remainder = total % types.len();

        let mut next_id = self.neurons.keys().max().map_or(1, |max_id| max_id + 1);

        let mut input_ids = Vec::new();
        let mut hidden_ids = Vec::new();
        let mut output_ids = Vec::new();

        for (i, neuron_type) in types.iter().enumerate() {
            let count = base + if i < remainder { 1 } else { 0 };

            for _ in 0..count {
                let neuron = Neuron::new(next_id, *neuron_type);

                match neuron_type {
                    Type::Input => input_ids.push(next_id),
                    Type::Hidden => hidden_ids.push(next_id),
                    Type::Output => output_ids.push(next_id),
                }

                self.add_neuron(neuron);
                next_id += 1;
            }
        }

        (input_ids, hidden_ids, output_ids)
    }

    // connects one neuron to another by id.
    pub fn connect(&mut self, from_id: usize, to_id: usize, weight: f32) -> bool {
        let Some(to_position) = self.neurons.get(&to_id).map(|neuron| neuron.position) else {
            return false;
        };

        let Some(from_neuron) = self.neurons.get_mut(&from_id) else {
            return false;
        };

        from_neuron.connect_to(to_id, to_position, weight);
        true
    }

    // connects layers
    pub fn connect_layers(&mut self, from_ids: &[usize], to_ids: &[usize], weight: f32) -> usize {
        let mut rng = rand::rng();
        let mut connected = 0;

        for &from_id in from_ids {
            let Some((from_position, from_radius)) = self
                .neurons
                .get(&from_id)
                .map(|neuron| (neuron.position, neuron.effective_radius))
            else {
                continue;
            };

            for &to_id in to_ids {
                if from_id == to_id {
                    continue;
                }

                let Some(to_position) = self.neurons.get(&to_id).map(|neuron| neuron.position)
                else {
                    continue;
                };

                let distance = from_position.distance_to(&to_position);

                // probability of connecting falls off with distance,
                // scaled by the source neuron's own effective_radius:
                // distance == 0 -> probability 1.0, distance == radius ->
                // probability 0.5, further away -> approaches 0
                let probability = (from_radius / (from_radius + distance)).clamp(0.0, 1.0);

                if rng.random::<f32>() < probability && self.connect(from_id, to_id, weight) {
                    connected += 1;
                }
            }
        }

        connected
    }

    // sends a signal into a neuron right now
    pub fn send_signal(&mut self, start_id: usize, signal: f32) -> bool {
        if !self.neuron_exists(start_id) {
            return false;
        }

        self.event_queue.push(Event {
            time: self.time,
            target_id: start_id,
            influence: signal,
            source_id: None, // externally injected, no neuron caused this
        });

        self.run_to_quiescence()
    }

    // queues a signal to arrive at the network's current time
    pub fn queue_signal(&mut self, start_id: usize, signal: f32) -> bool {
        if !self.neuron_exists(start_id) {
            return false;
        }

        self.event_queue.push(Event {
            time: self.time,
            target_id: start_id,
            influence: signal,
            source_id: None, // externally injected, no neuron caused this
        });

        true
    }

    // advances the clock by `dt
    pub fn step(&mut self, dt: f32) -> bool {
        self.time += dt;
        self.deliver_due_events()
    }

    fn deliver_due_events(&mut self) -> bool {
        let mut anything_fired = false;

        while let Some(event) = self.event_queue.peek() {
            if event.time > self.time {
                break;
            }
            let event = self.event_queue.pop().unwrap();

            // O(1) direct lookup by id -- no scanning through every neuron
            let Some(neuron) = self.neurons.get_mut(&event.target_id) else {
                continue;
            };

            neuron.receive(event.influence, event.time);

            if neuron.fire(event.time) {
                anything_fired = true;

                let firing_id = neuron.id;
                let connections = neuron.connections.clone();
                for connection in connections {
                    self.event_queue.push(Event {
                        time: event.time + connection.delay,
                        target_id: connection.target_id,
                        // a fixed spike amplitude scaled by connection weight,
                        // rather than just the raw weight -- this is what
                        // actually lets a spike cascade to downstream neurons
                        influence: SPIKE_AMPLITUDE * connection.weight,
                        source_id: Some(firing_id),
                    });
                }
            }
        }

        anything_fired
    }


    fn run_to_quiescence(&mut self) -> bool {
        let mut anything_fired = false;
        let mut safety = 0;
        const MAX_EVENTS: usize = 100_000;

        while let Some(event) = self.event_queue.peek().copied() {
            safety += 1;
            if safety > MAX_EVENTS {
                break;
            }

            self.time = event.time;
            if self.deliver_due_events() {
                anything_fired = true;
            }
        }

        anything_fired
    }
}