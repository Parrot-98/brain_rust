use crate::learning;
use crate::neuron::{Neuron, Type};
use rand::RngExt;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

const SPIKE_AMPLITUDE: f32 = 20.0;
const ELIGIBILITY_DECAY_RATE: f32 = 0.05;

// a signal in flight, scheduled to arrive at a neuron at a given time
#[derive(Debug, Clone, Copy)]
struct Event {
    time: f32,
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
    last_reward_time: f32,
}

impl Network {
    /// Creates an empty network: no neurons, clock at zero, and an empty event queue.
    pub fn new() -> Self {
        Self {
            neurons: HashMap::new(),
            time: 0.0,
            event_queue: BinaryHeap::new(),
            last_reward_time: 0.0,
        }
    }

    /// Looks up a neuron by id, creating it with that id and type if it doesn't
    /// exist yet, and returns a mutable reference to it.
    pub fn get_neuron(&mut self, id: usize, neuron_type: Type) -> &mut Neuron {
        self.neurons
            .entry(id)
            .or_insert_with(|| Neuron::new(id, neuron_type))
    }

    /// True if a neuron with this id is in the network.
    pub fn neuron_exists(&self, id: usize) -> bool {
        self.neurons.contains_key(&id)
    }

    /// Inserts a neuron, keyed by its id. A no-op if that id is already taken.
    pub fn add_neuron(&mut self, neuron: Neuron) {
        self.neurons.entry(neuron.id).or_insert(neuron);
    }

    /// Fills the network with `total` neurons split roughly evenly into input,
    /// hidden, and output groups (leftover neurons go to the earlier groups).
    /// Ids continue past whatever ids already exist. Returns the id lists for the
    /// three groups so the caller can wire the layers together.
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

    /// Adds a directed synapse from `from_id` to `to_id` with the given weight.
    /// Returns false if either neuron is missing; otherwise delegates to the
    /// source neuron, which computes the conduction delay from their positions.
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

    /// Wires every neuron in `from_ids` toward every neuron in `to_ids`, but each
    /// individual connection is made only with a probability that decays with the
    /// distance between the two neurons (self-connections are skipped). Returns
    /// how many connections were actually created.
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

                // probability of connecting falls off with distance, scaled by the
                // source neuron's effective_radius: distance == 0 -> probability 1.0,
                // distance == radius -> probability 0.5, further away -> approaches 0
                let probability = (from_radius / (from_radius + distance)).clamp(0.0, 1.0);

                if rng.random::<f32>() < probability && self.connect(from_id, to_id, weight) {
                    connected += 1;
                }
            }
        }

        connected
    }

    /// Broadcasts a reward (or punishment, if negative) to the whole network.
    /// Every connection's eligibility trace is first decayed by how long it's been
    /// since the last reward, then multiplied by `reward` and applied to the
    /// weight. Traces are cleared afterward. This is the reward-modulated STDP
    /// step: recent, well-timed synapses move in the rewarded direction.
    pub fn apply_reward(&mut self, reward: f32) {
        let elapsed = (self.time - self.last_reward_time).max(0.0);
        let decay = (-ELIGIBILITY_DECAY_RATE * elapsed).exp();

        for neuron in self.neurons.values_mut() {
            for connection in neuron.connections.iter_mut() {
                let decayed_eligibility = connection.eligibility * decay;
                connection.adjust_weight(decayed_eligibility * reward);
                connection.eligibility = 0.0;
            }
        }

        self.last_reward_time = self.time;
    }

    /// Injects an external signal into `start_id` at the current time and then
    /// runs the event loop until nothing is left in flight. Returns whether any
    /// neuron fired. Returns false immediately if the neuron doesn't exist.
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

    /// Like `send_signal`, but only enqueues the signal (to arrive at the current
    /// time) without running the simulation. The caller advances time with `step`.
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

    /// Advances the simulation clock by `dt` and delivers any events that have
    /// now come due. Returns whether anything fired during this step.
    pub fn step(&mut self, dt: f32) -> bool {
        self.time += dt;
        self.deliver_due_events()
    }

    /// Drains every event whose scheduled time is at or before the current clock.
    /// For each: deliver the signal to the target neuron and check if it fires.
    /// A firing neuron (a) schedules new events on all its outgoing connections
    /// after their conduction delays, and (b) triggers STDP learning between the
    /// firing neuron and both its upstream source and its downstream targets.
    /// Returns whether any neuron fired.
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
                for connection in &connections {
                    self.event_queue.push(Event {
                        time: event.time + connection.delay,
                        target_id: connection.target_id,
                        // a fixed spike amplitude scaled by connection weight is what
                        // actually lets a spike cascade to downstream neurons
                        influence: SPIKE_AMPLITUDE * connection.weight,
                        source_id: Some(firing_id),
                    });
                }

                if let Some(source_id) = event.source_id {
                    if let Some(pre_neuron) = self.neurons.get_mut(&source_id) {
                        learning::train(pre_neuron, event.target_id, Some(event.time));
                    }
                }

                for connection in &connections {
                    let target_last_fired = self
                        .neurons
                        .get(&connection.target_id)
                        .and_then(|target| target.last_fired);

                    if let Some(firing_neuron) = self.neurons.get_mut(&firing_id) {
                        learning::train(firing_neuron, connection.target_id, target_last_fired);
                    }
                }
            }
        }

        anything_fired
    }

    /// Repeatedly jumps the clock straight to the next queued event and processes
    /// it, until the queue is empty (the network has "gone quiet"). A safety cap
    /// of MAX_EVENTS stops runaway self-sustaining activity. Returns whether
    /// anything fired along the way.
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
