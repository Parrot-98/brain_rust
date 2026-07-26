use rand::RngExt;

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

#[derive(Debug, Clone, Copy)]
pub struct Connection {
    pub target_id: usize,
    pub weight: f32,
}

impl Connection {
    pub fn new(target_id: usize, weight: f32) -> Self {
        Self {
            target_id,
            weight,
        }
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
    pub threshold: f32,

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

            position: Position {
                x: rng.random_range(-100.0..=100.0),
                y: rng.random_range(-100.0..=100.0),
                z: rng.random_range(-100.0..=100.0),
            },
            effective_radius: rng.random_range(5.0..=20.0),

            connections: Vec::new(),
        }
    }

    pub fn connect_to(&mut self, target_id: usize, weight: f32) {
        self.connections
            .push(Connection::new(target_id, weight));
    }
}

#[derive(Debug)]
pub struct Network {
    pub neurons: Vec<Neuron>,
}

impl Network {
    pub fn new() -> Self {
        Self {
            neurons: Vec::new(),
        }
    }

    pub fn get_or_create_neuron(
        &mut self,
        id: usize,
        neuron_type: Type,
    ) -> &mut Neuron {
        // search fo the neuron 
        if let Some(index) = self.neurons.iter().position(|neuron| neuron.id == id) {
            return &mut self.neurons[index];
        }

        // if not found create anew neuron
        let new_neuron = Neuron::new(id, neuron_type);
        self.neurons.push(new_neuron);

        // return the new neuron 
        let index = self.neurons.len() - 1;
        &mut self.neurons[index]
    }

    pub fn neuron_exists(&self, id: usize) -> bool {
        self.neurons.iter().any(|neuron| neuron.id == id)
    }
}

