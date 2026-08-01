use crate::neuron::{Network, Neuron, Type};
use std::time::Instant;

mod neuron;

fn main() {
    let deuration = Instant::now();
    let mut network = Network::new();

    let mut neuron = Neuron::new(1, Type::Input);

    println!("Before signal: {}", neuron.voltage);

    neuron.receive(0, 20.0);

    println!("After signal: {}", neuron.voltage);

    let fired = neuron.fire();

    println!("Fired: {:?}", fired);
    println!("Final voltage: {}", neuron.voltage);

    network.add_neuron(neuron);

    println!("time taken: {:?}ms", deuration.elapsed().as_micros());

    println!("{network:#?}");
}