use crate::neuron::Type;
use crate::neuron::Network;


mod neuron;


fn main() {
    let mut network = Network::new();

    let _neuron = network.get_or_create_neuron(10, Type::Input);

    let _neuron = network.get_or_create_neuron(10, Type::Input);

    network.get_or_create_neuron(20, Type::Output);

    println!("Neuron 10 exists: {}", network.neuron_exists(10));
    println!("Neuron 50 exists: {}", network.neuron_exists(50));
    println!("Total neurons: {}", network.neurons.len());
}