use crate::neuron::Network;
use std::thread;
use std::time::{Duration, Instant};

mod learning;
mod neuron;

fn main() {
    let mut network = Network::new();

    let total_neurons = 120;
    // divides all neurons
    let (input_ids, hidden_ids, output_ids) = network.populate(total_neurons);

    println!(
        "Created {} neurons -> {} input, {} hidden, {} output",
        total_neurons,
        input_ids.len(),
        hidden_ids.len(),
        output_ids.len()
    );
    println!("Input ids:  {:?}", input_ids);
    println!("Hidden ids: {:?}", hidden_ids);
    println!("Output ids: {:?}", output_ids);

    let input_to_hidden = network.connect_layers(&input_ids, &hidden_ids, 0.8);
    let hidden_to_output = network.connect_layers(&hidden_ids, &output_ids, 0.9);
    let hidden_to_hidden = network.connect_layers(&hidden_ids, &hidden_ids, 0.5);
    let output_to_hidden = network.connect_layers(&output_ids, &hidden_ids, 0.3);

    println!(
        "Connections made: {} input->hidden, {} hidden->output, {} hidden->hidden, {} output->hidden",
        input_to_hidden, hidden_to_output, hidden_to_hidden, output_to_hidden
    );
    let first_input = input_ids[0];

    let tick_rate: f32 = 16.0;
    let tick_duration = Duration::from_secs_f32(1.0 / tick_rate);

    let sim_dt: f32 = 0.1;
    let sub_steps_per_tick = ((1.0 / tick_rate) / sim_dt).round() as u32;

    let mut tick: u64 = 0;
    let max_ticks: u64 = 640;

    loop {
        let tick_start = Instant::now();

        // send signel at start for now
        if tick == 0 {
            network.queue_signal(first_input, 20.0);
        }

        let mut fired_this_tick = false;
        for _ in 0..sub_steps_per_tick {
            if network.step(sim_dt) {
                fired_this_tick = true;
            }
        }

        if fired_this_tick {
            println!(
                "[tick {tick}] sim_time={:.2} -> something fired",
                network.time
            );
        }

        tick += 1;
        if tick >= max_ticks {
            break;
        }

        let elapsed = tick_start.elapsed();
        if elapsed < tick_duration {
            thread::sleep(tick_duration - elapsed);
        }
    }

    println!("sim time: {:.2}", network.time);


    if let Some(&from_id) = input_ids.first() {
        let real_target = network
            .neurons
            .get(&from_id)
            .and_then(|n| n.connections.first())
            .map(|c| c.target_id);

        match real_target {
            None => println!(
                "Neuron {from_id} has no outgoing connections"
            ),
            Some(to_id) => {
                let to_last_fired = network.neurons.get(&to_id).map(|n| n.last_fired).flatten();

                if let Some(from_neuron) = network.neurons.get_mut(&from_id) {
                    let did_train = learning::train(from_neuron, to_id, to_last_fired, 5.0, 0.1);
                    println!(
                        "{from_id} -> {to_id}: train() adjusted a weight: {did_train}"
                    );
                }
            }
        }
    }
}