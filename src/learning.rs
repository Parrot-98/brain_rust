use crate::neuron::Neuron;

pub fn train(
    neuron_1: &mut Neuron,
    target_id: usize,
    target_last_fired: Option<f32>,
    window: f32,
    delta: f32,
) -> bool {
    let (Some(t_pre), Some(t_post)) = (neuron_1.last_fired, target_last_fired) else {
        return false;
    };

    let time_gap = (t_post - t_pre).abs();

    if time_gap > window {
        return false;
    }

    let Some(connection) = neuron_1
        .connections
        .iter_mut()
        .find(|c| c.target_id == target_id)
    else {
        return false;
    };

    connection.adjust_weight(delta);
    true
}