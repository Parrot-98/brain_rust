use crate::neuron::Neuron;

const TAU_PLUS: f32 = 20.0;
const TAU_MINUS: f32 = 20.0;
const A_PLUS: f32 = 0.1;
const A_MINUS: f32 = 0.12;
const WINDOW: f32 = 50.0;

// STDP: strengthens or weakens the synapse's eligibility based on spike timing
pub fn train(neuron_1: &mut Neuron, target_id: usize, target_last_fired: Option<f32>) -> bool {
    let (Some(t_pre), Some(t_post)) = (neuron_1.last_fired, target_last_fired) else {
        return false;
    };

    let delta_t = t_post - t_pre;

    if delta_t == 0.0 || delta_t.abs() > WINDOW {
        return false;
    }

    let weight_change = if delta_t > 0.0 {
        A_PLUS * (-delta_t / TAU_PLUS).exp()
    } else {
        -A_MINUS * (delta_t / TAU_MINUS).exp()
    };

    let Some(connection) = neuron_1
        .connections
        .iter_mut()
        .find(|c| c.target_id == target_id)
    else {
        return false;
    };

    connection.accumulate_eligibility(weight_change);
    true
}