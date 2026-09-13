/*
    
    * 2 main techniques can be used here

    1. location stimulation: in a input processing part of the brain there nodes are structured into a building.
    can be cause this to our advantage by stimulation specific floors by the input.

    2. firing rate: location mapping is not enough because there are too many cases where 2 different inputs can be stimulating the same region making our
    brain not able differentiate between 2 inputs.
    to fix this we can add fire rate because on the input so the brain can differentiate the inputs its given
 */


pub enum Signal {
    NumbersGuess(i32,f64),
}

pub fn electrical_signals(signal: Signal) {
    match signal {
        Signal::NumbersGuess(n,m) => numbers_guess(n,m),
    }
}

/*
    About this function:
        this function takes in a f32 and convert the number into electrical signals.
        its uses log 10 scale to squish the number between 0 to 1
        this function uses both techniques to infuse the signal in to the brain
*/
pub fn numbers_guess(input: i32, total_electrodes: f64) {
    // id the number is negative
    let is_negative = input < 0;
    let abs_input = input.unsigned_abs();

    // digit length
    let digit_len = match abs_input.checked_ilog10() {
        Some(n) => n + 1,
        None => 1,
    };

    // logarithmic scale 1 to 1 billion
    let float_val = (abs_input as f64).max(1.0);
    let scale = (float_val.log10() / 9.0).clamp(0.0, 1.0);

    // normalized location 0.0 to 1.0
    let normalized_location = scale;

    // physical electrode index
    let max_index = (total_electrodes - 1.0).max(0.0);
    let physical_location = (scale * max_index).round() as u32;

    // firing rate
    let firing_rate_hz = 10.0 + (scale * 240.0);

    println!(
        "input: {} (neg: {}) , digits: {} , normalized location: {:.4} , Physical Index: {}/{} , rate: {:.1}",
        input,
        is_negative,
        digit_len,
        normalized_location,
        physical_location,
        total_electrodes as u32,
        firing_rate_hz
    );
}