use std::fs::File;
use std::io::BufReader;

pub fn json_data() -> Result<Vec<u64>, Box<dyn std::error::Error>> {
    let file = File::open("C:/main files/Desktop/brain_rust/data.json")?;
    let reader = BufReader::new(file);

    let numbers: Vec<u64> = serde_json::from_reader(reader)?;

    Ok(numbers)
}