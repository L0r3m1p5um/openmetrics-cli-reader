use client::{create_client, get_metricset};

mod client;
mod metrics;

#[tokio::main]
async fn main() -> color_eyre::Result<()> {
    let client = create_client()?;
    let metricset = get_metricset("http://localhost:9000/multiple_counters", client).await?;
    println!("{}", serde_json::to_string(&metricset)?);
    Ok(())
}
