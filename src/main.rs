use client::{create_client, get_metricset};
use clap::Parser;

mod client;
mod metrics;

#[tokio::main]
async fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    let args = Cli::parse();
    let client = create_client()?;
    let metricset = get_metricset(&args.url, client).await?;
    println!("{}", serde_json::to_string(&metricset)?);
    Ok(())
}

#[derive(Parser)]
struct Cli {
    url: String,
}