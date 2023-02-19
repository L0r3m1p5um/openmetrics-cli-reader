use clap::Parser;
use client::print_metrics;
use tokio::time::Duration;

mod client;
mod metrics;

#[tokio::main]
async fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    let args = Cli::parse();

    print_metrics(
        &args.url,
        match args.interval {
            Some(secs) => Some(Duration::from_secs(secs)),
            None => None,
        },
    )
    .await?;
    Ok(())
}

#[derive(Parser)]
struct Cli {
    url: String,

    #[clap(short, long)]
    interval: Option<u64>,
}
