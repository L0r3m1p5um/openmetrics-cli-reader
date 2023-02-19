use std::{error::Error, fmt::Display};

use nom_supreme::{error::ErrorTree, final_parser::final_parser};
use reqwest::Client;
use tokio::time::{sleep, Duration};

use crate::metrics::{parse_metric_set, render_error, MetricSet, Span};

pub async fn get_metricset(url: &str, client: & Client) -> color_eyre::Result<MetricSet> {
    let response = client.get(url).send().await?.text().await?;
    let src = response.clone();
    let metric_set =
        final_parser(|it| parse_metric_set::<ErrorTree<Span>>(it))(response.as_str().into())
            .or_else(|e| {
                render_error(&src, e);
                Err(MetricsParseError {})
            })?;
    Ok(metric_set)
}

pub async fn print_metrics(url: &str, interval: Option<Duration>) -> color_eyre::Result<()> {
    let client = create_client()?;
    match interval {
        None => println!("{}", serde_json::to_string(&get_metricset(url, &client).await?)?),
        Some(interval) => {
            loop {
                println!("{}", serde_json::to_string(&get_metricset(url, &client).await?)?);
                sleep(interval).await;
            }
        }
    }
    Ok(())
}

#[derive(Debug)]
pub struct MetricsParseError {}

impl Display for MetricsParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Failed to parse metrics")
    }
}

impl Error for MetricsParseError {}

pub fn create_client() -> Result<Client, reqwest::Error> {
    let mut headers = reqwest::header::HeaderMap::new();
    headers.append(
        reqwest::header::CONTENT_TYPE,
        "application/openmetrics-text; version=1.0.0; charset=utf-8"
            .parse()
            .unwrap(),
    );
    let client = reqwest::ClientBuilder::new();
    client.default_headers(headers).build()
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_create_client() {
        let _client = create_client().unwrap();
    }
}
