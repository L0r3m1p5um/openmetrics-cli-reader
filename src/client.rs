use std::{error::Error, fmt::Display, sync::Arc};

use nom_supreme::{error::ErrorTree, final_parser::final_parser};
use reqwest::Client;
use tokio::{
    task::JoinSet,
    time::{sleep, Duration},
};

use crate::metrics::{parse_metric_set, render_error, Label, MetricSet, Span};

pub async fn get_metricset(url: Arc<String>, client: Arc<Client>) -> color_eyre::Result<MetricSet> {
    let (label, url) = split_url(&url);
    let response = client.get(url).send().await?.text().await?;
    let src = response.clone();
    let metric_set = final_parser(|it| {
        parse_metric_set::<ErrorTree<Span>>(
            it,
            &Some(Label {
                name: "metrics_source".into(),
                value: label.to_string(),
            }),
        )
    })(response.as_str().into())
    .or_else(|e| {
        render_error(&src, e);
        Err(MetricsParseError {})
    })?;
    Ok(metric_set)
}

fn split_url(url: &str) -> (&str, &str) {
    match url.split_once(":=") {
        Some((label, url)) => (label, url),
        None => (url, url),
    }
}

pub async fn merge_metricsets(
    urls: &Vec<Arc<String>>,
    client: Arc<Client>,
) -> color_eyre::Result<MetricSet> {
    let mut handles = JoinSet::new();
    for url in urls {
        handles.spawn(get_metricset(url.clone(), client.clone()));
    }
    let mut metric_families = vec![];
    while let Some(res) = handles.join_next().await {
        let mut metrics = res??.metric_families;
        metric_families.append(&mut metrics);
    }
    Ok(MetricSet { metric_families })
}

pub async fn print_metrics(
    urls: Vec<String>,
    interval: Option<Duration>,
) -> color_eyre::Result<()> {
    let client = Arc::new(create_client()?);
    let urls: Vec<Arc<String>> = urls.into_iter().map(|url| Arc::new(url)).collect();
    match interval {
        None => println!(
            "{}",
            serde_json::to_string(&merge_metricsets(&urls.clone(), client.clone()).await?)?
        ),
        Some(interval) => loop {
            println!(
                "{}",
                serde_json::to_string(&merge_metricsets(&urls.clone(), client.clone()).await?)?
            );
            sleep(interval).await;
        },
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
        reqwest::header::ACCEPT,
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
