use std::{error::Error, fmt::Display};

use nom_supreme::{error::ErrorTree, final_parser::final_parser};

use crate::metrics::{parse_metric_set, render_error, MetricSet, Span};

pub async fn get_metricset(url: &str, client: reqwest::Client) -> Result<MetricSet, ClientError> {
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|err| ClientError::ReqwestError(err))?
        .text()
        .await
        .map_err(|err| ClientError::ReqwestError(err))?;
    let src = response.clone();
    let metric_set =
        final_parser(|it| parse_metric_set::<ErrorTree<Span>>(it))(response.as_str().into())
            .or_else(|e| {
                render_error(&src, e);
                Err(ClientError::ParseError)
            })?;
    Ok(metric_set)
}

#[derive(Debug)]
pub enum ClientError {
    ReqwestError(reqwest::Error),
    ParseError,
}

impl Display for ClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClientError::ReqwestError(err) => write!(f, "{err}"),
            err => write!(f, "{err:?}"),
        }
    }
}

impl Error for ClientError {}

pub fn create_client() -> Result<reqwest::Client, reqwest::Error> {
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
