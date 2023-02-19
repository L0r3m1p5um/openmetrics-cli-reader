use chrono::{DateTime, Utc};
use nom::{
    branch::alt,
    bytes::{
        complete::is_a,
        streaming::{tag, take_until1},
    },
    character::streaming::{multispace1, one_of, space0, space1},
    combinator::{map, not, opt},
    error::ParseError,
    number::streaming::double,
    sequence::{preceded, terminated, tuple},
    IResult,
};
use serde::Serialize;

use super::{parse_labelset, Label, Metric, MetricPoint, Span, METRIC_NAME_CHARS};

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub enum MetricType {
    Unknown,
    Gauge,
    Counter,
    StateSet,
    Info,
    Histogram,
    GagueHistogram,
    Summary,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MetricValue {
    UnknownValue(IntOrFloat),
    GaugeValue(IntOrFloat),
    // CounterValue(CounterValue),
    // HistogramValue,
    // StateSetValue,
    // InfoValue,
    // SummaryValue,
}

impl Serialize for MetricValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer {
            match self {
                MetricValue::GaugeValue(x) => x.serialize(serializer),
                MetricValue::UnknownValue(x) => x.serialize(serializer),
            }
    }
}

pub fn parse_gauge_metric<'a, E: ParseError<Span<'a>>>(
    i: Span<'a>,
    name: &str,
) -> IResult<Span<'a>, (Vec<Label>, MetricPoint), E> {
    preceded(
        tag(name),
        map(
            tuple((
                terminated(parse_labelset, space0),
                parse_int_or_float,
                parse_timestamp,
            )),
            |(labels, value, timestamp)| {
                (
                    labels,
                    MetricPoint::new(MetricValue::GaugeValue(value), timestamp),
                )
            },
        ),
    )(i)
}

pub fn parse_unknown_metric_with_name<'a, E: ParseError<Span<'a>>>(
    i: Span<'a>,
    name: &str,
) -> IResult<Span<'a>, (Vec<Label>, MetricPoint), E> {
    map(
        tuple((
            tag(name),
            terminated(parse_labelset, space0),
            parse_int_or_float,
            parse_timestamp,
        )),
        |(_, labels, value, timestamp)| {
            (
                labels,
                MetricPoint::new(MetricValue::UnknownValue(value), timestamp),
            )
        },
    )(i)
}

pub fn parse_unknown_metric_without_name<'a, E: ParseError<Span<'a>>>(
    i: Span<'a>,
) -> IResult<Span<'a>, (Span<'a>, Vec<Label>, MetricPoint), E> {
    map(
        tuple((
            is_a(METRIC_NAME_CHARS),
            terminated(parse_labelset, space0),
            parse_int_or_float,
            parse_timestamp,
        )),
        |(name, labels, value, timestamp)| {
            (
                name,
                labels,
                MetricPoint::new(MetricValue::UnknownValue(value), timestamp),
            )
        },
    )(i)
}

#[derive(Debug, Copy, Clone,  PartialEq)]
pub enum IntOrFloat {
    Int(i64),
    Float(f64),
}

impl Serialize for IntOrFloat {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer {
        match self {
            IntOrFloat::Float(x) => x.serialize(serializer),
            IntOrFloat::Int(x) => x.serialize(serializer),
        }
    }
}

fn parse_int_or_float<'a, E: ParseError<Span<'a>>>(
    i: Span<'a>,
) -> IResult<Span<'a>, IntOrFloat, E> {
    alt((
        map(
            terminated(nom::character::streaming::i64, not(one_of(".e"))),
            |value| IntOrFloat::Int(value),
        ),
        map(double, |value| IntOrFloat::Float(value)),
    ))(i)
}

pub fn parse_timestamp<'a, E: ParseError<Span<'a>>>(
    i: Span<'a>,
) -> IResult<Span<'a>, Option<f64>, E> {
    opt(preceded(tag(" "), double))(i)
}

#[cfg(test)]
mod test {

    use crate::metrics::{parse_metric, render_error, Label};

    use super::*;

    use nom_supreme::{error::ErrorTree, final_parser::final_parser};

    #[test]
    fn parse_int_to_int_or_float() {
        let (_, value) = parse_int_or_float::<ErrorTree<Span>>("156\nA".into()).unwrap();
        assert_eq!(value, IntOrFloat::Int(156))
    }

    #[test]
    fn parse_negative_int_to_int_or_float() {
        let (_, value) = parse_int_or_float::<ErrorTree<Span>>("-156\nA".into()).unwrap();
        assert_eq!(value, IntOrFloat::Int(-156))
    }

    #[test]
    fn parse_float_to_int_or_float() {
        let (_, value) = parse_int_or_float::<ErrorTree<Span>>("1.5\nA".into()).unwrap();
        assert_eq!(value, IntOrFloat::Float(1.5))
    }

    #[test]
    fn parse_negative_float_to_int_or_float() {
        let (_, value) = parse_int_or_float::<ErrorTree<Span>>("-1.5\nA".into()).unwrap();
        assert_eq!(value, IntOrFloat::Float(-1.5))
    }

    #[test]
    fn parse_scientific_notation_to_int_or_float() {
        let (_, value) = parse_int_or_float::<ErrorTree<Span>>("1.89e-7\nA".into()).unwrap();
        assert_eq!(value, IntOrFloat::Float(0.000000189))
    }

    #[test]
    fn parse_gauge_metric_without_labels() {
        let src = "foo_seconds 100\nt";
        let (_, metric) = parse_gauge_metric::<ErrorTree<Span>>(src.into(), "foo_seconds").unwrap();
        assert_eq!(
            metric,
            (vec![], MetricValue::GaugeValue(IntOrFloat::Int(100)).into())
        );
    }

    #[test]
    fn parse_gauge_metric_with_timestamp() {
        let src = "foo_seconds 100 1520879607.789\nt";
        let (_, metric) = parse_gauge_metric::<ErrorTree<Span>>(src.into(), "foo_seconds").unwrap();
        assert_eq!(
            metric,
            (
                vec![],
                MetricPoint::new(
                    MetricValue::GaugeValue(IntOrFloat::Int(100)).into(),
                    Some(1520879607.789)
                )
            )
        );
    }

    #[test]
    fn parse_gauge_metric_with_float_value() {
        let src = "foo_seconds 100.5\nt";
        let (_, metric) = parse_gauge_metric::<ErrorTree<Span>>(src.into(), "foo_seconds").unwrap();
        assert_eq!(
            metric,
            (
                vec![],
                MetricValue::GaugeValue(IntOrFloat::Float(100.5)).into()
            )
        );
    }

    #[test]
    fn parse_gauge_metric_with_labels() {
        let src = "foo_seconds{label=\"value\"} 100\n";
        let (_, metric) = parse_gauge_metric::<ErrorTree<Span>>(src.into(), "foo_seconds").unwrap();
        assert_eq!(
            metric,
            (
                vec![Label {
                    name: "label".to_string(),
                    value: "value".to_string()
                }],
                MetricValue::GaugeValue(IntOrFloat::Int(100)).into()
            )
        );
    }

    #[test]
    fn parse_unknown_metric_with_labels() {
        let src = "foo_seconds{label=\"value\"} 100\n";
        let (_, (name, labels, metric)) =
            parse_unknown_metric_without_name::<ErrorTree<Span>>(src.into()).unwrap();
        assert_eq!(
            labels,
            vec![Label {
                name: "label".to_string(),
                value: "value".to_string()
            }],
        );
        assert_eq!(
            metric,
            MetricValue::UnknownValue(IntOrFloat::Int(100)).into()
        );
        assert_eq!(name, "foo_seconds".into());
    }
}
