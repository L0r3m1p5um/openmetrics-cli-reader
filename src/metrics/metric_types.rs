use nom::{
    branch::alt,
    bytes::{
        complete::is_a,
        streaming::{tag, take_until1},
    },
    character::streaming::{multispace1, one_of, space0, space1},
    combinator::{map, not},
    error::ParseError,
    number::streaming::double,
    sequence::{preceded, terminated, tuple},
    IResult,
};
use serde::Serialize;

use super::{parse_labelset, Metric, MetricPoint, Span, METRIC_NAME_CHARS};

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

pub fn parse_gauge_metric<'a, E: ParseError<Span<'a>>>(
    i: Span<'a>,
    name: &str,
) -> IResult<Span<'a>, Metric, E> {
    preceded(
        tag(name),
        map(
            tuple((terminated(parse_labelset, space0), parse_int_or_float)),
            |(labels, value)| Metric {
                labels,
                metric_points: vec![MetricPoint::GaugeValue(value)],
            },
        ),
    )(i)
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub enum IntOrFloat {
    Int(i64),
    Float(f64),
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
            Metric {
                labels: vec![],
                metric_points: vec![MetricPoint::GaugeValue(IntOrFloat::Int(100))]
            }
        );
    }

    #[test]
    fn parse_gauge_metric_with_float_value() {
        let src = "foo_seconds 100.5\nt";
        let (_, metric) = parse_gauge_metric::<ErrorTree<Span>>(src.into(), "foo_seconds").unwrap();
        assert_eq!(
            metric,
            Metric {
                labels: vec![],
                metric_points: vec![MetricPoint::GaugeValue(IntOrFloat::Float(100.5))]
            }
        );
    }

    #[test]
    fn parse_gauge_metric_with_labels() {
        let src = "foo_seconds{label=\"value\"} 100\n";
        let (_, metric) = parse_gauge_metric::<ErrorTree<Span>>(src.into(), "foo_seconds").unwrap();
        assert_eq!(
            metric,
            Metric {
                labels: vec![Label {
                    name: "label".to_string(),
                    value: "value".to_string()
                }],
                metric_points: vec![MetricPoint::GaugeValue(IntOrFloat::Int(100))]
            }
        );
    }
    
}
