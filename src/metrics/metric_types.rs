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

use super::{nom_err, parse_labelset, Label, Metric, MetricPoint, Span, METRIC_NAME_CHARS};


#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
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
    CounterValue(CounterValue),
    // HistogramValue,
    // StateSetValue,
    // InfoValue,
    // SummaryValue,
}

impl Serialize for MetricValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            MetricValue::GaugeValue(x) => x.serialize(serializer),
            MetricValue::UnknownValue(x) => x.serialize(serializer),
            MetricValue::CounterValue(x) => x.serialize(serializer),
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

pub fn parse_counter_metric<'a, E: ParseError<Span<'a>>>(
    i: Span<'a>,
    name: &str,
) -> IResult<Span<'a>, (Vec<Label>, MetricPoint), E> {
    let mut labels: Option<Vec<Label>> = None;
    let mut total: Option<IntOrFloat> = None;
    let mut created: Option<IntOrFloat> = None;
    let mut timestamp: Option<IntOrFloat> = None;
    let mut i = i;
    while let Ok((i_, (field_labels, value, field_timestamp))) = preceded(
        tag(name),
        map(
            tuple((
                alt((
                    map(tag("_total"), |_| CounterField::Total(0.into())),
                    map(tag("_created"), |_| CounterField::Created(0.into())),
                )),
                terminated(parse_labelset::<E>, space0),
                parse_int_or_float,
                parse_timestamp,
                multispace1,
            )),
            |(field_type, labels, value, field_timestamp, _)| {
                (
                    labels,
                    match field_type {
                        CounterField::Total(_) => CounterField::Total(value),
                        CounterField::Created(_) => CounterField::Created(value),
                    },
                    field_timestamp,
                )
            },
        ),
    )(i)
    {
        match labels {
            None => labels = Some(field_labels),
            Some(ref x) => {
                if x != &field_labels {
                    break;
                }
            }
        }
        match timestamp {
            None => timestamp = field_timestamp,
            x => {
                if x != field_timestamp {
                    break;
                }
            }
        }
        i = i_;
        match value {
            CounterField::Created(x) => created = Some(x),
            CounterField::Total(x) => total = Some(x),
        }
    }

    match total {
        Some(total) => Ok((
            i,
            (
                match labels {
                    Some(labels) => labels,
                    None => vec![],
                },
                MetricPoint::new(
                    MetricValue::CounterValue(CounterValue { total, created }),
                    timestamp,
                ),
            ),
        )),
        None => Err(nom_err(i)),
    }
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

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum IntOrFloat {
    Int(i64),
    Float(f64),
}

impl From<i64> for IntOrFloat {
    fn from(value: i64) -> Self {
        IntOrFloat::Int(value)
    }
}

impl From<f64> for IntOrFloat {
    fn from(value: f64) -> Self {
        IntOrFloat::Float(value)
    }
}

impl Serialize for IntOrFloat {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
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
            |value| value.into(),
        ),
        map(double, |value| value.into()),
    ))(i)
}

pub fn parse_timestamp<'a, E: ParseError<Span<'a>>>(
    i: Span<'a>,
) -> IResult<Span<'a>, Option<IntOrFloat>, E> {
    opt(preceded(tag(" "), parse_int_or_float))(i)
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CounterValue {
    total: IntOrFloat,
    created: Option<IntOrFloat>,
}

impl CounterValue {
    pub fn new(total: IntOrFloat, created: Option<IntOrFloat>) -> Self {
        CounterValue { total, created }
    }
}

impl From<CounterValue> for MetricValue {
    fn from(value: CounterValue) -> Self {
        MetricValue::CounterValue(value)
    }
}

enum CounterField {
    Total(IntOrFloat),
    Created(IntOrFloat),
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
                    Some(1520879607.789.into())
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
    fn parse_counter_metric_test() {
        let src = "foo_seconds_total 100\nt";
        let (_, metric) =
            parse_counter_metric::<ErrorTree<Span>>(src.into(), "foo_seconds").unwrap();
        assert_eq!(
            metric,
            (
                vec![],
                MetricPoint {
                    value: MetricValue::CounterValue(CounterValue {
                        total: 100.into(),
                        created: None
                    }),
                    timestamp: None
                }
            )
        );
    }

    #[test]
    fn parse_counter_metric_with_timestamp() {
        let src = "foo_seconds_total 100 123\nfoo_seconds_total 200 456\nt";
        let (_, metric) =
            parse_counter_metric::<ErrorTree<Span>>(src.into(), "foo_seconds").unwrap();
        assert_eq!(
            metric,
            (
                vec![],
                MetricPoint {
                    value: MetricValue::CounterValue(CounterValue {
                        total: 100.into(),
                        created: None
                    }),
                    timestamp: Some(123.into())
                }
            )
        );
    }

    #[test]
    fn parse_counter_metric_with_created() {
        let src =
            "foo_seconds_total{label=\"value\"} 100\nfoo_seconds_created{label=\"value\"} 123\nt";
        let (_, metric) =
            parse_counter_metric::<ErrorTree<Span>>(src.into(), "foo_seconds").unwrap();
        assert_eq!(
            metric,
            (
                vec![Label {
                    name: "label".into(),
                    value: "value".into()
                }],
                MetricPoint {
                    value: MetricValue::CounterValue(CounterValue {
                        total: 100.into(),
                        created: Some(123.into())
                    }),
                    timestamp: None
                }
            )
        );
    }

    #[test]
    fn parse_counter_metric_with_labels() {
        let src =
            "foo_seconds_total{label=\"value1\"} 100\nfoo_seconds_total{label=\"value2\"} 123\nt";
        let (_, metric) =
            parse_counter_metric::<ErrorTree<Span>>(src.into(), "foo_seconds").unwrap();
        assert_eq!(
            metric,
            (
                vec![Label {
                    name: "label".into(),
                    value: "value1".into()
                }],
                MetricPoint {
                    value: MetricValue::CounterValue(CounterValue {
                        total: 100.into(),
                        created: None
                    }),
                    timestamp: None
                }
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
