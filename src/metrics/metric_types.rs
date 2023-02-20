use nom::{
    branch::alt,
    bytes::complete::{is_a, tag},
    character::complete::{multispace1, one_of, space0},
    combinator::{map, not, opt},
    error::ParseError,
    number::complete::double,
    sequence::{preceded, terminated, tuple},
    IResult,
};
use serde::Serialize;

use super::{nom_err, parse_labelset, Label, LabelSet, MetricPoint, Span, METRIC_NAME_CHARS};

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
    InfoValue(IntOrFloat),
    SummaryValue(SummaryValue),
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
            MetricValue::InfoValue(x) => x.serialize(serializer),
            MetricValue::SummaryValue(x) => x.serialize(serializer),
        }
    }
}

pub fn parse_gauge_metric<'a, E: ParseError<Span<'a>>>(
    i: Span<'a>,
    name: &str,
) -> IResult<Span<'a>, (LabelSet, MetricPoint), E> {
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

pub fn parse_info_metric<'a, E: ParseError<Span<'a>>>(
    i: Span<'a>,
    name: &str,
) -> IResult<Span<'a>, (LabelSet, MetricPoint), E> {
    preceded(
        tuple((tag(name), tag("_info"))),
        map(
            tuple((terminated(parse_labelset, space0), parse_int_or_float)),
            |(labels, value)| (labels, MetricValue::InfoValue(value).into()),
        ),
    )(i)
}

pub fn parse_counter_metric<'a, E: ParseError<Span<'a>>>(
    i: Span<'a>,
    name: &str,
) -> IResult<Span<'a>, (LabelSet, MetricPoint), E> {
    let mut labels: Option<LabelSet> = None;
    let mut total: Option<IntOrFloat> = None;
    let mut created: Option<IntOrFloat> = None;
    let mut timestamp: Option<IntOrFloat> = None;
    let mut i = i;
    while let Ok((i_, (field_labels, value, field_timestamp))) =
        parse_counter_metric_line::<E>(i, name)
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
                    None => LabelSet::new(),
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

fn parse_counter_metric_line<'a, E: ParseError<Span<'a>>>(
    i: Span<'a>,
    name: &str,
) -> IResult<Span<'a>, (LabelSet, CounterField, Option<IntOrFloat>), E> {
    preceded(
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
}

pub fn parse_unknown_metric_with_name<'a, E: ParseError<Span<'a>>>(
    i: Span<'a>,
    name: &str,
) -> IResult<Span<'a>, (LabelSet, MetricPoint), E> {
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
) -> IResult<Span<'a>, (Span<'a>, LabelSet, MetricPoint), E> {
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
            terminated(nom::character::complete::i64, not(one_of(".e"))),
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

#[derive(Debug, PartialEq, Clone, Serialize)]
pub struct SummaryValue {
    sum: Option<IntOrFloat>,
    count: Option<u64>,
    created: Option<IntOrFloat>,
    quantiles: Vec<Quantile>,
}

impl Into<MetricValue> for SummaryValue {
    fn into(self) -> MetricValue {
        MetricValue::SummaryValue(self)
    }
}

#[derive(Debug, PartialEq, Clone, Serialize)]
pub struct Quantile {
    quantile: f64,
    value: f64,
}

#[derive(Debug, PartialEq)]
enum SummaryField {
    Sum(IntOrFloat),
    Count(u64),
    Created(IntOrFloat),
    Quantile(Quantile),
}

#[derive(Debug, PartialEq, Clone)]
enum SummaryValueType {
    IntOrFloat(IntOrFloat),
    U64(u64),
    F64(f64),
}

pub fn parse_summary_metric<'a, E: ParseError<Span<'a>>>(
    i: Span<'a>,
    name: &str,
) -> IResult<Span<'a>, (LabelSet, MetricPoint), E> {
    let mut labels: Option<LabelSet> = None;
    let mut summary_value = SummaryValue {
        count: None,
        sum: None,
        created: None,
        quantiles: vec![],
    };
    let mut timestamp: Option<IntOrFloat> = None;
    let mut i = i;
    while let Ok((i_, (field_labels, value, field_timestamp))) =
        parse_summary_metric_line::<E>(i, name)
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
            SummaryField::Created(x) => summary_value.created = Some(x),
            SummaryField::Count(x) => summary_value.count = Some(x),
            SummaryField::Sum(x) => summary_value.sum = Some(x),
            SummaryField::Quantile(x) => summary_value.quantiles.push(x),
        }
    }
    if !(matches!(
        summary_value,
        SummaryValue {
            count: None,
            sum: None,
            created: None,
            quantiles: _
        }
    ) && summary_value.quantiles.len() == 0)
    {
        Ok((
            i,
            (
                match labels {
                    Some(labels) => labels,
                    None => LabelSet::new(),
                },
                MetricPoint {
                    value: summary_value.into(),
                    timestamp,
                },
            ),
        ))
    } else {
        Err(nom_err(i))
    }
}

fn parse_summary_metric_line<'a, E: ParseError<Span<'a>>>(
    i: Span<'a>,
    name: &str,
) -> IResult<Span<'a>, (LabelSet, SummaryField, Option<IntOrFloat>), E> {
    let (i, (field_type, labelset)) = preceded(
        tag(name),
        tuple((
            alt((
                map(tag("_count"), |_| SummaryField::Count(0)),
                map(tag("_created"), |_| SummaryField::Created(0.into())),
                map(tag("_sum"), |_| SummaryField::Sum(0.into())),
                map(tag(""), |_| {
                    SummaryField::Quantile(Quantile {
                        quantile: 0.0,
                        value: 0.0,
                    })
                }),
            )),
            terminated(parse_labelset::<E>, space0),
        )),
    )(i)?;
    println!("Parsing: {field_type:?}");

    let (i, value) = match field_type {
        SummaryField::Count(_) => terminated(
            map(nom::character::complete::u64, {
                |value| SummaryValueType::U64(value)
            }),
            opt(is_a(".0")),
        )(i)?,
        SummaryField::Created(_) | SummaryField::Sum(_) => map(parse_int_or_float, {
            |value| SummaryValueType::IntOrFloat(value)
        })(i)?,
        _ => map(double, |value| SummaryValueType::F64(value))(i)?,
    };
    println!("Parsed value {value:?}");
    println!("{i}");
    let (i, timestamp) = terminated(parse_timestamp, multispace1)(i)?;
    println!("After");
    println!("{i}");

    let field = match (field_type, value) {
        (SummaryField::Count(_), SummaryValueType::U64(x)) => Ok(SummaryField::Count(x)),
        (SummaryField::Created(_), SummaryValueType::IntOrFloat(x)) => Ok(SummaryField::Created(x)),
        (SummaryField::Sum(_), SummaryValueType::IntOrFloat(x)) => Ok(SummaryField::Sum(x)),
        (SummaryField::Quantile(_), SummaryValueType::F64(x)) => {
            let quantile: &Label = labelset
                .labels
                .iter()
                .find(|label| &label.name == "quantile")
                .ok_or_else(|| nom_err(i))?;
            println!("Label {quantile:?}");
            let value: Result<f64, _> = quantile.value.parse();
            println!("Parsed: {value:?}");
            Ok(SummaryField::Quantile(Quantile {
                quantile: quantile.value.parse().map_err(|err| nom_err(i))?,
                value: x,
            }))
        }
        (_, _) => Err(nom_err(i)),
    }?;
    Ok((i, (labelset, field, timestamp)))
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
            (
                LabelSet::new(),
                MetricValue::GaugeValue(IntOrFloat::Int(100)).into()
            )
        );
    }

    #[test]
    fn parse_gauge_metric_with_timestamp() {
        let src = "foo_seconds 100 1520879607.789\nt";
        let (_, metric) = parse_gauge_metric::<ErrorTree<Span>>(src.into(), "foo_seconds").unwrap();
        assert_eq!(
            metric,
            (
                LabelSet::new(),
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
                LabelSet::new(),
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
                LabelSet::new(),
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
                LabelSet::new(),
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
                }]
                .into(),
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
                }]
                .into(),
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
                }]
                .into(),
                MetricValue::GaugeValue(IntOrFloat::Int(100)).into()
            )
        );
    }

    #[test]
    fn parse_info_metric_test() {
        let src = "foo_info{name=\"pretty name\",version=\"8.3.7\"} 1\n";
        let (_, metric) = parse_info_metric::<ErrorTree<Span>>(src.into(), "foo").unwrap();
        assert_eq!(
            metric,
            (
                vec![
                    Label {
                        name: "name".to_string(),
                        value: "pretty name".to_string()
                    },
                    Label {
                        name: "version".to_string(),
                        value: "8.3.7".to_string()
                    }
                ]
                .into(),
                MetricValue::InfoValue(IntOrFloat::Int(1)).into()
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
            }]
            .into(),
        );
        assert_eq!(
            metric,
            MetricValue::UnknownValue(IntOrFloat::Int(100)).into()
        );
        assert_eq!(name, "foo_seconds".into());
    }

    #[test]
    fn parse_summary_metric_line_count() {
        let src = "foo_count{label=\"value\"} 100 123\n";
        let (_, result) = parse_summary_metric_line::<ErrorTree<Span>>(src.into(), "foo").unwrap();
        assert_eq!(
            result,
            (
                vec![Label {
                    name: "label".into(),
                    value: "value".into()
                }]
                .into(),
                SummaryField::Count(100),
                Some(123.into()),
            )
        );
    }

    #[test]
    fn parse_summary_metric_line_sum() {
        let src = "foo_sum{label=\"value\"} 100\n";
        let (_, result) = parse_summary_metric_line::<ErrorTree<Span>>(src.into(), "foo").unwrap();
        assert_eq!(
            result,
            (
                vec![Label {
                    name: "label".into(),
                    value: "value".into()
                }]
                .into(),
                SummaryField::Sum(100.into()),
                None,
            )
        );
    }

    #[test]
    fn parse_summary_metric_line_created() {
        let src = "foo_created{label=\"value\"} 100\n";
        let (_, result) = parse_summary_metric_line::<ErrorTree<Span>>(src.into(), "foo").unwrap();
        assert_eq!(
            result,
            (
                vec![Label {
                    name: "label".into(),
                    value: "value".into()
                }]
                .into(),
                SummaryField::Created(100.into()),
                None,
            )
        );
    }

    #[test]
    fn parse_summary_metric_test() {
        let src = "acme_http_router_request_seconds_sum{path=\"/api/v1\",method=\"GET\"} 9036.32\n\
acme_http_router_request_seconds_count{path=\"/api/v1\",method=\"GET\"} 807283.0\n\
acme_http_router_request_seconds_created{path=\"/api/v1\",method=\"GET\"} 1605281325.0\n";
        let (_, result) =
            parse_summary_metric::<ErrorTree<Span>>(src.into(), "acme_http_router_request_seconds")
                .unwrap();
        assert_eq!(
            result,
            (
                vec![
                    Label {
                        name: "path".into(),
                        value: "/api/v1".into()
                    },
                    Label {
                        name: "method".into(),
                        value: "GET".into()
                    }
                ]
                .into(),
                MetricPoint {
                    value: MetricValue::SummaryValue(SummaryValue {
                        sum: Some(9036.32.into()),
                        count: Some(807283),
                        created: Some(1605281325.0.into()),
                        quantiles: vec![]
                    }),
                    timestamp: None,
                }
            )
        )
    }

    #[test]
    fn parse_summary_metric_with_quantile() {
        let src = "acme_http_router_request_seconds_sum{path=\"/api/v1\",method=\"GET\"} 9036.32\n\
acme_http_router_request_seconds{path=\"/api/v1\",method=\"GET\",quantile=\"0.50\"} 123.0\n\
acme_http_router_request_seconds{path=\"/api/v1\",method=\"GET\",quantile=\"0.90\"} 234.0\n";
        let (_, result) =
            parse_summary_metric::<ErrorTree<Span>>(src.into(), "acme_http_router_request_seconds")
                .unwrap();
        assert_eq!(
            result,
            (
                vec![
                    Label {
                        name: "path".into(),
                        value: "/api/v1".into()
                    },
                    Label {
                        name: "method".into(),
                        value: "GET".into()
                    }
                ]
                .into(),
                MetricPoint {
                    value: MetricValue::SummaryValue(SummaryValue {
                        sum: Some(9036.32.into()),
                        count: None,
                        created: None,
                        quantiles: vec![
                            Quantile {
                                quantile: 0.5,
                                value: 123.0
                            },
                            Quantile {
                                quantile: 0.9,
                                value: 234.0
                            }
                        ]
                    }),
                    timestamp: None,
                }
            )
        )
    }
}
