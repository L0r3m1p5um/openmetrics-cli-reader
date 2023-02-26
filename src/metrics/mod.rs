mod metric_types;

use std::{collections::HashMap, error::Error};

use metric_types::*;
use miette::GraphicalReportHandler;
use nom::{
    branch::alt,
    bytes::complete::{escaped, is_a, tag},
    character::{
        complete::space0,
        complete::{
            alphanumeric1, line_ending, multispace0, none_of, not_line_ending, one_of, space1,
        },
    },
    combinator::{map, opt, value},
    error::ParseError,
    multi::separated_list0,
    sequence::{delimited, preceded, terminated, tuple},
    IResult,
};
use nom_locate::LocatedSpan;
use nom_supreme::error::{BaseErrorKind, ErrorTree, GenericErrorTree};
use serde::Serialize;

const METRIC_NAME_CHARS: &str = "abcdefghijklmnopqrstuvwxyz_";

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct MetricSet {
    metric_families: Vec<MetricFamily>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct MetricFamily {
    name: String,
    metric_type: MetricType,
    unit: Option<String>,
    help: Option<String>,
    metrics: Vec<Metric>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Metric {
    labels: HashMap<String, String>,
    metric_points: Vec<MetricPoint>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq, PartialOrd)]
pub struct Label {
    pub name: String,
    pub value: String,
}

#[derive(Debug, Clone, PartialOrd)]
pub struct LabelSet {
    labels: Vec<Label>,
}

impl From<Vec<Label>> for LabelSet {
    fn from(value: Vec<Label>) -> Self {
        LabelSet { labels: value }
    }
}

impl Serialize for LabelSet {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.labels.serialize(serializer)
    }
}

impl PartialEq for LabelSet {
    fn eq(&self, other: &Self) -> bool {
        self.labels
            .iter()
            .filter(|label| label_filter(&label.name))
            .eq(other
                .labels
                .iter()
                .filter(|label| label_filter(&label.name)))
    }
}

impl LabelSet {
    pub fn new() -> Self {
        LabelSet { labels: vec![] }
    }

    pub fn filter_le_and_quantile(self) -> Self {
        self.labels
            .into_iter()
            .filter(|label| label_filter(&label.name))
            .collect::<Vec<Label>>()
            .into()
    }
}

fn label_filter(name: &str) -> bool {
    name != "quantile" && name != "le"
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct MetricPoint {
    value: MetricValue,
    timestamp: Option<IntOrFloat>,
}

impl MetricPoint {
    fn new(value: MetricValue, timestamp: Option<IntOrFloat>) -> Self {
        MetricPoint { value, timestamp }
    }
}

impl From<MetricValue> for MetricPoint {
    fn from(value: MetricValue) -> Self {
        MetricPoint {
            value,
            timestamp: None,
        }
    }
}

pub type Span<'a> = LocatedSpan<&'a str>;

#[derive(thiserror::Error, Debug, miette::Diagnostic)]
#[error("bad input")]
struct BadInput {
    #[source_code]
    src: String,

    #[label("{kind}")]
    bad_bit: miette::SourceSpan,

    kind: BaseErrorKind<&'static str, Box<dyn std::error::Error + Send + Sync>>,
}

pub fn render_error(src_input: &str, e: ErrorTree<Span>) {
    match e {
        GenericErrorTree::Base { location, kind } => {
            render_base_error(src_input, location, kind);
        }
        GenericErrorTree::Stack { base, contexts } => unimplemented!("stack"),
        GenericErrorTree::Alt(x) => {
            for y in x {
                match y {
                    GenericErrorTree::Base { location, kind } => {
                        render_base_error(src_input, location, kind);
                    }
                    _ => unimplemented!(),
                }
            }
        }
    }
}

fn render_base_error(
    src_input: &str,
    location: LocatedSpan<&str>,
    kind: BaseErrorKind<&'static str, Box<dyn Error + Send + Sync>>,
) {
    let offset = location.location_offset().into();
    let err = BadInput {
        src: src_input.to_string(),
        bad_bit: miette::SourceSpan::new(offset, 0.into()),
        kind,
    };
    let mut s = String::new();
    GraphicalReportHandler::new()
        .render_report(&mut s, &err)
        .unwrap();
    eprintln!("{s}");
}

fn parse_label<'a, E: ParseError<Span<'a>>>(i: Span<'a>) -> IResult<Span<'a>, Label, E> {
    let (input, (name, _, value)) = tuple((
        alphanumeric1,
        tag("="),
        delimited(
            tag("\""),
            escaped(none_of("\\\""), '\\', one_of(r#"\""#)),
            tag("\""),
        ),
    ))(i)?;
    Ok((
        input,
        Label {
            name: name.to_string(),
            value: value.to_string(),
        },
    ))
}

fn parse_labelset<'a, E: ParseError<Span<'a>>>(i: Span<'a>) -> IResult<Span<'a>, LabelSet, E> {
    alt((
        map(
            delimited(
                tag("{"),
                separated_list0(tag(","), parse_label::<E>),
                tag("}"),
            ),
            { |labels| labels.into() },
        ),
        map(tag(" "), |_| LabelSet { labels: vec![] }),
    ))(i)
}

pub fn parse_metric_set<'a, E: ParseError<Span<'a>>>(
    i: Span<'a>,
    source_label: &Option<Label>,
) -> IResult<Span<'a>, MetricSet, E> {
    let mut metric_families = vec![];
    let mut i = i;
    while let Ok((input, (metric_family, eof))) = tuple((
        { |it| parse_metric_family::<ErrorTree<Span<'a>>>(it, source_label) },
        opt(tuple((
            tag("# EOF"),
            nom::bytes::complete::take_while(|_| true),
        ))),
    ))(i)
    {
        metric_families.push(metric_family);
        i = input;
        if eof.is_some() {
            break;
        }
    }

    Ok((i, MetricSet { metric_families }))
}

fn parse_metric_family<'a, E: ParseError<Span<'a>>>(
    i: Span<'a>,
    source_label: &Option<Label>,
) -> IResult<Span<'a>, MetricFamily, E> {
    let (mut i, _) = multispace0(i)?;

    let mut metric_family_name: Option<&str> = None;
    let mut metric_family_type: MetricType = MetricType::Unknown;
    let mut metric_family_unit: Option<String> = None;
    let mut metric_family_help: Option<String> = None;

    while let Ok((input, (line_name, metadata))) =
        parse_metric_family_metadata::<ErrorTree<Span<'a>>>(i, metric_family_name)
    {
        i = input;
        if metric_family_name == None {
            metric_family_name = Some(*line_name.fragment());
        }
        match metadata {
            MetricFamilyMetadata::MetricType(metric_type) => {
                metric_family_type = metric_type;
            }
            MetricFamilyMetadata::Unit(unit) => {
                metric_family_unit = Some(unit);
            }
            MetricFamilyMetadata::Help(help) => {
                metric_family_help = Some(help);
            }
        }
    }

    let mut metrics = vec![];
    while let Ok((input, (line_name, metric))) =
        parse_metric::<ErrorTree<Span<'a>>>(i, metric_family_name, source_label, metric_family_type)
    {
        if metric_family_name == None {
            if let Some(name) = line_name {
                metric_family_name = Some(*name.fragment());
            }
        }
        i = input;
        metrics.push(metric);
    }

    match metric_family_name {
        Some(name) => Ok((
            i,
            MetricFamily {
                name: name.to_string(),
                metric_type: metric_family_type,
                unit: metric_family_unit,
                help: metric_family_help,
                metrics,
            },
        )),
        None => Err(nom::Err::Failure(E::from_error_kind(
            i,
            nom::error::ErrorKind::Fail,
        ))),
    }
}

fn parse_metrictype<'a, E: ParseError<Span<'a>>>(i: Span<'a>) -> IResult<Span<'a>, MetricType, E> {
    alt((
        value(MetricType::GaugeHistogram, tag("gaugehistogram")),
        value(MetricType::Gauge, tag("gauge")),
        value(MetricType::Counter, tag("counter")),
        value(MetricType::StateSet, tag("stateset")),
        value(MetricType::Info, tag("info")),
        value(MetricType::Histogram, tag("histogram")),
        value(MetricType::Summary, tag("summary")),
    ))(i)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum MetricFamilyMetadata {
    MetricType(MetricType),
    Unit(String),
    Help(String),
}

fn parse_metric_family_metadata<'a, E: ParseError<Span<'a>>>(
    i: Span<'a>,
    name: Option<&str>,
) -> IResult<Span<'a>, (Span<'a>, MetricFamilyMetadata), E> {
    preceded(
        tuple((tag("#"), space1)),
        alt((
            |it| parse_type_metadata::<E>(it, name),
            |it| parse_unit_metadata::<E>(it, name),
            |it| parse_help_metadata::<E>(it, name),
        )),
    )(i)
}

fn parse_help_metadata<'a, E: ParseError<Span<'a>>>(
    i: Span<'a>,
    name: Option<&str>,
) -> IResult<Span<'a>, (Span<'a>, MetricFamilyMetadata), E> {
    let name_match: Box<
        dyn Fn(
            LocatedSpan<&'a str>,
        ) -> Result<(LocatedSpan<&'a str>, LocatedSpan<&'a str>), nom::Err<E>>,
    > = match name {
        Some(name) => Box::new(tag(name)),
        None => Box::new(is_a(METRIC_NAME_CHARS)),
    };
    preceded(
        tuple((tag("HELP"), space1)),
        tuple((
            name_match,
            delimited(
                space0,
                map(not_line_ending, {
                    |help: Span<'a>| MetricFamilyMetadata::Help(help.to_string())
                }),
                line_ending,
            ),
        )),
    )(i)
}

fn parse_unit_metadata<'a, E: ParseError<Span<'a>>>(
    i: Span<'a>,
    name: Option<&str>,
) -> IResult<Span<'a>, (Span<'a>, MetricFamilyMetadata), E> {
    let name_match: Box<
        dyn Fn(
            LocatedSpan<&'a str>,
        ) -> Result<(LocatedSpan<&'a str>, LocatedSpan<&'a str>), nom::Err<E>>,
    > = match name {
        Some(name) => Box::new(tag(name)),
        None => Box::new(is_a(METRIC_NAME_CHARS)),
    };
    preceded(
        tuple((tag("UNIT"), space1)),
        tuple((
            name_match,
            delimited(
                space0,
                map(not_line_ending, {
                    |unit: Span<'a>| MetricFamilyMetadata::Unit(unit.to_string())
                }),
                line_ending,
            ),
        )),
    )(i)
}

fn parse_type_metadata<'a, E: ParseError<Span<'a>>>(
    i: Span<'a>,
    name: Option<&str>,
) -> IResult<Span<'a>, (Span<'a>, MetricFamilyMetadata), E> {
    let name_match: Box<
        dyn Fn(
            LocatedSpan<&'a str>,
        ) -> Result<(LocatedSpan<&'a str>, LocatedSpan<&'a str>), nom::Err<E>>,
    > = match name {
        Some(name) => Box::new(tag(name)),
        None => Box::new(is_a(METRIC_NAME_CHARS)),
    };
    preceded(
        tuple((tag("TYPE"), space1)),
        tuple((
            name_match,
            delimited(
                space0,
                map(parse_metrictype::<E>, {
                    |mtype| MetricFamilyMetadata::MetricType(mtype)
                }),
                line_ending,
            ),
        )),
    )(i)
}

fn parse_metric<'a, E: ParseError<Span<'a>>>(
    i: Span<'a>,
    name: Option<&str>,
    source_label: &Option<Label>,
    metric_type: MetricType,
) -> IResult<Span<'a>, (Option<Span<'a>>, Metric), E> {
    let mut i = i;
    let mut metric_points = vec![];
    let mut metric_labels: Option<LabelSet> = None;
    let mut metric_name: Option<Span<'a>> = None;

    while let Ok((i_, (name, (labels, metric_point)))) = match name {
        Some(name) => match metric_type {
            MetricType::Unknown => terminated(
                map(
                    |it| parse_unknown_metric_with_name::<E>(it, name),
                    |metric| (None, metric),
                ),
                multispace0,
            )(i),
            MetricType::Gauge => terminated(
                map(|it| parse_gauge_metric(it, name), |metric| (None, metric)),
                multispace0,
            )(i),
            MetricType::Info => terminated(
                map(|it| parse_info_metric(it, name), |metric| (None, metric)),
                multispace0,
            )(i),
            MetricType::Counter => terminated(
                map(|it| parse_counter_metric(it, name), |metric| (None, metric)),
                multispace0,
            )(i),
            MetricType::Summary => terminated(
                map(|it| parse_summary_metric(it, name), |metric| (None, metric)),
                multispace0,
            )(i),
            MetricType::Histogram => terminated(
                map(
                    |it| parse_histogram_metric(it, name),
                    |metric| (None, metric),
                ),
                multispace0,
            )(i),
            MetricType::StateSet => terminated(
                map(
                    |it| parse_stateset_metric(it, name),
                    |metric| (None, metric),
                ),
                multispace0,
            )(i),
            MetricType::GaugeHistogram => terminated(
                map(
                    |it| parse_histogram_metric(it, name),
                    |metric| (None, metric),
                ),
                multispace0,
            )(i),
            mtype => unimplemented!("Parsing has not been implemented for metric type {mtype:?}"),
        },
        None => match metric_type {
            MetricType::Unknown => terminated(
                map(
                    parse_unknown_metric_without_name,
                    |(name, labels, metric)| (Some(name), (labels, metric)),
                ),
                multispace0,
            )(i),
            _ => unimplemented!("Metric type cannot be parsed without a known name"),
        },
    } {
        match metric_labels {
            Some(ref metric_labels) => {
                if labels != *metric_labels {
                    break;
                }
            }
            None => metric_labels = Some(labels),
        }
        i = i_;
        metric_points.push(metric_point);
        metric_name = name;
    }
    let metric_labels = match metric_labels {
        Some(labels) => labels,
        None => LabelSet { labels: vec![] },
    };

    if metric_points.len() > 0 {
        let mut label_map = HashMap::new();
        for label in metric_labels.labels {
            label_map.insert(label.name, label.value);
        }
        if let Some(label) = source_label {
            label_map.insert(label.name.clone(), label.value.clone());
        }

        Ok((
            i,
            (
                metric_name,
                Metric {
                    labels: label_map,
                    metric_points,
                },
            ),
        ))
    } else {
        Err(nom_err(i, None))
    }
}

fn nom_err<'a, E: ParseError<Span<'a>>>(i: Span<'a>, message: Option<&str>) -> nom::Err<E> {
    if let Some(message) = message {
        eprintln!("{message}");
    }
    nom::Err::Error(E::from_error_kind(i, nom::error::ErrorKind::Fail))
}

#[cfg(test)]
mod test {

    use nom::sequence::terminated;
    use nom_supreme::{error::ErrorTree, final_parser::final_parser};

    use super::*;

    #[test]
    fn parse_label_test() {
        let src = "name=\"value\"";
        let label = final_parser(parse_label::<ErrorTree<Span>>)(Span::new(src))
            .or_else(|e| {
                render_error(src, e);
                Err(())
            })
            .unwrap();
        assert_eq!(
            label,
            Label {
                name: "name".to_string(),
                value: "value".to_string()
            }
        );
    }

    #[test]
    fn parse_label_escaped() {
        let src = "name=\"1\\\"2\"";
        let label = final_parser(parse_label::<ErrorTree<Span>>)(Span::new(src))
            .or_else(|e| {
                render_error(src, e);
                Err(())
            })
            .unwrap();
        assert_eq!(
            label,
            Label {
                name: "name".to_string(),
                value: "1\\\"2".to_string()
            }
        );
    }

    #[test]
    fn compare_labelset() {
        let labels1 = vec![
            Label {
                name: "label".into(),
                value: "value".into(),
            },
            Label {
                name: "label2".into(),
                value: "value2".into(),
            },
        ];
        let labels2 = vec![
            Label {
                name: "label".into(),
                value: "value".into(),
            },
            Label {
                name: "label2".into(),
                value: "value2".into(),
            },
        ];

        assert_eq!(LabelSet { labels: labels1 }, LabelSet { labels: labels2 })
    }

    #[test]
    fn compare_labelset_ne() {
        let labels1 = vec![
            Label {
                name: "label".into(),
                value: "value".into(),
            },
            Label {
                name: "label2".into(),
                value: "value2".into(),
            },
        ];
        let labels2 = vec![
            Label {
                name: "label".into(),
                value: "value".into(),
            },
            Label {
                name: "label2".into(),
                value: "value3".into(),
            },
        ];
        assert_ne!(LabelSet { labels: labels1 }, LabelSet { labels: labels2 })
    }

    #[test]
    fn compare_labelset_with_quantile() {
        let labels1 = vec![
            Label {
                name: "label".into(),
                value: "value".into(),
            },
            Label {
                name: "quantile".into(),
                value: "0.5".into(),
            },
        ];
        let labels2 = vec![
            Label {
                name: "label".into(),
                value: "value".into(),
            },
            Label {
                name: "quantile".into(),
                value: "0.8".into(),
            },
        ];
        assert_eq!(LabelSet { labels: labels1 }, LabelSet { labels: labels2 })
    }

    #[test]
    fn compare_labelset_with_le() {
        let labels1 = vec![
            Label {
                name: "label".into(),
                value: "value".into(),
            },
            Label {
                name: "le".into(),
                value: "0.5".into(),
            },
        ];
        let labels2 = vec![
            Label {
                name: "label".into(),
                value: "value".into(),
            },
            Label {
                name: "le".into(),
                value: "0.8".into(),
            },
        ];
        assert_eq!(LabelSet { labels: labels1 }, LabelSet { labels: labels2 })
    }

    #[test]
    fn parse_label_with_special_chars() {
        let src = "name=\"Hello! This is a test.\"";
        let label = final_parser(parse_label::<ErrorTree<Span>>)(Span::new(src))
            .or_else(|e| {
                render_error(src, e);
                Err(())
            })
            .unwrap();
        assert_eq!(
            label,
            Label {
                name: "name".to_string(),
                value: "Hello! This is a test.".to_string()
            }
        );
    }

    #[test]
    fn parse_labelset_test() {
        let (input, labelset) =
            parse_labelset::<ErrorTree<Span>>(Span::new("{name1=\"value1\",name2=\"value2\"}test"))
                .unwrap();
        assert_eq!(*input.fragment(), "test");
        assert_eq!(
            labelset,
            vec![
                Label {
                    name: "name1".to_string(),
                    value: "value1".to_string()
                },
                Label {
                    name: "name2".to_string(),
                    value: "value2".to_string()
                },
            ]
            .into()
        );
    }

    #[test]
    fn parse_empty_labelset_test() {
        let (_, labelset) = parse_labelset::<ErrorTree<Span>>(" ".into()).unwrap();
        assert_eq!(labelset, LabelSet::new())
    }

    #[test]
    fn parse_summary_metrictype_test() {
        let (_, metric_type) = parse_metrictype::<ErrorTree<Span>>("summary".into()).unwrap();
        assert_eq!(metric_type, MetricType::Summary);
    }

    #[test]
    fn parse_type_metadata_test() {
        let src = "# TYPE foo_seconds counter\n";
        let (name, metric_type) =
            final_parser(|it| parse_metric_family_metadata::<ErrorTree<Span>>(it, None))(
                src.into(),
            )
            .or_else(|e| {
                render_error(src, e);
                Err(())
            })
            .unwrap();
        assert_eq!(*name.fragment(), "foo_seconds");
        assert_eq!(
            metric_type,
            MetricFamilyMetadata::MetricType(MetricType::Counter)
        );
    }

    #[test]
    fn parse_type_metadata_test_with_name() {
        let src = "# TYPE foo_seconds counter\n";
        let (name, metric_type) = final_parser(|it| {
            parse_metric_family_metadata::<ErrorTree<Span>>(it, Some("foo_seconds"))
        })(src.into())
        .or_else(|e| {
            render_error(src, e);
            Err(())
        })
        .unwrap();
        assert_eq!(*name.fragment(), "foo_seconds");
        assert_eq!(
            metric_type,
            MetricFamilyMetadata::MetricType(MetricType::Counter)
        );
    }

    #[test]
    fn parse_type_metadata_test_with_wrong_name() {
        let src = "# TYPE foo_seconds counter\n";
        let result = final_parser(|it| {
            parse_metric_family_metadata::<ErrorTree<Span>>(it, Some("bar_seconds"))
        })(src.into())
        .or_else(|e| {
            render_error(src, e);
            Err(())
        });
        assert!(result.is_err());
    }

    #[test]
    fn parse_type_metadata_test_without_label() {
        let src = "TYPE foo_seconds counter\n";
        let (name, metric_type) =
            final_parser(|it| parse_type_metadata::<ErrorTree<Span>>(it, None))(src.into())
                .or_else(|e| {
                    render_error(src, e);
                    Err(())
                })
                .unwrap();
        assert_eq!(*name.fragment(), "foo_seconds");
        assert_eq!(
            metric_type,
            MetricFamilyMetadata::MetricType(MetricType::Counter)
        );
    }

    #[test]
    fn parse_unit_metadata_test() {
        let (_, (name, metric_type)) = parse_metric_family_metadata::<ErrorTree<Span>>(
            "# UNIT foo_seconds seconds\n".into(),
            None,
        )
        .unwrap();
        assert_eq!(*name.fragment(), "foo_seconds");
        assert_eq!(
            metric_type,
            MetricFamilyMetadata::Unit("seconds".to_string())
        );
    }

    #[test]
    fn parse_unit_metadata_test_with_name() {
        let (_, (name, metric_type)) = parse_metric_family_metadata::<ErrorTree<Span>>(
            "# UNIT foo_seconds seconds\n".into(),
            Some("foo_seconds"),
        )
        .unwrap();
        assert_eq!(*name.fragment(), "foo_seconds");
        assert_eq!(
            metric_type,
            MetricFamilyMetadata::Unit("seconds".to_string())
        );
    }

    #[test]
    fn parse_help_metadata_test() {
        let src = "# HELP foo_seconds help text\n";
        let (name, metric_type) =
            final_parser(|it| parse_metric_family_metadata::<ErrorTree<Span>>(it, None))(
                src.into(),
            )
            .or_else(|e| {
                render_error(src, e);
                Err(())
            })
            .unwrap();
        assert_eq!(*name.fragment(), "foo_seconds");
        assert_eq!(
            metric_type,
            MetricFamilyMetadata::Help("help text".to_string())
        );
    }

    #[test]
    fn parse_help_metadata_test_with_name() {
        let src = "# HELP foo_seconds help text\n";
        let (name, metric_type) = final_parser(|it| {
            parse_metric_family_metadata::<ErrorTree<Span>>(it, Some("foo_seconds"))
        })(src.into())
        .or_else(|e| {
            render_error(src, e);
            Err(())
        })
        .unwrap();
        assert_eq!(*name.fragment(), "foo_seconds");
        assert_eq!(
            metric_type,
            MetricFamilyMetadata::Help("help text".to_string())
        );
    }

    #[test]
    fn until_space_test() {
        let (_, (one, two)) = tuple((
            alphanumeric1::<Span, ErrorTree<Span>>,
            preceded(space0, alphanumeric1),
        ))("one two\n".into())
        .unwrap();
        assert_eq!(*one.fragment(), "one");
        assert_eq!(*two.fragment(), "two");
    }

    #[test]
    fn parse_metric_family_with_only_metadata() {
        let src =
            "# TYPE foo_seconds gauge\n# UNIT foo_seconds seconds\n# HELP foo_seconds help text\n";
        let metric_family =
            final_parser({ |it| parse_metric_family::<ErrorTree<Span>>(it, &None) })(src.into())
                .or_else(|e| {
                    render_error(src, e);
                    Err(())
                })
                .unwrap();

        let expected = MetricFamily {
            name: "foo_seconds".to_string(),
            metric_type: MetricType::Gauge,
            unit: Some("seconds".to_string()),
            help: Some("help text".to_string()),
            metrics: vec![],
        };

        assert_eq!(metric_family, expected);
    }

    #[test]
    fn parse_metric_family_with_metric() {
        let src = "# TYPE foo_seconds gauge\nfoo_seconds{label=\"value\"} 150\n# EOF";
        let metric_family = final_parser(terminated(
            |it| parse_metric_family::<ErrorTree<Span>>(it, &None),
            preceded(multispace0, tag("# EOF")),
        ))(src.into())
        .or_else(|e| {
            render_error(src, e);
            Err(())
        })
        .unwrap();

        let expected = MetricFamily {
            name: "foo_seconds".to_string(),
            metric_type: MetricType::Gauge,
            unit: None,
            help: None,
            metrics: vec![Metric {
                labels: serde_json::from_str("{\"label\":\"value\"}").unwrap(),
                metric_points: vec![MetricPoint {
                    value: MetricValue::GaugeValue(IntOrFloat::Int(150)),
                    timestamp: None,
                }],
            }],
        };

        assert_eq!(metric_family, expected);
    }

    #[test]
    fn parse_metric_family_without_metadata() {
        let src = "foo_seconds{label=\"value\"} 150\n# EOF";
        let metric_family = final_parser(terminated(
            |it| parse_metric_family::<ErrorTree<Span>>(it, &None),
            tag("# EOF"),
        ))(src.into())
        .or_else(|e| {
            render_error(src, e);
            Err(())
        })
        .unwrap();

        let expected = MetricFamily {
            name: "foo_seconds".to_string(),
            metric_type: MetricType::Unknown,
            unit: None,
            help: None,
            metrics: vec![Metric {
                labels: serde_json::from_str("{\"label\":\"value\"}").unwrap(),
                metric_points: vec![MetricValue::UnknownValue(IntOrFloat::Int(150)).into()],
            }],
        };

        assert_eq!(metric_family, expected);
    }

    #[test]
    fn parse_metric_family_with_multiple_metrics() {
        let src = "# TYPE foo_seconds gauge\nfoo_seconds{label=\"value1\"} 150\nfoo_seconds{label=\"value2\"} 100\n# EOF";
        let metric_family = final_parser(terminated(
            |it| parse_metric_family::<ErrorTree<Span>>(it, &None),
            tag("# EOF"),
        ))(src.into())
        .or_else(|e| {
            render_error(src, e);
            Err(())
        })
        .unwrap();

        let expected = MetricFamily {
            name: "foo_seconds".to_string(),
            metric_type: MetricType::Gauge,
            unit: None,
            help: None,
            metrics: vec![
                Metric {
                    labels: serde_json::from_str("{\"label\":\"value1\"}").unwrap(),
                    metric_points: vec![MetricValue::GaugeValue(IntOrFloat::Int(150)).into()],
                },
                Metric {
                    labels: serde_json::from_str("{\"label\":\"value2\"}").unwrap(),
                    metric_points: vec![MetricValue::GaugeValue(IntOrFloat::Int(100)).into()],
                },
            ],
        };

        assert_eq!(metric_family, expected);
    }

    #[test]
    fn parse_gauge_metric() {
        let src = "foo_seconds{label=\"value\"} 99\n# EOF";
        let (_, (_, metric)) = parse_metric::<ErrorTree<Span>>(
            src.into(),
            Some("foo_seconds"),
            &None,
            MetricType::Gauge,
        )
        .unwrap();
        assert_eq!(
            metric,
            Metric {
                labels: serde_json::from_str("{\"label\":\"value\"}").unwrap(),
                metric_points: vec![MetricValue::GaugeValue(IntOrFloat::Int(99)).into()]
            }
        );
    }

    #[test]
    fn parse_metric_with_multiple_metric_points() {
        let src =
            "foo_seconds{label=\"value\"} 99 123\nfoo_seconds{label=\"value\"} 100 456\n# EOF";
        let (_, (_, metric)) = parse_metric::<ErrorTree<Span>>(
            src.into(),
            Some("foo_seconds"),
            &None,
            MetricType::Gauge,
        )
        .unwrap();
        assert_eq!(
            metric,
            Metric {
                labels: serde_json::from_str("{\"label\":\"value\"}").unwrap(),
                metric_points: vec![
                    MetricPoint::new(
                        MetricValue::GaugeValue(IntOrFloat::Int(99)),
                        Some(123.into())
                    ),
                    MetricPoint::new(
                        MetricValue::GaugeValue(IntOrFloat::Int(100)),
                        Some(456.into())
                    )
                ]
            }
        );
    }

    #[test]
    fn parse_counter_with_multiple_metric_points() {
        let src =
            "foo_seconds_total{label=\"value\"} 99 123\nfoo_seconds_total{label=\"value\"} 100 456\n# EOF";
        let (_, (_, metric)) = parse_metric::<ErrorTree<Span>>(
            src.into(),
            Some("foo_seconds"),
            &None,
            MetricType::Counter,
        )
        .unwrap();
        assert_eq!(
            metric,
            Metric {
                labels: serde_json::from_str("{\"label\":\"value\"}").unwrap(),
                metric_points: vec![
                    MetricPoint::new(
                        CounterValue::new(IntOrFloat::Int(99), None).into(),
                        Some(123.into())
                    ),
                    MetricPoint::new(
                        CounterValue::new(IntOrFloat::Int(100), None).into(),
                        Some(456.into())
                    )
                ]
            }
        );
    }

    #[test]
    fn parse_unknown_metric_with_name_test() {
        let src = "foo_seconds{label=\"value\"} 99\n# EOF";
        let (_, (_, metric)) = parse_metric::<ErrorTree<Span>>(
            src.into(),
            Some("foo_seconds"),
            &None,
            MetricType::Unknown,
        )
        .unwrap();
        assert_eq!(
            metric,
            Metric {
                labels: serde_json::from_str("{\"label\":\"value\"}").unwrap(),
                metric_points: vec![MetricValue::UnknownValue(IntOrFloat::Int(99)).into()]
            }
        );
    }

    #[test]
    fn parse_unknown_metric_without_name_test() {
        let src = "foo_seconds{label=\"value\"} 99\n# EOF";
        let (_, (name, metric)) =
            parse_metric::<ErrorTree<Span>>(src.into(), None, &None, MetricType::Unknown).unwrap();
        assert_eq!(name, Some("foo_seconds".into()));
        assert_eq!(
            metric,
            Metric {
                labels: serde_json::from_str("{\"label\":\"value\"}").unwrap(),
                metric_points: vec![MetricValue::UnknownValue(IntOrFloat::Int(99)).into()]
            }
        );
    }

    #[test]
    fn parse_info_metric_test() {
        let src = "foo_info{version=\"1.0\"} 1\n# EOF";
        let (_, (_, metric)) =
            parse_metric::<ErrorTree<Span>>(src.into(), "foo".into(), &None, MetricType::Info)
                .unwrap();
        assert_eq!(
            metric,
            Metric {
                labels: serde_json::from_str("{\"version\":\"1.0\"}").unwrap(),
                metric_points: vec![MetricValue::InfoValue(IntOrFloat::Int(1)).into()]
            }
        );
    }

    #[test]
    fn parse_metric_set_test() {
        let src = "# TYPE foo_seconds gauge\nfoo_seconds{label=\"value\"} 150\n# TYPE bar_seconds gauge\nbar_seconds{label=\"value\"} 50\n# EOF";
        let metric_set =
            final_parser({ |i| parse_metric_set::<ErrorTree<Span>>(i, &None) })(src.into())
                .or_else(|e| {
                    render_error(src, e);
                    Err(())
                })
                .unwrap();

        let expected_1 = MetricFamily {
            name: "foo_seconds".to_string(),
            metric_type: MetricType::Gauge,
            unit: None,
            help: None,
            metrics: vec![Metric {
                labels: serde_json::from_str("{\"label\":\"value\"}").unwrap(),
                metric_points: vec![MetricValue::GaugeValue(IntOrFloat::Int(150)).into()],
            }],
        };

        let expected_2 = MetricFamily {
            name: "bar_seconds".to_string(),
            metric_type: MetricType::Gauge,
            unit: None,
            help: None,
            metrics: vec![Metric {
                labels: serde_json::from_str("{\"label\":\"value\"}").unwrap(),
                metric_points: vec![MetricValue::GaugeValue(IntOrFloat::Int(50)).into()],
            }],
        };

        let expected_metricset = MetricSet {
            metric_families: vec![expected_1, expected_2],
        };

        assert_eq!(metric_set, expected_metricset);
    }

    #[test]
    fn parse_histogram_metric_test() {
        let src = "foo_bucket{le=\"0.0\"} 0\n\
foo_bucket{le=\"1e-05\"} 0\n\
foo_bucket{le=\"0.0001\"} 5\n\
foo_bucket{le=\"1.0\"} 10\n\
foo_bucket{le=\"10.0\"} 11\n\
foo_bucket{le=\"+Inf\"} 17\n\
foo_count 17\n\
foo_sum 324789.3\n\
foo_created 1520430000.123\n";
        let (_, (name, result)) =
            parse_metric::<ErrorTree<Span>>(src.into(), Some("foo"), &None, MetricType::Histogram)
                .unwrap();
        assert_eq!(
            result,
            Metric {
                labels: HashMap::new(),
                metric_points: vec![MetricPoint {
                    timestamp: None,
                    value: MetricValue::HistogramValue(HistogramValue::new(
                        Some(324789.3.into()),
                        Some(17),
                        Some(1520430000.123.into()),
                        vec![
                            Bucket {
                                upper_bound: Some(0.0),
                                count: 0
                            },
                            Bucket {
                                upper_bound: Some(1e-05),
                                count: 0
                            },
                            Bucket {
                                upper_bound: Some(0.0001),
                                count: 5
                            },
                            Bucket {
                                upper_bound: Some(1.0),
                                count: 10
                            },
                            Bucket {
                                upper_bound: Some(10.0),
                                count: 11
                            },
                            Bucket {
                                upper_bound: Some(f64::INFINITY),
                                count: 17
                            },
                        ]
                    ))
                }]
            }
        );
    }

    #[test]
    fn parse_gaugehistogram_metric_test() {
        let src = "foo_bucket{le=\"0.0\"} 0\n\
foo_bucket{le=\"1e-05\"} 0\n\
foo_bucket{le=\"0.0001\"} 5\n\
foo_bucket{le=\"1.0\"} 10\n\
foo_bucket{le=\"10.0\"} 11\n\
foo_bucket{le=\"+Inf\"} 17\n\
foo_gcount 17\n\
foo_gsum 324789.3\n\
foo_created 1520430000.123\n";
        let (_, (name, result)) = parse_metric::<ErrorTree<Span>>(
            src.into(),
            Some("foo"),
            &None,
            MetricType::GaugeHistogram,
        )
        .unwrap();
        assert_eq!(
            result,
            Metric {
                labels: HashMap::new(),
                metric_points: vec![MetricPoint {
                    timestamp: None,
                    value: MetricValue::HistogramValue(HistogramValue::new(
                        Some(324789.3.into()),
                        Some(17),
                        Some(1520430000.123.into()),
                        vec![
                            Bucket {
                                upper_bound: Some(0.0),
                                count: 0
                            },
                            Bucket {
                                upper_bound: Some(1e-05),
                                count: 0
                            },
                            Bucket {
                                upper_bound: Some(0.0001),
                                count: 5
                            },
                            Bucket {
                                upper_bound: Some(1.0),
                                count: 10
                            },
                            Bucket {
                                upper_bound: Some(10.0),
                                count: 11
                            },
                            Bucket {
                                upper_bound: Some(f64::INFINITY),
                                count: 17
                            },
                        ]
                    ))
                }]
            }
        );
    }

    #[test]
    fn parse_stateset_test() {
        let src = "foo{foo=\"a\"} 0\nfoo{foo=\"bb\"} 1\nfoo{foo=\"ccc\"} 0\n";
        let (_, (_, metric)) =
            parse_metric::<ErrorTree<Span>>(src.into(), Some("foo"), &None, MetricType::StateSet)
                .unwrap();
        let mut states = HashMap::new();
        states.insert("a".to_string(), false);
        states.insert("bb".to_string(), true);
        states.insert("ccc".to_string(), false);
        assert_eq!(
            metric,
            Metric {
                labels: HashMap::new(),
                metric_points: vec![MetricPoint {
                    timestamp: None,
                    value: MetricValue::StateSetValue(StateSetValue::new(states))
                }]
            }
        )
    }

    #[test]
    fn parse_metric_with_source_label() {
        let src = "foo_seconds{label=\"value\"} 99\n# EOF";
        let (_, (_, metric)) = parse_metric::<ErrorTree<Span>>(
            src.into(),
            Some("foo_seconds"),
            &Some(Label {
                name: "metrics_source".into(),
                value: "test".into(),
            }),
            MetricType::Gauge,
        )
        .unwrap();
        assert_eq!(
            metric,
            Metric {
                labels: serde_json::from_str("{\"label\":\"value\",\"metrics_source\":\"test\"}").unwrap(),
                metric_points: vec![MetricValue::GaugeValue(IntOrFloat::Int(99)).into()]
            }
        );
    }
}
