use std::error::Error;

use chrono::{DateTime, Utc};
use miette::GraphicalReportHandler;
use nom::{
    branch::alt,
    bytes::streaming::{tag, take_until1},
    character::{
        streaming::space0,
        streaming::{
            alpha1, alphanumeric1, line_ending, multispace0, multispace1, not_line_ending, space1,
        },
    },
    combinator::{map, map_res, value},
    error::{ErrorKind, ParseError},
    multi::separated_list0,
    number::streaming::double,
    sequence::{delimited, preceded, terminated, tuple},
    IResult,
};
use nom_locate::LocatedSpan;
use nom_supreme::{
    error::{BaseErrorKind, ErrorTree, GenericErrorTree},
    final_parser::Location,
};
use serde::Serialize;

const METRIC_FAMILY_NAME_CHARS: &str = "abcdefghijklmnopqrstuvwxyz_";

#[derive(Debug, Clone, Serialize)]
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

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
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

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Metric {
    labels: Vec<Label>,
    metric_points: Vec<MetricPoint>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct Label {
    name: String,
    value: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub enum MetricPoint {
    // UnknownValue(IntOrFloat),
    GaugeValue(IntOrFloat),
    // CounterValue(CounterValue),
    // HistogramValue,
    // StateSetValue,
    // InfoValue,
    // SummaryValue,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
enum IntOrFloat {
    Int(i64),
    Float(f64),
}

type Span<'a> = LocatedSpan<&'a str>;

#[derive(thiserror::Error, Debug, miette::Diagnostic)]
#[error("bad input")]
struct BadInput {
    #[source_code]
    src: String,

    #[label("{kind}")]
    bad_bit: miette::SourceSpan,

    kind: BaseErrorKind<&'static str, Box<dyn std::error::Error + Send + Sync>>,
}

fn render_error(src_input: &str, e: ErrorTree<Span>) {
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
    println!("{s}");
}

fn parse_label<'a, E: ParseError<Span<'a>>>(i: Span<'a>) -> IResult<Span<'a>, Label, E> {
    let (input, (name, _, value)) = tuple((
        alphanumeric1,
        tag("="),
        delimited(tag("\""), alphanumeric1, tag("\"")),
    ))(i)?;
    Ok((
        input,
        Label {
            name: name.to_string(),
            value: value.to_string(),
        },
    ))
}

fn parse_labelset<'a, E: ParseError<Span<'a>>>(i: Span<'a>) -> IResult<Span<'a>, Vec<Label>, E> {
    delimited(
        tag("{"),
        separated_list0(tag(","), parse_label::<E>),
        tag("}"),
    )(i)
}

fn parse_metric_family<'a, E: ParseError<Span<'a>>>(
    i: Span<'a>,
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

    match metric_family_name {
        Some(name) => Ok((
            i,
            MetricFamily {
                name: name.to_string(),
                metric_type: metric_family_type,
                unit: metric_family_unit,
                help: metric_family_help,
                metrics: vec![],
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
        value(MetricType::Gauge, tag("gauge")),
        value(MetricType::Counter, tag("counter")),
        value(MetricType::StateSet, tag("state_set")),
        value(MetricType::Info, tag("info")),
        value(MetricType::Histogram, tag("histogram")),
        value(MetricType::GagueHistogram, tag("gague_histogram")),
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
        None => Box::new(take_until1(" ")),
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
        None => Box::new(take_until1(" ")),
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
        None => Box::new(take_until1(" ")),
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

// fn parse_gauge_metric<'a, E: ParseError<Span<'a>>>(
//     i: Span<'a>,
//     name: Option<&str>,
// ) -> IResult<Span<'a>, (Span<'a>, MetricFamilyMetadata), E> {
//     // let name_match: Box<
//     //     dyn Fn(
//     //         LocatedSpan<&'a str>,
//     //     ) -> Result<(LocatedSpan<&'a str>, LocatedSpan<&'a str>), nom::Err<E>>,
//     // > = match name {
//     //     Some(name) => Box::new(tag(name)),
//     //     None => Box::new(take_until1("{")),
//     // };

//     alt((
//         tuple((take_until1("{"), parse_labelset)),
//         tuple(take_until1(" ")),
//     ))(i)
// }

fn parse_int_or_float<'a, E: ParseError<Span<'a>>>(
    i: Span<'a>,
) -> IResult<Span<'a>, IntOrFloat, E> {
    alt((
        map(
            terminated(nom::character::streaming::i64, multispace1),
            |value| IntOrFloat::Int(value),
        ),
        map(terminated(double, multispace1), |value| {
            IntOrFloat::Float(value)
        }),
    ))(i)
}

#[cfg(test)]
mod test {

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
        );
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
            "# TYPE foo_seconds counter\n# UNIT foo_seconds seconds\n# HELP foo_seconds help text\n";
        let metric_family = final_parser(parse_metric_family::<ErrorTree<Span>>)(src.into())
            .or_else(|e| {
                render_error(src, e);
                Err(())
            })
            .unwrap();

        let expected = MetricFamily {
            name: "foo_seconds".to_string(),
            metric_type: MetricType::Counter,
            unit: Some("seconds".to_string()),
            help: Some("help text".to_string()),
            metrics: vec![],
        };

        assert_eq!(metric_family, expected);
    }

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
        assert_eq!(value, IntOrFloat::Float(1.5))
    }

    #[test]
    fn parse_scientific_notation_to_int_or_float() {
        let (_, value) = parse_int_or_float::<ErrorTree<Span>>("1.89e-7\nA".into()).unwrap();
        assert_eq!(value, IntOrFloat::Float(0.000000189))
    }
}
