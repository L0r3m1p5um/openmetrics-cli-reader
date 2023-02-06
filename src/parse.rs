use nom::{
    bytes::complete::tag,
    character::complete::alpha1,
    sequence::{delimited, tuple},
    IResult,
};

use crate::metrics::Label;

fn parse_label(i: &str) -> IResult<&str, Label> {
    let (input, (name, _, value)) =
        tuple((alpha1, tag("="), delimited(tag("\""), alpha1, tag("\""))))(i)?;
    Ok((input, Label {
        name: name.to_string(),
        value: name.to_string(),
    }))
}

#[cfg(test)]
mod test {}
