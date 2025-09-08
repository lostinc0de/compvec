use std::cmp::PartialEq;
use std::error::Error;
use std::fmt::{Display, Formatter, Result};

#[derive(Clone, Debug, PartialEq)]
pub enum CompVecError {
    Error(String),
    ConversionError(String),
    OutOfBoundsError(String),
}

impl Error for CompVecError {}

impl Display for CompVecError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            CompVecError::Error(msg) => write!(f, "Error: {}", msg),
            CompVecError::ConversionError(msg) => write!(f, "Conversion error: {}", msg),
            CompVecError::OutOfBoundsError(msg) => write!(f, "Out of Bounds error: {}", msg),
        }
    }
}
