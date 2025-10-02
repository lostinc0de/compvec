use crate::error::CompVecError;
use std::cmp::Ord;
use std::fmt::{Debug, Display, Binary};
use std::iter::Sum;
use std::ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign, Shl, Shr, BitAnd, BitOr, BitAndAssign, BitOrAssign, Not};

pub trait Integer
where
    Self: Copy
        + Shl<usize, Output = Self>
        + Shr<usize, Output = Self>
        + BitAnd<Output = Self>
        + BitOr<Output = Self>
        + BitAndAssign
        + BitOrAssign
        + Not<Output = Self>
        + Add<Output = Self>
        + AddAssign
        + Sub<Output = Self>
        + SubAssign
        + Mul<Output = Self>
        + MulAssign
        + Div<Output = Self>
        + DivAssign
        + Ord
        + Sum
        + Display
        + Debug
        + Default
        + Binary,
{
    const ZERO: Self;
    const ONE: Self;
    const TWO: Self;
    const MIN: Self;
    const MAX: Self;
    const MAX_U64: u64;
    const N_BITS: usize;
    const N_BITS_U64: u64;
    const SIGNED: bool;

    /// Checks, if the integer is negative.
    fn is_neg(&self) -> bool {
        if Self::SIGNED {
            if *self < Self::ZERO {
                return true;
            }
        }
        false
    }

    /// Returns the absolute value of the integer.
    fn abs(&self) -> Self {
        if self.is_neg() {
            // Compute the absolute value using two's complement
            return !(*self) + Self::ONE;
        }
        *self
    }

    /// Returns a negated integer, if integer is signed.
    fn negate(&self) -> Self {
        if Self::SIGNED {
            return !(*self - Self::ONE);
        }
        return *self;
    }
}

macro_rules! impl_integer {
    ($t: ty) => {
        impl Integer for $t {
            const ZERO: Self = 0 as $t;
            const ONE: Self = 1 as $t;
            const TWO: Self = 2 as $t;
            const MIN: Self = <$t>::MIN;
            const MAX: Self = <$t>::MAX;
            const MAX_U64: u64 = <$t>::MAX as u64;
            const N_BITS: usize = <$t>::BITS as usize;
            const N_BITS_U64: u64 = <$t>::BITS as u64;
            const SIGNED: bool = <$t>::MIN != 0;
        }
    };
}

impl_integer!(u8);
impl_integer!(u16);
impl_integer!(u32);
impl_integer!(u64);
impl_integer!(u128);
impl_integer!(i8);
impl_integer!(i16);
impl_integer!(i32);
impl_integer!(i64);
impl_integer!(i128);
impl_integer!(usize);
impl_integer!(isize);

/// Trait for converting an integer safely into a truncated smaller integer
pub trait IntoTruncated<U: Integer>
where
    Self: Sized,
{
    /// Cast self into a smaller truncated integer type.
    fn into_trunc(&self) -> U;

    /// Safe version of into_trunc().
    /// Returns None if the bigger integer would be sliced.
    fn into_trunc_safe(&self) -> Result<U, CompVecError>;
}

/// Trait for converting a smaller integer type into a larger one
/// We cannot use the Into trait here since usize only implements
/// the From trait with u8 and u16.
pub trait FromTruncated<U: Integer>
where
    Self: Sized,
{
    /// Create an integer from a truncated one
    fn from_trunc(other: U) -> Self;

    /// Safe version of from_trunc().
    fn from_trunc_safe(other: U) -> Result<Self, CompVecError>;
}

/// Macro implementing the traits IntoTruncated and FromTruncated for an integer type.
macro_rules! trunc_conv {
    ($t: ty, $u: ty) => {
        impl IntoTruncated<$u> for $t {
            fn into_trunc(&self) -> $u {
                *self as $u
            }

            fn into_trunc_safe(&self) -> Result<$u, CompVecError> {
                if <$u>::MAX_U64 > <$t>::MAX_U64 {
                    let msg = String::from("Truncated type should be smaller");
                    return Err(CompVecError::ConversionError(msg));
                }
                if *self > <$u>::MAX as $t {
                    let msg = String::from("Destination type is too small");
                    return Err(CompVecError::ConversionError(msg));
                }
                Ok(*self as $u)
            }
        }

        impl FromTruncated<$u> for $t {
            fn from_trunc(other: $u) -> Self {
                other as $t
            }

            fn from_trunc_safe(other: $u) -> Result<Self, CompVecError> {
                if <$u>::MAX_U64 > <$t>::MAX_U64 {
                    let msg = String::from("Truncated type should be smaller");
                    return Err(CompVecError::ConversionError(msg));
                }
                Ok(other as $t)
            }
        }
    };
}

trunc_conv!(u16, u8);
trunc_conv!(u32, u8);
trunc_conv!(u32, u16);
trunc_conv!(u64, u8);
trunc_conv!(u64, u16);
trunc_conv!(u64, u32);
trunc_conv!(u128, u8);
trunc_conv!(u128, u16);
trunc_conv!(u128, u32);
trunc_conv!(u128, u64);
trunc_conv!(i16, u8);
trunc_conv!(i32, u8);
trunc_conv!(i32, u16);
trunc_conv!(i64, u8);
trunc_conv!(i64, u16);
trunc_conv!(i64, u32);
trunc_conv!(i128, u8);
trunc_conv!(i128, u16);
trunc_conv!(i128, u32);
trunc_conv!(i128, u64);
trunc_conv!(usize, u8);
trunc_conv!(usize, u16);
trunc_conv!(usize, u32);
trunc_conv!(isize, u8);
trunc_conv!(isize, u16);
trunc_conv!(isize, u32);
