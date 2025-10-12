use crate::types::*;
use crate::veclike::VecLike;

/// ByteVec stores data of type T in a smaller integer type U.
#[derive(Clone, Debug)]
pub struct ByteVec<T: Integer, U: Integer>
where
    T: FromTruncated<U> + IntoTruncated<U>,
{
    values: Vec<U>,
    max: T,
    stride: usize,
    n: usize,
}

impl<T: Integer, U: Integer> ByteVec<T, U>
where
    T: FromTruncated<U> + IntoTruncated<U>,
{
    /// Returns the maximum value and the new stride, such that val fits.
    fn compute_max_value_and_stride(val: T) -> (T, usize) {
        let mul = T::from_trunc(U::MAX);
        let mut max: T = if T::SIGNED {
            // For signed integers leave out the signed bit
            T::from_trunc(U::MAX / U::TWO)
        } else {
            mul
        };
        let mut stride: usize = 1;
        // Multiply max value until it is large enough
        while max < val.abs() {
            max *= mul;
            stride += 1;
        }
        (max, stride)
    }

    /// Resizes the underlying Vec of smaller integers, if val is too large.
    fn check_and_resize(&mut self, val: T) {
        if val.abs() > self.max {
            let mut cop = Self::with_max_value(self.len(), val);
            for v in self.iterate() {
                cop.push(v);
            }
            *self = cop;
        }
    }

    /// Returns a byte vec container with a maximum value.
    /// Internal variables are set, such that the underlying
    /// data vec does not have to be reallocated.
    pub fn with_max_value(n: usize, val_max: T) -> Self {
        let (max, stride) = Self::compute_max_value_and_stride(val_max);
        Self {
            values: Vec::<U>::with_capacity(n * stride),
            max,
            stride,
            n: 0,
        }
    }

    /// Return the stride for debugging.
    pub fn stride(&self) -> usize {
        self.stride
    }
}

impl<T: Integer, U: Integer> VecLike for ByteVec<T, U>
where
    T: FromTruncated<U> + IntoTruncated<U>,
{
    type Type = T;

    fn with_capacity(n: usize) -> Self {
        let max = if T::SIGNED {
            T::from_trunc(U::MAX / U::TWO)
        } else {
            T::from_trunc(U::MAX)
        };
        Self {
            values: Vec::<U>::with_capacity(n),
            max,
            stride: 1,
            n: 0,
        }
    }

    fn get(&self, i: usize) -> T {
        let start = i * self.stride;
        // Loop unrolling for performance
        let mut ret = match self.stride {
            1 => T::from_trunc(self.values[start]),
            2 => {
                T::from_trunc(self.values[start])
                    | (T::from_trunc(self.values[start + 1]) << U::N_BITS)
            }
            3 => {
                T::from_trunc(self.values[start])
                    | (T::from_trunc(self.values[start + 1]) << U::N_BITS)
                    | (T::from_trunc(self.values[start + 2]) << 2 * U::N_BITS)
            }
            4 => {
                T::from_trunc(self.values[start])
                    | (T::from_trunc(self.values[start + 1]) << U::N_BITS)
                    | (T::from_trunc(self.values[start + 2]) << 2 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 3]) << 3 * U::N_BITS)
            }
            5 => {
                T::from_trunc(self.values[start])
                    | (T::from_trunc(self.values[start + 1]) << U::N_BITS)
                    | (T::from_trunc(self.values[start + 2]) << 2 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 3]) << 3 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 4]) << 4 * U::N_BITS)
            }
            6 => {
                T::from_trunc(self.values[start])
                    | (T::from_trunc(self.values[start + 1]) << U::N_BITS)
                    | (T::from_trunc(self.values[start + 2]) << 2 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 3]) << 3 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 4]) << 4 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 5]) << 5 * U::N_BITS)
            }
            7 => {
                T::from_trunc(self.values[start])
                    | (T::from_trunc(self.values[start + 1]) << U::N_BITS)
                    | (T::from_trunc(self.values[start + 2]) << 2 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 3]) << 3 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 4]) << 4 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 5]) << 5 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 6]) << 6 * U::N_BITS)
            }
            8 => {
                T::from_trunc(self.values[start])
                    | (T::from_trunc(self.values[start + 1]) << U::N_BITS)
                    | (T::from_trunc(self.values[start + 2]) << 2 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 3]) << 3 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 4]) << 4 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 5]) << 5 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 6]) << 6 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 7]) << 7 * U::N_BITS)
            }
            9 => {
                T::from_trunc(self.values[start])
                    | (T::from_trunc(self.values[start + 1]) << U::N_BITS)
                    | (T::from_trunc(self.values[start + 2]) << 2 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 3]) << 3 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 4]) << 4 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 5]) << 5 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 6]) << 6 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 7]) << 7 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 8]) << 8 * U::N_BITS)
            }
            10 => {
                T::from_trunc(self.values[start])
                    | (T::from_trunc(self.values[start + 1]) << U::N_BITS)
                    | (T::from_trunc(self.values[start + 2]) << 2 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 3]) << 3 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 4]) << 4 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 5]) << 5 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 6]) << 6 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 7]) << 7 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 8]) << 8 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 9]) << 9 * U::N_BITS)
            }
            11 => {
                T::from_trunc(self.values[start])
                    | (T::from_trunc(self.values[start + 1]) << U::N_BITS)
                    | (T::from_trunc(self.values[start + 2]) << 2 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 3]) << 3 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 4]) << 4 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 5]) << 5 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 6]) << 6 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 7]) << 7 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 8]) << 8 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 9]) << 9 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 10]) << 10 * U::N_BITS)
            }
            12 => {
                T::from_trunc(self.values[start])
                    | (T::from_trunc(self.values[start + 1]) << U::N_BITS)
                    | (T::from_trunc(self.values[start + 2]) << 2 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 3]) << 3 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 4]) << 4 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 5]) << 5 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 6]) << 6 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 7]) << 7 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 8]) << 8 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 9]) << 9 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 10]) << 10 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 11]) << 11 * U::N_BITS)
            }
            13 => {
                T::from_trunc(self.values[start])
                    | (T::from_trunc(self.values[start + 1]) << U::N_BITS)
                    | (T::from_trunc(self.values[start + 2]) << 2 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 3]) << 3 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 4]) << 4 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 5]) << 5 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 6]) << 6 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 7]) << 7 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 8]) << 8 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 9]) << 9 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 10]) << 10 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 11]) << 11 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 12]) << 12 * U::N_BITS)
            }
            14 => {
                T::from_trunc(self.values[start])
                    | (T::from_trunc(self.values[start + 1]) << U::N_BITS)
                    | (T::from_trunc(self.values[start + 2]) << 2 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 3]) << 3 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 4]) << 4 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 5]) << 5 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 6]) << 6 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 7]) << 7 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 8]) << 8 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 9]) << 9 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 10]) << 10 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 11]) << 11 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 12]) << 12 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 13]) << 13 * U::N_BITS)
            }
            15 => {
                T::from_trunc(self.values[start])
                    | (T::from_trunc(self.values[start + 1]) << U::N_BITS)
                    | (T::from_trunc(self.values[start + 2]) << 2 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 3]) << 3 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 4]) << 4 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 5]) << 5 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 6]) << 6 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 7]) << 7 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 8]) << 8 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 9]) << 9 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 10]) << 10 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 11]) << 11 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 12]) << 12 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 13]) << 13 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 14]) << 14 * U::N_BITS)
            }
            16 => {
                T::from_trunc(self.values[start])
                    | (T::from_trunc(self.values[start + 1]) << U::N_BITS)
                    | (T::from_trunc(self.values[start + 2]) << 2 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 3]) << 3 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 4]) << 4 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 5]) << 5 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 6]) << 6 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 7]) << 7 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 8]) << 8 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 9]) << 9 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 10]) << 10 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 11]) << 11 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 12]) << 12 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 13]) << 13 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 14]) << 14 * U::N_BITS)
                    | (T::from_trunc(self.values[start + 15]) << 15 * U::N_BITS)
            }
            _ => {
                let mut ret = T::from_trunc(self.values[start]);
                for i in 1..self.stride {
                    let val = T::from_trunc(self.values[start + i]);
                    ret |= val << (i * U::N_BITS);
                }
                ret
            }
        };
        if T::SIGNED {
            let sign = self.values[start + self.stride - 1] & (U::ONE << (U::N_BITS - 1));
            if sign > U::ZERO {
                // Remove the sign bit and negate the return value using two's complement
                ret &= !(T::ONE << (self.stride * U::N_BITS - 1));
                ret = ret.negate();
            }
        }
        ret
    }

    fn set(&mut self, i: usize, val: T) {
        self.check_and_resize(val);
        let start = i * self.stride;
        let val_abs = val.abs();
        self.values[start] = val_abs.into_trunc();
        // Loop unrolling for performance
        match self.stride {
            1 => {
            },
            2 => {
                self.values[start + 1] = (val_abs >> U::N_BITS).into_trunc();
            },
            3 => {
                self.values[start + 1] = (val_abs >> U::N_BITS).into_trunc();
                self.values[start + 2] = (val_abs >> 2 * U::N_BITS).into_trunc();
            },
            4 => {
                self.values[start + 1] = (val_abs >> U::N_BITS).into_trunc();
                self.values[start + 2] = (val_abs >> 2 * U::N_BITS).into_trunc();
                self.values[start + 3] = (val_abs >> 3 * U::N_BITS).into_trunc();
            },
            5 => {
                self.values[start + 1] = (val_abs >> U::N_BITS).into_trunc();
                self.values[start + 2] = (val_abs >> 2 * U::N_BITS).into_trunc();
                self.values[start + 3] = (val_abs >> 3 * U::N_BITS).into_trunc();
                self.values[start + 4] = (val_abs >> 4 * U::N_BITS).into_trunc();
            },
            6 => {
                self.values[start + 1] = (val_abs >> U::N_BITS).into_trunc();
                self.values[start + 2] = (val_abs >> 2 * U::N_BITS).into_trunc();
                self.values[start + 3] = (val_abs >> 3 * U::N_BITS).into_trunc();
                self.values[start + 4] = (val_abs >> 4 * U::N_BITS).into_trunc();
                self.values[start + 5] = (val_abs >> 5 * U::N_BITS).into_trunc();
            },
            7 => {
                self.values[start + 1] = (val_abs >> U::N_BITS).into_trunc();
                self.values[start + 2] = (val_abs >> 2 * U::N_BITS).into_trunc();
                self.values[start + 3] = (val_abs >> 3 * U::N_BITS).into_trunc();
                self.values[start + 4] = (val_abs >> 4 * U::N_BITS).into_trunc();
                self.values[start + 5] = (val_abs >> 5 * U::N_BITS).into_trunc();
                self.values[start + 6] = (val_abs >> 6 * U::N_BITS).into_trunc();
            },
            8 => {
                self.values[start + 1] = (val_abs >> U::N_BITS).into_trunc();
                self.values[start + 2] = (val_abs >> 2 * U::N_BITS).into_trunc();
                self.values[start + 3] = (val_abs >> 3 * U::N_BITS).into_trunc();
                self.values[start + 4] = (val_abs >> 4 * U::N_BITS).into_trunc();
                self.values[start + 5] = (val_abs >> 5 * U::N_BITS).into_trunc();
                self.values[start + 6] = (val_abs >> 6 * U::N_BITS).into_trunc();
                self.values[start + 7] = (val_abs >> 7 * U::N_BITS).into_trunc();
            },
            9 => {
                self.values[start + 1] = (val_abs >> U::N_BITS).into_trunc();
                self.values[start + 2] = (val_abs >> 2 * U::N_BITS).into_trunc();
                self.values[start + 3] = (val_abs >> 3 * U::N_BITS).into_trunc();
                self.values[start + 4] = (val_abs >> 4 * U::N_BITS).into_trunc();
                self.values[start + 5] = (val_abs >> 5 * U::N_BITS).into_trunc();
                self.values[start + 6] = (val_abs >> 6 * U::N_BITS).into_trunc();
                self.values[start + 7] = (val_abs >> 7 * U::N_BITS).into_trunc();
                self.values[start + 8] = (val_abs >> 8 * U::N_BITS).into_trunc();
            },
            10 => {
                self.values[start + 1] = (val_abs >> U::N_BITS).into_trunc();
                self.values[start + 2] = (val_abs >> 2 * U::N_BITS).into_trunc();
                self.values[start + 3] = (val_abs >> 3 * U::N_BITS).into_trunc();
                self.values[start + 4] = (val_abs >> 4 * U::N_BITS).into_trunc();
                self.values[start + 5] = (val_abs >> 5 * U::N_BITS).into_trunc();
                self.values[start + 6] = (val_abs >> 6 * U::N_BITS).into_trunc();
                self.values[start + 7] = (val_abs >> 7 * U::N_BITS).into_trunc();
                self.values[start + 8] = (val_abs >> 8 * U::N_BITS).into_trunc();
                self.values[start + 9] = (val_abs >> 9 * U::N_BITS).into_trunc();
            },
            11 => {
                self.values[start + 1] = (val_abs >> U::N_BITS).into_trunc();
                self.values[start + 2] = (val_abs >> 2 * U::N_BITS).into_trunc();
                self.values[start + 3] = (val_abs >> 3 * U::N_BITS).into_trunc();
                self.values[start + 4] = (val_abs >> 4 * U::N_BITS).into_trunc();
                self.values[start + 5] = (val_abs >> 5 * U::N_BITS).into_trunc();
                self.values[start + 6] = (val_abs >> 6 * U::N_BITS).into_trunc();
                self.values[start + 7] = (val_abs >> 7 * U::N_BITS).into_trunc();
                self.values[start + 8] = (val_abs >> 8 * U::N_BITS).into_trunc();
                self.values[start + 9] = (val_abs >> 9 * U::N_BITS).into_trunc();
                self.values[start + 10] = (val_abs >> 10 * U::N_BITS).into_trunc();
            },
            12 => {
                self.values[start + 1] = (val_abs >> U::N_BITS).into_trunc();
                self.values[start + 2] = (val_abs >> 2 * U::N_BITS).into_trunc();
                self.values[start + 3] = (val_abs >> 3 * U::N_BITS).into_trunc();
                self.values[start + 4] = (val_abs >> 4 * U::N_BITS).into_trunc();
                self.values[start + 5] = (val_abs >> 5 * U::N_BITS).into_trunc();
                self.values[start + 6] = (val_abs >> 6 * U::N_BITS).into_trunc();
                self.values[start + 7] = (val_abs >> 7 * U::N_BITS).into_trunc();
                self.values[start + 8] = (val_abs >> 8 * U::N_BITS).into_trunc();
                self.values[start + 9] = (val_abs >> 9 * U::N_BITS).into_trunc();
                self.values[start + 10] = (val_abs >> 10 * U::N_BITS).into_trunc();
                self.values[start + 11] = (val_abs >> 11 * U::N_BITS).into_trunc();
            },
            13 => {
                self.values[start + 1] = (val_abs >> U::N_BITS).into_trunc();
                self.values[start + 2] = (val_abs >> 2 * U::N_BITS).into_trunc();
                self.values[start + 3] = (val_abs >> 3 * U::N_BITS).into_trunc();
                self.values[start + 4] = (val_abs >> 4 * U::N_BITS).into_trunc();
                self.values[start + 5] = (val_abs >> 5 * U::N_BITS).into_trunc();
                self.values[start + 6] = (val_abs >> 6 * U::N_BITS).into_trunc();
                self.values[start + 7] = (val_abs >> 7 * U::N_BITS).into_trunc();
                self.values[start + 8] = (val_abs >> 8 * U::N_BITS).into_trunc();
                self.values[start + 9] = (val_abs >> 9 * U::N_BITS).into_trunc();
                self.values[start + 10] = (val_abs >> 10 * U::N_BITS).into_trunc();
                self.values[start + 11] = (val_abs >> 11 * U::N_BITS).into_trunc();
                self.values[start + 12] = (val_abs >> 12 * U::N_BITS).into_trunc();
            },
            14 => {
                self.values[start + 1] = (val_abs >> U::N_BITS).into_trunc();
                self.values[start + 2] = (val_abs >> 2 * U::N_BITS).into_trunc();
                self.values[start + 3] = (val_abs >> 3 * U::N_BITS).into_trunc();
                self.values[start + 4] = (val_abs >> 4 * U::N_BITS).into_trunc();
                self.values[start + 5] = (val_abs >> 5 * U::N_BITS).into_trunc();
                self.values[start + 6] = (val_abs >> 6 * U::N_BITS).into_trunc();
                self.values[start + 7] = (val_abs >> 7 * U::N_BITS).into_trunc();
                self.values[start + 8] = (val_abs >> 8 * U::N_BITS).into_trunc();
                self.values[start + 9] = (val_abs >> 9 * U::N_BITS).into_trunc();
                self.values[start + 10] = (val_abs >> 10 * U::N_BITS).into_trunc();
                self.values[start + 11] = (val_abs >> 11 * U::N_BITS).into_trunc();
                self.values[start + 12] = (val_abs >> 12 * U::N_BITS).into_trunc();
                self.values[start + 13] = (val_abs >> 13 * U::N_BITS).into_trunc();
            },
            15 => {
                self.values[start + 1] = (val_abs >> U::N_BITS).into_trunc();
                self.values[start + 2] = (val_abs >> 2 * U::N_BITS).into_trunc();
                self.values[start + 3] = (val_abs >> 3 * U::N_BITS).into_trunc();
                self.values[start + 4] = (val_abs >> 4 * U::N_BITS).into_trunc();
                self.values[start + 5] = (val_abs >> 5 * U::N_BITS).into_trunc();
                self.values[start + 6] = (val_abs >> 6 * U::N_BITS).into_trunc();
                self.values[start + 7] = (val_abs >> 7 * U::N_BITS).into_trunc();
                self.values[start + 8] = (val_abs >> 8 * U::N_BITS).into_trunc();
                self.values[start + 9] = (val_abs >> 9 * U::N_BITS).into_trunc();
                self.values[start + 10] = (val_abs >> 10 * U::N_BITS).into_trunc();
                self.values[start + 11] = (val_abs >> 11 * U::N_BITS).into_trunc();
                self.values[start + 12] = (val_abs >> 12 * U::N_BITS).into_trunc();
                self.values[start + 13] = (val_abs >> 13 * U::N_BITS).into_trunc();
                self.values[start + 14] = (val_abs >> 14 * U::N_BITS).into_trunc();
            },
            16 => {
                self.values[start + 1] = (val_abs >> U::N_BITS).into_trunc();
                self.values[start + 2] = (val_abs >> 2 * U::N_BITS).into_trunc();
                self.values[start + 3] = (val_abs >> 3 * U::N_BITS).into_trunc();
                self.values[start + 4] = (val_abs >> 4 * U::N_BITS).into_trunc();
                self.values[start + 5] = (val_abs >> 5 * U::N_BITS).into_trunc();
                self.values[start + 6] = (val_abs >> 6 * U::N_BITS).into_trunc();
                self.values[start + 7] = (val_abs >> 7 * U::N_BITS).into_trunc();
                self.values[start + 8] = (val_abs >> 8 * U::N_BITS).into_trunc();
                self.values[start + 9] = (val_abs >> 9 * U::N_BITS).into_trunc();
                self.values[start + 10] = (val_abs >> 10 * U::N_BITS).into_trunc();
                self.values[start + 11] = (val_abs >> 11 * U::N_BITS).into_trunc();
                self.values[start + 12] = (val_abs >> 12 * U::N_BITS).into_trunc();
                self.values[start + 13] = (val_abs >> 13 * U::N_BITS).into_trunc();
                self.values[start + 14] = (val_abs >> 14 * U::N_BITS).into_trunc();
                self.values[start + 15] = (val_abs >> 15 * U::N_BITS).into_trunc();
            },
            _ => {
                for i in 1..self.stride {
                    let tmp = val_abs >> (i * U::N_BITS);
                    self.values[start + i] = tmp.into_trunc();
                }
            }
        }
        if T::SIGNED {
            if val.is_neg() {
                self.values[start + self.stride - 1] |= U::ONE << (U::N_BITS - 1);
            }
        }
    }

    fn len(&self) -> usize {
        self.n
    }

    fn push(&mut self, val: T) {
        // Append zeros to the underlying data vec
        self.values.resize(self.values.len() + self.stride, U::ZERO);
        self.n += 1;
        let last = self.n - 1;
        self.set(last, val);
    }

    fn pop(&mut self) -> Option<T> {
        if self.n > 0 {
            self.n -= 1;
            let last = self.n - 1;
            return Some(self.get(last));
        }
        None
    }

    fn iterate(&self) -> impl Iterator<Item = T> {
        (0..self.n).map(|i| self.get(i))
    }

    fn clear(&mut self) {
        self.values.clear();
        self.n = 0;
        self.stride = 1;
        self.max = Self::Type::ZERO;
    }

    fn resize(&mut self, new_len: usize, value: Self::Type) {
        self.n = new_len;
        if new_len < self.len() {
            let len_values = new_len * self.stride;
            self.values.resize(len_values, U::ZERO);
        } else {
            for _i in self.len()..new_len {
                self.push(value);
            }
        }
    }

    fn reserve(&mut self, additional: usize) {
        let add = additional * self.stride;
        self.values.reserve(add);
    }
}

impl<T: Integer, U: Integer> From<Vec<T>> for ByteVec<T, U>
where
    T: FromTruncated<U> + IntoTruncated<U>,
{
    fn from(v: Vec<T>) -> Self {
        let val_max = v.iter().cloned().max().unwrap_or(T::ZERO);
        let mut ret = ByteVec::<T, U>::with_max_value(v.len(), val_max);
        for &x in v.iter() {
            ret.push(x);
        }
        ret
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn byte_vec_gen<T: Integer + From<u8>, U: Integer>(n: usize, mul: T, offset: T)
    where
        T: FromTruncated<U> + IntoTruncated<U>,
    {
        let mut vec_ref = vec![T::ZERO; n];
        for i in 0..n {
            vec_ref[i] = T::from((i % u8::MAX as usize) as u8) * mul + offset;
        }
        // Test copying from the reference vec
        let byte_vec = ByteVec::<T, U>::from(vec_ref.clone());
        for i in 0..n {
            assert_eq!(vec_ref[i], byte_vec.get(i));
        }
        // Test iterator
        assert_eq!(byte_vec.iterate().max(), vec_ref.iterate().max());
        for (v0, v1) in byte_vec.iterate().zip(vec_ref.iterate()) {
            assert_eq!(v0, v1);
        }
        // Test by pushing values
        let mut byte_vec = ByteVec::<T, U>::new();
        for i in 0..n {
            byte_vec.push(vec_ref[i]);
            assert_eq!(vec_ref[i], byte_vec.get(i));
        }
        assert_eq!(vec_ref.len(), byte_vec.len());
        // Test inserting values
        for i in (0..n).step_by(n / 16) {
            vec_ref.insert(i, vec_ref[i]);
            byte_vec.insert(i, byte_vec.get(i));
        }
        assert_eq!(vec_ref.len(), byte_vec.len());
        for i in 0..vec_ref.len() {
            assert_eq!(vec_ref[i], byte_vec.get(i));
        }
        // Test removing values
        for i in (0..n).step_by(n / 16) {
            let v0 = vec_ref.remove(i);
            let v1 = byte_vec.remove(i);
            assert_eq!(v0, v1);
        }
        assert_eq!(vec_ref.len(), byte_vec.len());
        for i in 0..vec_ref.len() {
            assert_eq!(vec_ref[i], byte_vec.get(i));
        }
    }
    
    #[test]
    fn byte_vec_len_one_insert() {
        // Test inserting a value at index 0 and 1 for a Vec / ByteVec of length one.
        let mut vec_ref = vec![64];
        let mut byte_vec = ByteVec::<u64, u8>::from(vec_ref.clone());
        vec_ref.insert(0, 12);
        byte_vec.insert(0, 12);
        assert_eq!(vec_ref.get(0), byte_vec.get(0));
        assert_eq!(vec_ref.get(1), byte_vec.get(1));
        vec_ref.resize(1, 0);
        byte_vec.resize(1, 0);
        assert_eq!(vec_ref.get(0), byte_vec.get(0));
        assert_eq!(vec_ref.len(), byte_vec.len());
        vec_ref.insert(1, 24);
        byte_vec.insert(1, 24);
        assert_eq!(vec_ref.get(0), byte_vec.get(0));
        assert_eq!(vec_ref.get(1), byte_vec.get(1));
    }

    #[test]
    fn byte_vec_u16_u8() {
        byte_vec_gen::<u16, u8>(1_000, 1, 0);
        byte_vec_gen::<u16, u8>(1_000, 2, 0);
        byte_vec_gen::<u16, u8>(1_000, 4, 0);
        byte_vec_gen::<u16, u8>(1_000, 8, 0);
        byte_vec_gen::<u16, u8>(1_000, 16, 0);
        byte_vec_gen::<u16, u8>(1_000, 32, 0);
        byte_vec_gen::<u16, u8>(1_000, 128, 0);
        byte_vec_gen::<u16, u8>(4_000, 192, 0);
    }

    #[test]
    fn byte_vec_u32_u8() {
        byte_vec_gen::<u32, u8>(1_000, 1, 0);
        byte_vec_gen::<u32, u8>(1_000, 2, 0);
        byte_vec_gen::<u32, u8>(1_000, 4, 0);
        byte_vec_gen::<u32, u8>(1_000, 8, 0);
        byte_vec_gen::<u32, u8>(1_000, 16, 0);
        byte_vec_gen::<u32, u8>(1_000, 128, 0);
        byte_vec_gen::<u32, u8>(4_000, 256, 0);
        byte_vec_gen::<u32, u8>(4_000, 512, 0);
        byte_vec_gen::<u32, u8>(4_000, 1024, 0);
        byte_vec_gen::<u32, u8>(16_000, 2048, 0);
        byte_vec_gen::<u32, u8>(16_000, 4096, 0);
        byte_vec_gen::<u32, u8>(16_000, 65_000, 0);
        byte_vec_gen::<u32, u8>(128_000, 520_000, 0);
        byte_vec_gen::<u32, u8>(128_000, 1_000_000, 0);
        byte_vec_gen::<u32, u8>(128_000, 8_000_000, 0);
    }

    #[test]
    fn byte_vec_u32_u16() {
        byte_vec_gen::<u32, u16>(1_000, 1, 0);
        byte_vec_gen::<u32, u16>(1_000, 2, 0);
        byte_vec_gen::<u32, u16>(1_000, 4, 0);
        byte_vec_gen::<u32, u16>(1_000, 8, 0);
        byte_vec_gen::<u32, u16>(1_000, 16, 0);
        byte_vec_gen::<u32, u16>(1_000, 128, 0);
        byte_vec_gen::<u32, u16>(4_000, 256, 0);
        byte_vec_gen::<u32, u16>(4_000, 512, 0);
        byte_vec_gen::<u32, u16>(4_000, 1024, 0);
        byte_vec_gen::<u32, u16>(16_000, 2048, 0);
        byte_vec_gen::<u32, u16>(16_000, 4096, 0);
        byte_vec_gen::<u32, u16>(16_000, 65_000, 0);
        byte_vec_gen::<u32, u16>(128_000, 520_000, 0);
        byte_vec_gen::<u32, u16>(128_000, 1_000_000, 0);
        byte_vec_gen::<u32, u16>(128_000, 8_000_000, 0);
    }

    #[test]
    fn byte_vec_u64_u8() {
        byte_vec_gen::<u64, u8>(1_000, 1, 0);
        byte_vec_gen::<u64, u8>(1_000, 2, 0);
        byte_vec_gen::<u64, u8>(1_000, 4, 0);
        byte_vec_gen::<u64, u8>(1_000, 8, 0);
        byte_vec_gen::<u64, u8>(1_000, 16, 0);
        byte_vec_gen::<u64, u8>(1_000, 32, 0);
        byte_vec_gen::<u64, u8>(4_000, 32, 0);
        byte_vec_gen::<u64, u8>(4_000, 64, 0);
        byte_vec_gen::<u64, u8>(4_000, 128, 0);
        byte_vec_gen::<u64, u8>(16_000, 256, 0);
        byte_vec_gen::<u64, u8>(16_000, 512, 0);
        byte_vec_gen::<u64, u8>(16_000, 1024, 0);
        byte_vec_gen::<u64, u8>(128_000, 4096, 0);
        byte_vec_gen::<u64, u8>(128_000, 65_000, 0);
        byte_vec_gen::<u64, u8>(128_000, 520_000, 0);
        byte_vec_gen::<u64, u8>(1_000_000, 1_000_000, 0);
        byte_vec_gen::<u64, u8>(1_000_000, 16_000_000, 0);
        byte_vec_gen::<u64, u8>(1_000_000, 8_000_000_000, 0);
    }

    #[test]
    fn byte_vec_u64_u16() {
        byte_vec_gen::<u64, u16>(1_000, 1, 0);
        byte_vec_gen::<u64, u16>(1_000, 2, 0);
        byte_vec_gen::<u64, u16>(1_000, 4, 0);
        byte_vec_gen::<u64, u16>(1_000, 8, 0);
        byte_vec_gen::<u64, u16>(1_000, 16, 0);
        byte_vec_gen::<u64, u16>(1_000, 24, 0);
        byte_vec_gen::<u64, u16>(4_000, 32, 0);
        byte_vec_gen::<u64, u16>(4_000, 64, 0);
        byte_vec_gen::<u64, u16>(4_000, 128, 0);
        byte_vec_gen::<u64, u16>(16_000, 256, 0);
        byte_vec_gen::<u64, u16>(16_000, 512, 0);
        byte_vec_gen::<u64, u16>(16_000, 1024, 0);
        byte_vec_gen::<u64, u16>(128_000, 4096, 0);
        byte_vec_gen::<u64, u16>(128_000, 65_000, 0);
        byte_vec_gen::<u64, u16>(128_000, 520_000, 0);
        byte_vec_gen::<u64, u16>(1_000_000, 1_000_000, 0);
        byte_vec_gen::<u64, u16>(1_000_000, 16_000_000, 0);
        byte_vec_gen::<u64, u16>(1_000_000, 8_000_000_000, 0);
    }

    #[test]
    fn byte_vec_u64_u32() {
        byte_vec_gen::<u64, u32>(1_000, 1, 0);
        byte_vec_gen::<u64, u32>(1_000, 2, 0);
        byte_vec_gen::<u64, u32>(1_000, 4, 0);
        byte_vec_gen::<u64, u32>(1_000, 8, 0);
        byte_vec_gen::<u64, u32>(1_000, 16, 0);
        byte_vec_gen::<u64, u32>(1_000, 24, 0);
        byte_vec_gen::<u64, u32>(4_000, 32, 0);
        byte_vec_gen::<u64, u32>(4_000, 64, 0);
        byte_vec_gen::<u64, u32>(4_000, 128, 0);
        byte_vec_gen::<u64, u32>(16_000, 256, 0);
        byte_vec_gen::<u64, u32>(16_000, 512, 0);
        byte_vec_gen::<u64, u32>(16_000, 1024, 0);
        byte_vec_gen::<u64, u32>(128_000, 4096, 0);
        byte_vec_gen::<u64, u32>(128_000, 65_000, 0);
        byte_vec_gen::<u64, u32>(128_000, 520_000, 0);
        byte_vec_gen::<u64, u32>(1_000_000, 1_000_000, 0);
        byte_vec_gen::<u64, u32>(1_000_000, 16_000_000, 0);
        byte_vec_gen::<u64, u32>(1_000_000, 8_000_000_000, 0);
    }

    #[test]
    fn byte_vec_u128_u8() {
        byte_vec_gen::<u128, u8>(1_000, 1, 0);
        byte_vec_gen::<u128, u8>(1_000, 2, 0);
        byte_vec_gen::<u128, u8>(1_000, 4, 0);
        byte_vec_gen::<u128, u8>(1_000, 8, 0);
        byte_vec_gen::<u128, u8>(1_000, 16, 0);
        byte_vec_gen::<u128, u8>(1_000, 32, 0);
        byte_vec_gen::<u128, u8>(4_000, 32, 0);
        byte_vec_gen::<u128, u8>(4_000, 64, 0);
        byte_vec_gen::<u128, u8>(4_000, 128, 0);
        byte_vec_gen::<u128, u8>(16_000, 256, 0);
        byte_vec_gen::<u128, u8>(16_000, 512, 0);
        byte_vec_gen::<u128, u8>(16_000, 1024, 0);
        byte_vec_gen::<u128, u8>(128_000, 4096, 0);
        byte_vec_gen::<u128, u8>(128_000, 65_000, 0);
        byte_vec_gen::<u128, u8>(128_000, 520_000, 0);
        byte_vec_gen::<u128, u8>(1_000_000, 1_000_000, 0);
        byte_vec_gen::<u128, u8>(1_000_000, 16_000_000, 0);
        byte_vec_gen::<u128, u8>(1_000_000, 8_000_000_000, 0);
        byte_vec_gen::<u128, u8>(1_000_000, 16_000_000_000_000_000_000, 0);
    }

    #[test]
    fn byte_vec_u128_u16() {
        byte_vec_gen::<u128, u16>(1_000, 1, 0);
        byte_vec_gen::<u128, u16>(1_000, 2, 0);
        byte_vec_gen::<u128, u16>(1_000, 4, 0);
        byte_vec_gen::<u128, u16>(1_000, 8, 0);
        byte_vec_gen::<u128, u16>(1_000, 16, 0);
        byte_vec_gen::<u128, u16>(1_000, 24, 0);
        byte_vec_gen::<u128, u16>(4_000, 32, 0);
        byte_vec_gen::<u128, u16>(4_000, 64, 0);
        byte_vec_gen::<u128, u16>(4_000, 128, 0);
        byte_vec_gen::<u128, u16>(16_000, 256, 0);
        byte_vec_gen::<u128, u16>(16_000, 512, 0);
        byte_vec_gen::<u128, u16>(16_000, 1024, 0);
        byte_vec_gen::<u128, u16>(128_000, 4096, 0);
        byte_vec_gen::<u128, u16>(128_000, 65_000, 0);
        byte_vec_gen::<u128, u16>(128_000, 520_000, 0);
        byte_vec_gen::<u128, u16>(1_000_000, 1_000_000, 0);
        byte_vec_gen::<u128, u16>(1_000_000, 16_000_000, 0);
        byte_vec_gen::<u128, u16>(1_000_000, 8_000_000_000, 0);
        byte_vec_gen::<u128, u16>(1_000_000, 16_000_000_000_000_000_000, 0);
    }

    #[test]
    fn byte_vec_u128_u32() {
        byte_vec_gen::<u128, u32>(1_000, 1, 0);
        byte_vec_gen::<u128, u32>(1_000, 2, 0);
        byte_vec_gen::<u128, u32>(1_000, 4, 0);
        byte_vec_gen::<u128, u32>(1_000, 8, 0);
        byte_vec_gen::<u128, u32>(1_000, 16, 0);
        byte_vec_gen::<u128, u32>(1_000, 24, 0);
        byte_vec_gen::<u128, u32>(4_000, 32, 0);
        byte_vec_gen::<u128, u32>(4_000, 64, 0);
        byte_vec_gen::<u128, u32>(4_000, 128, 0);
        byte_vec_gen::<u128, u32>(16_000, 256, 0);
        byte_vec_gen::<u128, u32>(16_000, 512, 0);
        byte_vec_gen::<u128, u32>(16_000, 1024, 0);
        byte_vec_gen::<u128, u32>(128_000, 4096, 0);
        byte_vec_gen::<u128, u32>(128_000, 65_000, 0);
        byte_vec_gen::<u128, u32>(128_000, 520_000, 0);
        byte_vec_gen::<u128, u32>(1_000_000, 1_000_000, 0);
        byte_vec_gen::<u128, u32>(1_000_000, 16_000_000, 0);
        byte_vec_gen::<u128, u32>(1_000_000, 8_000_000_000, 0);
        byte_vec_gen::<u128, u32>(1_000_000, 16_000_000_000_000_000_000, 0);
    }

    #[test]
    fn byte_vec_u128_u64() {
        byte_vec_gen::<u128, u64>(1_000, 1, 0);
        byte_vec_gen::<u128, u64>(1_000, 2, 0);
        byte_vec_gen::<u128, u64>(1_000, 4, 0);
        byte_vec_gen::<u128, u64>(1_000, 8, 0);
        byte_vec_gen::<u128, u64>(1_000, 16, 0);
        byte_vec_gen::<u128, u64>(1_000, 24, 0);
        byte_vec_gen::<u128, u64>(4_000, 32, 0);
        byte_vec_gen::<u128, u64>(4_000, 64, 0);
        byte_vec_gen::<u128, u64>(4_000, 128, 0);
        byte_vec_gen::<u128, u64>(16_000, 256, 0);
        byte_vec_gen::<u128, u64>(16_000, 512, 0);
        byte_vec_gen::<u128, u64>(16_000, 1024, 0);
        byte_vec_gen::<u128, u64>(128_000, 4096, 0);
        byte_vec_gen::<u128, u64>(128_000, 65_000, 0);
        byte_vec_gen::<u128, u64>(128_000, 520_000, 0);
        byte_vec_gen::<u128, u64>(1_000_000, 1_000_000, 0);
        byte_vec_gen::<u128, u64>(1_000_000, 16_000_000, 0);
        byte_vec_gen::<u128, u64>(1_000_000, 8_000_000_000, 0);
        byte_vec_gen::<u128, u64>(1_000_000, 16_000_000_000_000_000_000, 0);
    }

    #[test]
    fn byte_vec_usize() {
        byte_vec_gen::<usize, u8>(128_000, 32, 0);
        byte_vec_gen::<usize, u8>(128_000, 64, 0);
        byte_vec_gen::<usize, u8>(128_000, 128, 0);
        byte_vec_gen::<usize, u16>(128_000, 32, 0);
        byte_vec_gen::<usize, u16>(128_000, 64, 0);
        byte_vec_gen::<usize, u16>(128_000, 128, 0);
        byte_vec_gen::<usize, u32>(128_000, 256, 0);
        byte_vec_gen::<usize, u32>(128_000, 1024, 0);
        byte_vec_gen::<usize, u32>(128_000, 4096, 0);
        byte_vec_gen::<usize, u32>(128_000, 1_000_000, 0);
    }

    #[test]
    fn byte_vec_i16_u8() {
        byte_vec_gen::<i16, u8>(1_000, 1, -500);
        byte_vec_gen::<i16, u8>(1_000, 2, -600);
        byte_vec_gen::<i16, u8>(1_000, 4, -700);
        byte_vec_gen::<i16, u8>(1_000, 8, -800);
        byte_vec_gen::<i16, u8>(1_000, 16, -900);
        byte_vec_gen::<i16, u8>(1_000, 24, -950);
        byte_vec_gen::<i16, u8>(1_000, 32, -975);
        byte_vec_gen::<i16, u8>(4_000, 64, -5_000);
    }

    #[test]
    fn byte_vec_i32_u8() {
        byte_vec_gen::<i32, u8>(1_000, 1, -500);
        byte_vec_gen::<i32, u8>(1_000, 2, -600);
        byte_vec_gen::<i32, u8>(1_000, 4, -700);
        byte_vec_gen::<i32, u8>(1_000, 8, -800);
        byte_vec_gen::<i32, u8>(1_000, 16, -900);
        byte_vec_gen::<i32, u8>(1_000, 24, -950);
        byte_vec_gen::<i32, u8>(1_000, 32, -975);
        byte_vec_gen::<i32, u8>(4_000, 128, -5_000);
        byte_vec_gen::<i32, u8>(4_000, 256, -6_000);
        byte_vec_gen::<i32, u8>(4_000, 512, -7_000);
        byte_vec_gen::<i32, u8>(16_000, 1024, -50_000);
        byte_vec_gen::<i32, u8>(16_000, 2048, -60_000);
        byte_vec_gen::<i32, u8>(16_000, 4096, -70_000);
        byte_vec_gen::<i32, u8>(128_000, 65_000, -500_000);
        byte_vec_gen::<i32, u8>(128_000, 520_000, -600_000);
        byte_vec_gen::<i32, u8>(128_000, 8_000_000, -700_000);
    }

    #[test]
    fn byte_vec_i32_u16() {
        byte_vec_gen::<i32, u16>(1_000, 1, -500);
        byte_vec_gen::<i32, u16>(1_000, 2, -600);
        byte_vec_gen::<i32, u16>(1_000, 4, -700);
        byte_vec_gen::<i32, u16>(1_000, 8, -800);
        byte_vec_gen::<i32, u16>(1_000, 16, -900);
        byte_vec_gen::<i32, u16>(1_000, 24, -950);
        byte_vec_gen::<i32, u16>(4_000, 128, -5_000);
        byte_vec_gen::<i32, u16>(4_000, 256, -6_000);
        byte_vec_gen::<i32, u16>(4_000, 512, -7_000);
        byte_vec_gen::<i32, u16>(16_000, 1024, -50_000);
        byte_vec_gen::<i32, u16>(16_000, 2048, -60_000);
        byte_vec_gen::<i32, u16>(16_000, 4096, -70_000);
        byte_vec_gen::<i32, u16>(128_000, 65_000, -500_000);
        byte_vec_gen::<i32, u16>(128_000, 520_000, -600_000);
        byte_vec_gen::<i32, u16>(128_000, 8_000_000, -700_000);
    }

    #[test]
    fn byte_vec_i64_u8() {
        byte_vec_gen::<i64, u8>(1_000, 1, -500);
        byte_vec_gen::<i64, u8>(1_000, 2, -600);
        byte_vec_gen::<i64, u8>(1_000, 4, -700);
        byte_vec_gen::<i64, u8>(1_000, 8, -800);
        byte_vec_gen::<i64, u8>(1_000, 16, -900);
        byte_vec_gen::<i64, u8>(1_000, 32, -950);
        byte_vec_gen::<i64, u8>(4_000, 64, -5_000);
        byte_vec_gen::<i64, u8>(4_000, 128, -6_000);
        byte_vec_gen::<i64, u8>(4_000, 256, -7_000);
        byte_vec_gen::<i64, u8>(16_000, 1024, -50_000);
        byte_vec_gen::<i64, u8>(16_000, 2048, -60_000);
        byte_vec_gen::<i64, u8>(16_000, 4096, -70_000);
        byte_vec_gen::<i64, u8>(128_000, 65_000, -500_000);
        byte_vec_gen::<i64, u8>(128_000, 130_000, -600_000);
        byte_vec_gen::<i64, u8>(128_000, 260_000, -700_000);
        byte_vec_gen::<i64, u8>(1_000_000, 1_000_000, -5_000_000);
        byte_vec_gen::<i64, u8>(1_000_000, 400_000_000, -6_000_000);
        byte_vec_gen::<i64, u8>(1_000_000, 8_000_000_000, -7_000_000);
    }

    #[test]
    fn byte_vec_i64_u16() {
        byte_vec_gen::<i64, u16>(1_000, 1, -500);
        byte_vec_gen::<i64, u16>(1_000, 2, -600);
        byte_vec_gen::<i64, u16>(1_000, 4, -700);
        byte_vec_gen::<i64, u16>(1_000, 8, -800);
        byte_vec_gen::<i64, u16>(1_000, 16, -900);
        byte_vec_gen::<i64, u16>(1_000, 24, -950);
        byte_vec_gen::<i64, u16>(4_000, 32, -5_000);
        byte_vec_gen::<i64, u16>(4_000, 64, -6_000);
        byte_vec_gen::<i64, u16>(4_000, 128, -7_000);
        byte_vec_gen::<i64, u16>(16_000, 1024, -50_000);
        byte_vec_gen::<i64, u16>(16_000, 2048, -60_000);
        byte_vec_gen::<i64, u16>(16_000, 4096, -70_000);
        byte_vec_gen::<i64, u16>(128_000, 65_000, -500_000);
        byte_vec_gen::<i64, u16>(128_000, 130_000, -600_000);
        byte_vec_gen::<i64, u16>(128_000, 260_000, -700_000);
        byte_vec_gen::<i64, u16>(1_000_000, 1_000_000, -5_000_000);
        byte_vec_gen::<i64, u16>(1_000_000, 400_000_000, -6_000_000);
        byte_vec_gen::<i64, u16>(1_000_000, 8_000_000_000, -7_000_000);
    }

    #[test]
    fn byte_vec_i64_u32() {
        byte_vec_gen::<i64, u32>(1_000, 1, -500);
        byte_vec_gen::<i64, u32>(1_000, 2, -600);
        byte_vec_gen::<i64, u32>(1_000, 4, -700);
        byte_vec_gen::<i64, u32>(1_000, 8, -800);
        byte_vec_gen::<i64, u32>(1_000, 16, -900);
        byte_vec_gen::<i64, u32>(1_000, 24, -950);
        byte_vec_gen::<i64, u32>(4_000, 32, -5_000);
        byte_vec_gen::<i64, u32>(4_000, 64, -6_000);
        byte_vec_gen::<i64, u32>(4_000, 128, -7_000);
        byte_vec_gen::<i64, u32>(16_000, 1024, -50_000);
        byte_vec_gen::<i64, u32>(16_000, 2048, -60_000);
        byte_vec_gen::<i64, u32>(16_000, 4096, -70_000);
        byte_vec_gen::<i64, u32>(128_000, 65_000, -500_000);
        byte_vec_gen::<i64, u32>(128_000, 130_000, -600_000);
        byte_vec_gen::<i64, u32>(128_000, 260_000, -700_000);
        byte_vec_gen::<i64, u32>(1_000_000, 1_000_000, -5_000_000);
        byte_vec_gen::<i64, u32>(1_000_000, 400_000_000, -6_000_000);
        byte_vec_gen::<i64, u32>(1_000_000, 8_000_000_000, -7_000_000);
    }

    #[test]
    fn byte_vec_i128_u8() {
        byte_vec_gen::<i128, u8>(1_000, 1, -500);
        byte_vec_gen::<i128, u8>(1_000, 2, -600);
        byte_vec_gen::<i128, u8>(1_000, 4, -700);
        byte_vec_gen::<i128, u8>(1_000, 8, -800);
        byte_vec_gen::<i128, u8>(1_000, 16, -900);
        byte_vec_gen::<i128, u8>(1_000, 32, -950);
        byte_vec_gen::<i128, u8>(4_000, 64, -5_000);
        byte_vec_gen::<i128, u8>(4_000, 128, -6_000);
        byte_vec_gen::<i128, u8>(4_000, 256, -7_000);
        byte_vec_gen::<i128, u8>(16_000, 1024, -50_000);
        byte_vec_gen::<i128, u8>(16_000, 2048, -60_000);
        byte_vec_gen::<i128, u8>(16_000, 4096, -70_000);
        byte_vec_gen::<i128, u8>(128_000, 65_000, -500_000);
        byte_vec_gen::<i128, u8>(128_000, 130_000, -600_000);
        byte_vec_gen::<i128, u8>(128_000, 260_000, -700_000);
        byte_vec_gen::<i128, u8>(1_000_000, 1_000_000, -5_000_000);
        byte_vec_gen::<i128, u8>(1_000_000, 400_000_000, -6_000_000);
        byte_vec_gen::<i128, u8>(1_000_000, 8_000_000_000, -7_000_000);
        byte_vec_gen::<i128, u8>(1_000_000, 16_000_000_000_000_000_000, -8_000_000_000_000_000);
    }

    #[test]
    fn byte_vec_i128_u16() {
        byte_vec_gen::<i128, u16>(1_000, 1, -500);
        byte_vec_gen::<i128, u16>(1_000, 2, -600);
        byte_vec_gen::<i128, u16>(1_000, 4, -700);
        byte_vec_gen::<i128, u16>(1_000, 8, -800);
        byte_vec_gen::<i128, u16>(1_000, 16, -900);
        byte_vec_gen::<i128, u16>(1_000, 24, -950);
        byte_vec_gen::<i128, u16>(4_000, 32, -5_000);
        byte_vec_gen::<i128, u16>(4_000, 64, -6_000);
        byte_vec_gen::<i128, u16>(4_000, 128, -7_000);
        byte_vec_gen::<i128, u16>(16_000, 1024, -50_000);
        byte_vec_gen::<i128, u16>(16_000, 2048, -60_000);
        byte_vec_gen::<i128, u16>(16_000, 4096, -70_000);
        byte_vec_gen::<i128, u16>(128_000, 65_000, -500_000);
        byte_vec_gen::<i128, u16>(128_000, 130_000, -600_000);
        byte_vec_gen::<i128, u16>(128_000, 260_000, -700_000);
        byte_vec_gen::<i128, u16>(1_000_000, 1_000_000, -5_000_000);
        byte_vec_gen::<i128, u16>(1_000_000, 400_000_000, -6_000_000);
        byte_vec_gen::<i128, u16>(1_000_000, 8_000_000_000, -7_000_000);
        byte_vec_gen::<i128, u16>(1_000_000, 16_000_000_000_000_000_000, -8_000_000_000_000_000);
    }

    #[test]
    fn byte_vec_i128_u32() {
        byte_vec_gen::<i128, u32>(1_000, 1, -500);
        byte_vec_gen::<i128, u32>(1_000, 2, -600);
        byte_vec_gen::<i128, u32>(1_000, 4, -700);
        byte_vec_gen::<i128, u32>(1_000, 8, -800);
        byte_vec_gen::<i128, u32>(1_000, 16, -900);
        byte_vec_gen::<i128, u32>(1_000, 24, -950);
        byte_vec_gen::<i128, u32>(4_000, 32, -5_000);
        byte_vec_gen::<i128, u32>(4_000, 64, -6_000);
        byte_vec_gen::<i128, u32>(4_000, 128, -7_000);
        byte_vec_gen::<i128, u32>(16_000, 1024, -50_000);
        byte_vec_gen::<i128, u32>(16_000, 2048, -60_000);
        byte_vec_gen::<i128, u32>(16_000, 4096, -70_000);
        byte_vec_gen::<i128, u32>(128_000, 65_000, -500_000);
        byte_vec_gen::<i128, u32>(128_000, 130_000, -600_000);
        byte_vec_gen::<i128, u32>(128_000, 260_000, -700_000);
        byte_vec_gen::<i128, u32>(1_000_000, 1_000_000, -5_000_000);
        byte_vec_gen::<i128, u32>(1_000_000, 400_000_000, -6_000_000);
        byte_vec_gen::<i128, u32>(1_000_000, 8_000_000_000, -7_000_000);
        byte_vec_gen::<i128, u32>(1_000_000, 16_000_000_000_000_000_000, -8_000_000_000_000_000);
    }

    #[test]
    fn byte_vec_i128_u64() {
        byte_vec_gen::<i128, u64>(1_000, 1, -500);
        byte_vec_gen::<i128, u64>(1_000, 2, -600);
        byte_vec_gen::<i128, u64>(1_000, 4, -700);
        byte_vec_gen::<i128, u64>(1_000, 8, -800);
        byte_vec_gen::<i128, u64>(1_000, 16, -900);
        byte_vec_gen::<i128, u64>(1_000, 24, -950);
        byte_vec_gen::<i128, u64>(4_000, 32, -5_000);
        byte_vec_gen::<i128, u64>(4_000, 64, -6_000);
        byte_vec_gen::<i128, u64>(4_000, 128, -7_000);
        byte_vec_gen::<i128, u64>(16_000, 1024, -50_000);
        byte_vec_gen::<i128, u64>(16_000, 2048, -60_000);
        byte_vec_gen::<i128, u64>(16_000, 4096, -70_000);
        byte_vec_gen::<i128, u64>(128_000, 65_000, -500_000);
        byte_vec_gen::<i128, u64>(128_000, 130_000, -600_000);
        byte_vec_gen::<i128, u64>(128_000, 260_000, -700_000);
        byte_vec_gen::<i128, u64>(1_000_000, 1_000_000, -5_000_000);
        byte_vec_gen::<i128, u64>(1_000_000, 400_000_000, -6_000_000);
        byte_vec_gen::<i128, u64>(1_000_000, 8_000_000_000, -7_000_000);
        byte_vec_gen::<i128, u64>(1_000_000, 16_000_000_000_000_000_000, -8_000_000_000_000_000);
    }

    #[test]
    fn byte_vec_isize() {
        byte_vec_gen::<isize, u8>(128_000, 32, -500);
        byte_vec_gen::<isize, u8>(128_000, 64, -600);
        byte_vec_gen::<isize, u8>(128_000, 128, -700);
        byte_vec_gen::<isize, u16>(128_000, 32, -5_000);
        byte_vec_gen::<isize, u16>(128_000, 64, -6_000);
        byte_vec_gen::<isize, u16>(128_000, 128, -7_000);
        byte_vec_gen::<isize, u32>(128_000, 256, -50_000);
        byte_vec_gen::<isize, u32>(128_000, 1024, -60_000);
        byte_vec_gen::<isize, u32>(128_000, 4096, -70_000);
        byte_vec_gen::<isize, u32>(128_000, 1_000_000, -8_000_000);
    }
}
