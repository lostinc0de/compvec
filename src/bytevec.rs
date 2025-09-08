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
    shifts: Vec<usize>,
}

impl<T: Integer, U: Integer> ByteVec<T, U>
where
    T: FromTruncated<U> + IntoTruncated<U>,
{
    /// Returns the maximum value and the new stride, such that val fits.
    fn compute_max_value_and_stride(val: T) -> (T, usize) {
        // For signed integers leave out the signed bit
        let mul: T = if T::SIGNED {
            T::from_trunc(U::MAX / U::TWO)
        } else {
            T::from_trunc(U::MAX)
        };
        let mut max: T = mul;
        let mut stride: usize = 1;
        while max < val.abs() {
            max *= mul;
            stride += 1;
        }
        (max, stride)
    }

    /// Resizes the underlying Vec of smaller integers, if val is too large.
    fn check_and_resize(&mut self, val: T) {
        if val.abs() > self.max {
            let (max_new, stride_new) = Self::compute_max_value_and_stride(val);
            let mut values_new = Vec::<U>::with_capacity(stride_new * self.len());
            let n_zeros_fill = stride_new - self.stride;
            for c in self.values.chunks_exact(self.stride) {
                for &x in c {
                    values_new.push(x);
                }
                for _ in 0..n_zeros_fill {
                    values_new.push(U::ZERO);
                }
            }
            self.max = max_new;
            self.stride = stride_new;
            self.values = values_new;
        }
    }

    /// Returns a vec containing the number of shifts for extracting a value.
    fn shifts() -> Vec<usize> {
        let n_shifts = (T::N_BITS / U::N_BITS) as usize;
        let mut shifts = vec![0usize; n_shifts];
        for i in 1..n_shifts {
            shifts[i] = i * U::N_BITS;
        }
        shifts
    }

    /// Returns a byte vec container with a maximum value.
    /// Internal variables are set, such that the underlying
    /// data vec does not have to be reallocated.
    pub fn with_max_value(n: usize, val_max: T) -> Self {
        let (max, stride) = Self::compute_max_value_and_stride(val_max);
        Self {
            values: Vec::<U>::with_capacity(n),
            max,
            stride,
            n: 0,
            shifts: Self::shifts(),
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
        Self {
            values: Vec::<U>::with_capacity(n),
            max: T::from_trunc(U::MAX),
            stride: 1,
            n: 0,
            shifts: Self::shifts(),
        }
    }

    fn get(&self, i: usize) -> T {
        let start = i * self.stride;
        // Loop unrolling for performance
        match self.stride {
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
                    ret |= val << self.shifts[i];
                }
                ret
            }
        }
    }

    fn set(&mut self, i: usize, val: T) {
        self.check_and_resize(val);
        let start = i * self.stride;
        self.values[start] = val.into_trunc();
        // Loop unrolling for performance
        match self.stride {
            1 => {
            },
            2 => {
                self.values[start + 1] = (val >> U::N_BITS).into_trunc();
            },
            3 => {
                self.values[start + 1] = (val >> U::N_BITS).into_trunc();
                self.values[start + 2] = (val >> 2 * U::N_BITS).into_trunc();
            },
            4 => {
                self.values[start + 1] = (val >> U::N_BITS).into_trunc();
                self.values[start + 2] = (val >> 2 * U::N_BITS).into_trunc();
                self.values[start + 3] = (val >> 3 * U::N_BITS).into_trunc();
            },
            5 => {
                self.values[start + 1] = (val >> U::N_BITS).into_trunc();
                self.values[start + 2] = (val >> 2 * U::N_BITS).into_trunc();
                self.values[start + 3] = (val >> 3 * U::N_BITS).into_trunc();
                self.values[start + 4] = (val >> 4 * U::N_BITS).into_trunc();
            },
            6 => {
                self.values[start + 1] = (val >> U::N_BITS).into_trunc();
                self.values[start + 2] = (val >> 2 * U::N_BITS).into_trunc();
                self.values[start + 3] = (val >> 3 * U::N_BITS).into_trunc();
                self.values[start + 4] = (val >> 4 * U::N_BITS).into_trunc();
                self.values[start + 5] = (val >> 5 * U::N_BITS).into_trunc();
            },
            7 => {
                self.values[start + 1] = (val >> U::N_BITS).into_trunc();
                self.values[start + 2] = (val >> 2 * U::N_BITS).into_trunc();
                self.values[start + 3] = (val >> 3 * U::N_BITS).into_trunc();
                self.values[start + 4] = (val >> 4 * U::N_BITS).into_trunc();
                self.values[start + 5] = (val >> 5 * U::N_BITS).into_trunc();
                self.values[start + 6] = (val >> 6 * U::N_BITS).into_trunc();
            },
            8 => {
                self.values[start + 1] = (val >> U::N_BITS).into_trunc();
                self.values[start + 2] = (val >> 2 * U::N_BITS).into_trunc();
                self.values[start + 3] = (val >> 3 * U::N_BITS).into_trunc();
                self.values[start + 4] = (val >> 4 * U::N_BITS).into_trunc();
                self.values[start + 5] = (val >> 5 * U::N_BITS).into_trunc();
                self.values[start + 6] = (val >> 6 * U::N_BITS).into_trunc();
                self.values[start + 7] = (val >> 7 * U::N_BITS).into_trunc();
            },
            9 => {
                self.values[start + 1] = (val >> U::N_BITS).into_trunc();
                self.values[start + 2] = (val >> 2 * U::N_BITS).into_trunc();
                self.values[start + 3] = (val >> 3 * U::N_BITS).into_trunc();
                self.values[start + 4] = (val >> 4 * U::N_BITS).into_trunc();
                self.values[start + 5] = (val >> 5 * U::N_BITS).into_trunc();
                self.values[start + 6] = (val >> 6 * U::N_BITS).into_trunc();
                self.values[start + 7] = (val >> 7 * U::N_BITS).into_trunc();
                self.values[start + 8] = (val >> 8 * U::N_BITS).into_trunc();
            },
            10 => {
                self.values[start + 1] = (val >> U::N_BITS).into_trunc();
                self.values[start + 2] = (val >> 2 * U::N_BITS).into_trunc();
                self.values[start + 3] = (val >> 3 * U::N_BITS).into_trunc();
                self.values[start + 4] = (val >> 4 * U::N_BITS).into_trunc();
                self.values[start + 5] = (val >> 5 * U::N_BITS).into_trunc();
                self.values[start + 6] = (val >> 6 * U::N_BITS).into_trunc();
                self.values[start + 7] = (val >> 7 * U::N_BITS).into_trunc();
                self.values[start + 8] = (val >> 8 * U::N_BITS).into_trunc();
                self.values[start + 9] = (val >> 9 * U::N_BITS).into_trunc();
            },
            11 => {
                self.values[start + 1] = (val >> U::N_BITS).into_trunc();
                self.values[start + 2] = (val >> 2 * U::N_BITS).into_trunc();
                self.values[start + 3] = (val >> 3 * U::N_BITS).into_trunc();
                self.values[start + 4] = (val >> 4 * U::N_BITS).into_trunc();
                self.values[start + 5] = (val >> 5 * U::N_BITS).into_trunc();
                self.values[start + 6] = (val >> 6 * U::N_BITS).into_trunc();
                self.values[start + 7] = (val >> 7 * U::N_BITS).into_trunc();
                self.values[start + 8] = (val >> 8 * U::N_BITS).into_trunc();
                self.values[start + 9] = (val >> 9 * U::N_BITS).into_trunc();
                self.values[start + 10] = (val >> 10 * U::N_BITS).into_trunc();
            },
            12 => {
                self.values[start + 1] = (val >> U::N_BITS).into_trunc();
                self.values[start + 2] = (val >> 2 * U::N_BITS).into_trunc();
                self.values[start + 3] = (val >> 3 * U::N_BITS).into_trunc();
                self.values[start + 4] = (val >> 4 * U::N_BITS).into_trunc();
                self.values[start + 5] = (val >> 5 * U::N_BITS).into_trunc();
                self.values[start + 6] = (val >> 6 * U::N_BITS).into_trunc();
                self.values[start + 7] = (val >> 7 * U::N_BITS).into_trunc();
                self.values[start + 8] = (val >> 8 * U::N_BITS).into_trunc();
                self.values[start + 9] = (val >> 9 * U::N_BITS).into_trunc();
                self.values[start + 10] = (val >> 10 * U::N_BITS).into_trunc();
                self.values[start + 11] = (val >> 11 * U::N_BITS).into_trunc();
            },
            13 => {
                self.values[start + 1] = (val >> U::N_BITS).into_trunc();
                self.values[start + 2] = (val >> 2 * U::N_BITS).into_trunc();
                self.values[start + 3] = (val >> 3 * U::N_BITS).into_trunc();
                self.values[start + 4] = (val >> 4 * U::N_BITS).into_trunc();
                self.values[start + 5] = (val >> 5 * U::N_BITS).into_trunc();
                self.values[start + 6] = (val >> 6 * U::N_BITS).into_trunc();
                self.values[start + 7] = (val >> 7 * U::N_BITS).into_trunc();
                self.values[start + 8] = (val >> 8 * U::N_BITS).into_trunc();
                self.values[start + 9] = (val >> 9 * U::N_BITS).into_trunc();
                self.values[start + 10] = (val >> 10 * U::N_BITS).into_trunc();
                self.values[start + 11] = (val >> 11 * U::N_BITS).into_trunc();
                self.values[start + 12] = (val >> 12 * U::N_BITS).into_trunc();
            },
            14 => {
                self.values[start + 1] = (val >> U::N_BITS).into_trunc();
                self.values[start + 2] = (val >> 2 * U::N_BITS).into_trunc();
                self.values[start + 3] = (val >> 3 * U::N_BITS).into_trunc();
                self.values[start + 4] = (val >> 4 * U::N_BITS).into_trunc();
                self.values[start + 5] = (val >> 5 * U::N_BITS).into_trunc();
                self.values[start + 6] = (val >> 6 * U::N_BITS).into_trunc();
                self.values[start + 7] = (val >> 7 * U::N_BITS).into_trunc();
                self.values[start + 8] = (val >> 8 * U::N_BITS).into_trunc();
                self.values[start + 9] = (val >> 9 * U::N_BITS).into_trunc();
                self.values[start + 10] = (val >> 10 * U::N_BITS).into_trunc();
                self.values[start + 11] = (val >> 11 * U::N_BITS).into_trunc();
                self.values[start + 12] = (val >> 12 * U::N_BITS).into_trunc();
                self.values[start + 13] = (val >> 13 * U::N_BITS).into_trunc();
            },
            15 => {
                self.values[start + 1] = (val >> U::N_BITS).into_trunc();
                self.values[start + 2] = (val >> 2 * U::N_BITS).into_trunc();
                self.values[start + 3] = (val >> 3 * U::N_BITS).into_trunc();
                self.values[start + 4] = (val >> 4 * U::N_BITS).into_trunc();
                self.values[start + 5] = (val >> 5 * U::N_BITS).into_trunc();
                self.values[start + 6] = (val >> 6 * U::N_BITS).into_trunc();
                self.values[start + 7] = (val >> 7 * U::N_BITS).into_trunc();
                self.values[start + 8] = (val >> 8 * U::N_BITS).into_trunc();
                self.values[start + 9] = (val >> 9 * U::N_BITS).into_trunc();
                self.values[start + 10] = (val >> 10 * U::N_BITS).into_trunc();
                self.values[start + 11] = (val >> 11 * U::N_BITS).into_trunc();
                self.values[start + 12] = (val >> 12 * U::N_BITS).into_trunc();
                self.values[start + 13] = (val >> 13 * U::N_BITS).into_trunc();
                self.values[start + 14] = (val >> 14 * U::N_BITS).into_trunc();
            },
            16 => {
                self.values[start + 1] = (val >> U::N_BITS).into_trunc();
                self.values[start + 2] = (val >> 2 * U::N_BITS).into_trunc();
                self.values[start + 3] = (val >> 3 * U::N_BITS).into_trunc();
                self.values[start + 4] = (val >> 4 * U::N_BITS).into_trunc();
                self.values[start + 5] = (val >> 5 * U::N_BITS).into_trunc();
                self.values[start + 6] = (val >> 6 * U::N_BITS).into_trunc();
                self.values[start + 7] = (val >> 7 * U::N_BITS).into_trunc();
                self.values[start + 8] = (val >> 8 * U::N_BITS).into_trunc();
                self.values[start + 9] = (val >> 9 * U::N_BITS).into_trunc();
                self.values[start + 10] = (val >> 10 * U::N_BITS).into_trunc();
                self.values[start + 11] = (val >> 11 * U::N_BITS).into_trunc();
                self.values[start + 12] = (val >> 12 * U::N_BITS).into_trunc();
                self.values[start + 13] = (val >> 13 * U::N_BITS).into_trunc();
                self.values[start + 14] = (val >> 14 * U::N_BITS).into_trunc();
                self.values[start + 15] = (val >> 15 * U::N_BITS).into_trunc();
            },
            _ => {
                for i in 1..self.stride {
                    let tmp = val >> self.shifts[i];
                    self.values[start + i] = tmp.into_trunc();
                }
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
        let mut byte_vec_ref = vec![T::ZERO; n];
        for i in 0..n {
            byte_vec_ref[i] = T::from((i % u8::MAX as usize) as u8) * mul + offset;
        }
        // Test copying from the reference vec
        let byte_vec = ByteVec::<T, U>::from(byte_vec_ref.clone());
        for i in 0..n {
            assert_eq!(byte_vec_ref[i], byte_vec.get(i));
        }
        // Test by pushing values
        let mut byte_vec = ByteVec::<T, U>::new();
        for i in 0..n {
            byte_vec.push(byte_vec_ref[i]);
            assert_eq!(byte_vec_ref[i], byte_vec.get(i));
        }
        // Test inserting values
        for i in (0..n).step_by(n / 16) {
            byte_vec_ref.insert(i, byte_vec_ref[i]);
            byte_vec.insert(i, byte_vec.get(i));
        }
        assert_eq!(byte_vec_ref.len(), byte_vec.len());
        for i in 0..byte_vec_ref.len() {
            assert_eq!(byte_vec_ref[i], byte_vec.get(i));
        }
        // Test removing values
        for i in (0..n).step_by(n / 16) {
            let v0 = byte_vec_ref.remove(i);
            let v1 = byte_vec.remove(i);
            assert_eq!(v0, v1);
        }
        assert_eq!(byte_vec_ref.len(), byte_vec.len());
        for i in 0..byte_vec_ref.len() {
            assert_eq!(byte_vec_ref[i], byte_vec.get(i));
        }
    }

    #[test]
    fn byte_vec() {
        byte_vec_gen::<u16, u8>(1_000, 12, 0);
        byte_vec_gen::<u32, u8>(1_000, 24, 0);
        byte_vec_gen::<u64, u8>(1_000, 32, 0);
        byte_vec_gen::<u32, u16>(1_000, 24, 0);
        byte_vec_gen::<u64, u16>(1_000, 24, 0);
        byte_vec_gen::<u64, u32>(1_000, 24, 0);
        byte_vec_gen::<i16, u8>(1_000, 12, -12200);
        byte_vec_gen::<i32, u8>(1_000, 1, 0);
        byte_vec_gen::<i64, u8>(1_000, 1, 0);
        byte_vec_gen::<i32, u16>(1_000, 1, 0);
        byte_vec_gen::<i32, u16>(1_000, 1, 0);
        byte_vec_gen::<i64, u32>(1_000, 1, 0);
        byte_vec_gen::<usize, u8>(1_000, 2, 0);
        byte_vec_gen::<usize, u16>(1_000, 4, 0);
        byte_vec_gen::<isize, u8>(1_000, 1, 0);
        byte_vec_gen::<isize, u16>(1_000, 1, 0);
        byte_vec_gen::<u16, u8>(1_000, 1, 0);
        byte_vec_gen::<u16, u8>(1_000, 2, 0);
        byte_vec_gen::<u16, u8>(1_000, 4, 0);
        byte_vec_gen::<u16, u8>(1_000, 8, 0);
        byte_vec_gen::<u16, u8>(1_000, 16, 0);
        byte_vec_gen::<u32, u8>(1_000, 1, 0);
        byte_vec_gen::<u32, u8>(1_000, 2, 0);
        byte_vec_gen::<u32, u8>(1_000, 4, 0);
        byte_vec_gen::<u32, u8>(1_000, 8, 0);
        byte_vec_gen::<u32, u8>(1_000, 16, 0);
        byte_vec_gen::<u32, u8>(10_000, 32, 0);
        byte_vec_gen::<u32, u8>(10_000, 64, 0);
        byte_vec_gen::<u32, u8>(10_000, 128, 0);
        byte_vec_gen::<u32, u8>(100_000, 32, 0);
        byte_vec_gen::<u32, u8>(100_000, 64, 0);
        byte_vec_gen::<u32, u8>(100_000, 128, 0);
        byte_vec_gen::<u32, u8>(1_000_000, 32, 0);
        byte_vec_gen::<u32, u8>(1_000_000, 64, 0);
        byte_vec_gen::<u32, u8>(1_000_000, 128, 0);
        byte_vec_gen::<u32, u16>(1_000, 1, 0);
        byte_vec_gen::<u32, u16>(1_000, 2, 0);
        byte_vec_gen::<u32, u16>(1_000, 4, 0);
        byte_vec_gen::<u32, u16>(1_000, 8, 0);
        byte_vec_gen::<u32, u16>(1_000, 16, 0);
        byte_vec_gen::<u32, u16>(10_000, 32, 0);
        byte_vec_gen::<u32, u16>(10_000, 64, 0);
        byte_vec_gen::<u32, u16>(10_000, 128, 0);
        byte_vec_gen::<u32, u16>(100_000, 32, 0);
        byte_vec_gen::<u32, u16>(100_000, 64, 0);
        byte_vec_gen::<u32, u16>(100_000, 128, 0);
        byte_vec_gen::<u32, u16>(1_000_000, 32, 0);
        byte_vec_gen::<u32, u16>(1_000_000, 64, 0);
        byte_vec_gen::<u32, u16>(1_000_000, 128, 0);
        byte_vec_gen::<u64, u8>(1_000, 1, 0);
        byte_vec_gen::<u64, u8>(1_000, 2, 0);
        byte_vec_gen::<u64, u8>(1_000, 4, 0);
        byte_vec_gen::<u64, u8>(1_000, 8, 0);
        byte_vec_gen::<u64, u8>(1_000, 16, 0);
        byte_vec_gen::<u64, u8>(10_000, 32, 0);
        byte_vec_gen::<u64, u8>(10_000, 64, 0);
        byte_vec_gen::<u64, u8>(10_000, 128, 0);
        byte_vec_gen::<u64, u8>(100_000, 32, 0);
        byte_vec_gen::<u64, u8>(100_000, 64, 0);
        byte_vec_gen::<u64, u8>(100_000, 128, 0);
        byte_vec_gen::<u64, u8>(1_000_000, 32, 0);
        byte_vec_gen::<u64, u8>(1_000_000, 64, 0);
        byte_vec_gen::<u64, u8>(1_000_000, 128, 0);
        byte_vec_gen::<u64, u16>(1_000, 1, 0);
        byte_vec_gen::<u64, u16>(1_000, 2, 0);
        byte_vec_gen::<u64, u16>(1_000, 4, 0);
        byte_vec_gen::<u64, u16>(1_000, 8, 0);
        byte_vec_gen::<u64, u16>(1_000, 16, 0);
        byte_vec_gen::<u64, u16>(10_000, 32, 0);
        byte_vec_gen::<u64, u16>(10_000, 64, 0);
        byte_vec_gen::<u64, u16>(10_000, 128, 0);
        byte_vec_gen::<u64, u16>(100_000, 32, 0);
        byte_vec_gen::<u64, u16>(100_000, 64, 0);
        byte_vec_gen::<u64, u16>(100_000, 128, 0);
        byte_vec_gen::<u64, u16>(1_000_000, 32, 0);
        byte_vec_gen::<u64, u16>(1_000_000, 64, 0);
        byte_vec_gen::<u64, u16>(1_000_000, 128, 0);
        byte_vec_gen::<u64, u32>(1_000, 1, 0);
        byte_vec_gen::<u64, u32>(1_000, 2, 0);
        byte_vec_gen::<u64, u32>(1_000, 4, 0);
        byte_vec_gen::<u64, u32>(1_000, 8, 0);
        byte_vec_gen::<u64, u32>(1_000, 16, 0);
        byte_vec_gen::<u64, u32>(10_000, 32, 0);
        byte_vec_gen::<u64, u32>(10_000, 64, 0);
        byte_vec_gen::<u64, u32>(10_000, 128, 0);
        byte_vec_gen::<u64, u32>(100_000, 32, 0);
        byte_vec_gen::<u64, u32>(100_000, 64, 0);
        byte_vec_gen::<u64, u32>(100_000, 128, 0);
        byte_vec_gen::<u64, u32>(1_000_000, 32, 0);
        byte_vec_gen::<u64, u32>(1_000_000, 64, 0);
        byte_vec_gen::<u64, u32>(1_000_000, 128, 0);
        byte_vec_gen::<i64, u8>(1_000, 1, 0);
        byte_vec_gen::<i64, u8>(1_000, 2, 0);
        byte_vec_gen::<i64, u8>(1_000, 4, 0);
        byte_vec_gen::<i64, u8>(1_000, 8, 0);
        byte_vec_gen::<i64, u8>(1_000, 16, 0);
        byte_vec_gen::<i64, u8>(10_000, 32, 0);
        byte_vec_gen::<i64, u8>(10_000, 64, 0);
        byte_vec_gen::<i64, u8>(10_000, 128, 0);
        byte_vec_gen::<i64, u8>(100_000, 32, 0);
        byte_vec_gen::<i64, u8>(100_000, 64, 0);
        byte_vec_gen::<i64, u8>(100_000, 128, 0);
        byte_vec_gen::<i64, u8>(1_000_000, 32, 0);
        byte_vec_gen::<i64, u8>(1_000_000, 64, 0);
        byte_vec_gen::<i64, u8>(1_000_000, 128, 0);
        byte_vec_gen::<i64, u16>(1_000, 1, 0);
        byte_vec_gen::<i64, u16>(1_000, 2, 0);
        byte_vec_gen::<i64, u16>(1_000, 4, 0);
        byte_vec_gen::<i64, u16>(1_000, 8, 0);
        byte_vec_gen::<i64, u16>(1_000, 16, 0);
        byte_vec_gen::<i64, u16>(10_000, 32, 0);
        byte_vec_gen::<i64, u16>(10_000, 64, 0);
        byte_vec_gen::<i64, u16>(10_000, 128, 0);
        byte_vec_gen::<i64, u16>(100_000, 32, 0);
        byte_vec_gen::<i64, u16>(100_000, 64, 0);
        byte_vec_gen::<i64, u16>(100_000, 128, 0);
        byte_vec_gen::<i64, u16>(1_000_000, 32, 0);
        byte_vec_gen::<i64, u16>(1_000_000, 64, 0);
        byte_vec_gen::<i64, u16>(1_000_000, 128, 0);
        byte_vec_gen::<i64, u32>(1_000, 1, 0);
        byte_vec_gen::<i64, u32>(1_000, 2, 0);
        byte_vec_gen::<i64, u32>(1_000, 4, 0);
        byte_vec_gen::<i64, u32>(1_000, 8, 0);
        byte_vec_gen::<i64, u32>(1_000, 16, 0);
        byte_vec_gen::<i64, u32>(10_000, 32, 0);
        byte_vec_gen::<i64, u32>(10_000, 64, 0);
        byte_vec_gen::<i64, u32>(10_000, 128, 0);
        byte_vec_gen::<i64, u32>(100_000, 32, 0);
        byte_vec_gen::<i64, u32>(100_000, 64, 0);
        byte_vec_gen::<i64, u32>(100_000, 128, 0);
        byte_vec_gen::<i64, u32>(1_000_000, 32, 0);
        byte_vec_gen::<i64, u32>(1_000_000, 64, 0);
        byte_vec_gen::<i64, u32>(1_000_000, 128, 0);
    }
}
