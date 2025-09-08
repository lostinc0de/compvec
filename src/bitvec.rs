use crate::types::*;
use crate::veclike::VecLike;

#[derive(Clone, Debug)]
pub struct BitVec<T: Integer, U: Integer>
where
    T: FromTruncated<U> + IntoTruncated<U>,
{
    values: Vec<U>,
    max: T,
    stride_bits: u64,
    n: usize,
    masks: Vec<U>,
}

impl<T: Integer, U: Integer> BitVec<T, U>
where
    T: FromTruncated<U> + IntoTruncated<U>,
    Self: VecLike
{
    /// Returns the maximum value and the new stride in bits, such that val fits.
    fn compute_max_value_and_stride(val: T) -> (T, u64) {
        let mut stride = 1;
        let mut max = (T::ONE << stride) - T::ONE;
        while max < val.abs() {
            stride += 1;
            max = (T::ONE << stride) - T::ONE;
        }
        // Reserve one additional bit for the sign if necessary
        if T::SIGNED {
            stride += 1;
        }
        (max, stride as u64)
    }
    
    /// Resizes the underlying Vec of smaller integers, if val is too large.
    fn check_and_resize(&mut self, val: T) {
        if val.abs() > self.max {
            let (max_new, stride_new) = Self::compute_max_value_and_stride(val);
            let cloned = self.clone();
            self.max = max_new;
            self.stride_bits = stride_new as u64;
            let last_bit_global = (self.n as u64 + 1) * self.stride_bits - 1;
            let n_values_new = last_bit_global / U::N_BITS_U64 + 1;
            self.values.resize(n_values_new as usize, U::ZERO);
            for i in 0..self.n {
                let v = cloned.get(i);
                self.set(i, v);
            }
        }
    }

    /// Returns the positions of the affected elements in the underlying values vec
    /// and the local start and end bit.
    pub fn comp_pos(&self, i: usize) -> (usize, usize, usize, usize) {
        let start_bit_global = (i as u64) * self.stride_bits;
        let end_bit_global = start_bit_global + self.stride_bits - 1;
        let start_el = start_bit_global / U::N_BITS_U64;
        let end_el = end_bit_global / U::N_BITS_U64;
        let start_bit = start_bit_global - start_el * U::N_BITS_U64;
        let end_bit = end_bit_global - end_el * U::N_BITS_U64;
        (start_el as usize, end_el as usize, start_bit as usize, end_bit as usize)
    }

    /// Returns a bit vec container with a maximum value.
    /// Internal variables are set, such that the underlying
    /// data vec does not have to be reallocated.
    pub fn with_max_value(n: usize, val_max: T) -> Self {
        let (max, stride_bits) = Self::compute_max_value_and_stride(val_max);
        Self {
            values: Vec::<U>::with_capacity(n),
            max,
            stride_bits,
            n: 0,
            masks: Self::masks(),
        }
    }

    /// Returns the number of bits for each integer in the underlying vec.
    pub fn stride_bits(&self) -> u64 {
        self.stride_bits
    }

    /// Returns the masks for filtering 1 to number of bits of U.
    fn masks() -> Vec<U> {
        let mut ret = Vec::<U>::with_capacity(U::N_BITS);
        for i in 0..U::N_BITS {
            let mask = (T::ONE << (i + 1)) - T::ONE;
            ret.push(mask.into_trunc());
        }
        ret
    }
    
    /// Returns the mask filled with ones from start_bit to end_bit.
    pub fn mask(&self, start_bit: usize, end_bit: usize) -> U {
        //let shift = end_bit - start_bit + 1;
        //let mask = (T::ONE << shift) - T::ONE;
        //mask.into_trunc() << start_bit
        if end_bit < start_bit {
            return U::ZERO;
        }
        //println!("{start_bit} {end_bit}");
        let ind = end_bit - start_bit;
        if ind >= self.masks.len() {
            return U::ZERO;
        }
        self.masks[ind] << start_bit
    }

    /// Returns the mask filled with zeroes from start_bit to end_bit, all the other bits are ones.
    pub fn mask_inverse(&self, start_bit: usize, end_bit: usize) -> U {
        !self.mask(start_bit, end_bit)
    }
}

impl<T: Integer, U: Integer> VecLike for BitVec<T, U>
where
    T: FromTruncated<U> + IntoTruncated<U>,
{
    type Type = T;

    fn with_capacity(n: usize) -> Self {
        Self {
            values: Vec::<U>::with_capacity(n),
            max: T::ONE,
            stride_bits: 1,
            n: 0,
            masks: Self::masks(),
        }
    }

    fn get(&self, i: usize) -> T {
        let (start_el, end_el, start_bit, end_bit) = self.comp_pos(i);
        // Shift the first element to the right
        let mut ret = T::from_trunc(self.values[start_el] >> start_bit);
        if start_el == end_el {
        assert!(start_bit < end_bit);
            // We just need to mask out the unnecessary bits after the last bit
            let mask = if T::SIGNED {
                // Don't include the signed bit
                //(T::ONE << (end_bit - start_bit)) - T::ONE
                self.mask(0, end_bit - start_bit - 1)
            } else {
                //(T::ONE << (end_bit - start_bit + 1)) - T::ONE
                self.mask(0, end_bit - start_bit)
            };
            //ret &= mask;
            ret &= T::from_trunc(mask);
        } else {
            // Iterate over all elements between the starting and ending element
            let n_elements_full = end_el - start_el - 1;
            for i in 0..n_elements_full {
                let ind_el = start_el + i + 1;
                let shift = (U::N_BITS - start_bit) + i * U::N_BITS;
                ret |= T::from_trunc(self.values[ind_el]) << shift;
            }
            // Mask the last element up to the ending bit
            let shift = (U::N_BITS - start_bit) + n_elements_full * U::N_BITS;
            let mask = if T::SIGNED {
                // Don't include the signed bit
                //(T::ONE << end_bit) - T::ONE
        //assert!(end_bit > 0);
                if end_bit > 0 {
                    self.mask(0, end_bit - 1)
                } else {
                    U::ZERO
                }
            } else {
                //(T::ONE << (end_bit + 1)) - T::ONE
                self.mask(0, end_bit)
            };
            //ret |= (T::from_trunc(self.values[end_el]) & mask) << shift;
            ret |= T::from_trunc(self.values[end_el] & mask) << shift;
        }
        // Add sign if necessary
        if T::SIGNED {
            let sign_bit = self.values[end_el] & (U::ONE << end_bit);
            if sign_bit > U::ZERO {
                ret = ret.negate();
            }
        }
        ret
    }
    
    fn set(&mut self, i: usize, val: T) {
        self.check_and_resize(val);
        let (start_el, end_el, start_bit, end_bit) = self.comp_pos(i);
        let val_abs = val.abs();
        if start_el == end_el {
            // Reset the bits from the previous value
            //let mask = (T::ONE << (end_bit - start_bit + 1)) - T::ONE;
            //let mask_reset = !(mask.into_trunc() << start_bit);
            let mask_reset = self.mask_inverse(start_bit, end_bit);
            self.values[start_el] &= mask_reset;
            self.values[start_el] |= val_abs.into_trunc() << start_bit;
        } else {
            // Reset previous values first
            //let mask = (T::ONE << (U::N_BITS - start_bit)) - T::ONE;
            //let mask_reset = !(mask.into_trunc() << start_bit);
            let mask_reset = self.mask_inverse(start_bit, U::N_BITS - 1);
            self.values[start_el] &= mask_reset;
            // The first entry does not have to be masked, since it will be truncated
            self.values[start_el] |= val_abs.into_trunc() << start_bit;
            // Set all elements between the starting and ending element
            let n_elements_full = end_el - start_el - 1;
            for i in 0..n_elements_full {
                let ind_el = start_el + i + 1;
                let shift = (U::N_BITS - start_bit) + i * U::N_BITS;
                self.values[ind_el] = (val_abs >> shift).into_trunc();
            }
            // Reset previous values first
            //let mask = (T::ONE << (end_bit + 1)) - T::ONE;
            //let mask_reset = !mask.into_trunc();
            let mask_reset = self.mask_inverse(0, end_bit);
            self.values[end_el] &= mask_reset;
            // Mask the last element up to the ending bit
            let shift = (U::N_BITS - start_bit) + n_elements_full * U::N_BITS;
            self.values[end_el] |= (val_abs >> shift).into_trunc();
        }
        // Set a one for a negative sign at the last bit
        if T::SIGNED {
            if val.is_neg() {
                self.values[end_el] |= U::ONE << end_bit;
            }
        }
    }
    
    fn push(&mut self, val: T) {
        let end_bit_global = (self.n as u64 + 1) * self.stride_bits - 1;
        let end_el = (end_bit_global / U::N_BITS_U64) as usize;
        self.values.resize(end_el + 1, U::ZERO);
        self.n += 1;
        let last = self.n - 1;
        self.set(last, val);
    }

    fn len(&self) -> usize {
        self.n
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
        self.max = Self::Type::ZERO;
        self.n = 0;
        self.stride_bits = 1;
    }

    fn resize(&mut self, new_len: usize, value: Self::Type) {
        self.n = new_len;
        if new_len < self.len() {
            let len_values = new_len * self.stride_bits as usize / Self::Type::N_BITS + 1;
            self.values.resize(len_values, U::ZERO);
        } else {
            for _i in self.len()..new_len {
                self.push(value);
            }
        }
    }

    fn reserve(&mut self, additional: usize) {
        let add = additional * self.stride_bits as usize / Self::Type::N_BITS + 1;
        self.values.reserve(add);
    }
}

impl<T: Integer, U: Integer> From<Vec<T>> for BitVec<T, U>
where
    T: FromTruncated<U> + IntoTruncated<U>,
{
    fn from(v: Vec<T>) -> Self {
        let val_max = v.iter().cloned().max().unwrap_or(T::ZERO);
        let mut ret = BitVec::<T, U>::with_max_value(v.len(), val_max);
        for &x in v.iter() {
            ret.push(x);
        }
        ret
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bit_vec_gen<T: Integer + From<u8>, U: Integer>(n: usize, mul: T, offset: T)
    where
        T: FromTruncated<U> + IntoTruncated<U>,
    {
        let mut bit_vec_ref = vec![T::ZERO; n];
        for i in 0..n {
            bit_vec_ref[i] = T::from((i as u8) % u8::MAX) * mul + offset;
        }
        let mut bit_vec = BitVec::<T, U>::from(bit_vec_ref.clone());
        for i in 0..n {
            assert_eq!(bit_vec.get(i), bit_vec_ref.get(i));
        }
        // Test iterator
        assert_eq!(bit_vec.iterate().sum::<T>(), bit_vec_ref.iterate().sum());
        for (v0, v1) in bit_vec.iterate().zip(bit_vec_ref.iterate()) {
            assert_eq!(v0, v1);
        }
        // Test pushing values
        let max = bit_vec_ref.iter().cloned().max().unwrap_or(T::ZERO);
        for i in 0..n {
            let val = max - bit_vec_ref.get(i);
            bit_vec.push(val);
            bit_vec_ref.push(val);
            assert_eq!(bit_vec.get(n + i), bit_vec_ref[n + i]);
        }
        // Test inserting values
        for i in (0..n).step_by(n / 16) {
            bit_vec_ref.insert(i, bit_vec_ref[i]);
            bit_vec.insert(i, bit_vec.get(i));
        }
        assert_eq!(bit_vec_ref.len(), bit_vec.len());
        for i in 0..bit_vec_ref.len() {
            assert_eq!(bit_vec_ref[i], bit_vec.get(i));
        }
        // Test removing values
        for i in (0..n).step_by(n / 8) {
            let v0 = bit_vec_ref.remove(i);
            let v1 = bit_vec.remove(i);
            assert_eq!(v0, v1);
        }
        assert_eq!(bit_vec_ref.len(), bit_vec.len());
        for i in 0..bit_vec_ref.len() {
            assert_eq!(bit_vec_ref[i], bit_vec.get(i));
        }
    }
    
    #[test]
    fn bit_vec() {
        bit_vec_gen::<u16, u8>(1_000, 1, 0);
        bit_vec_gen::<u16, u8>(1_000, 2, 0);
        bit_vec_gen::<u16, u8>(1_000, 4, 0);
        bit_vec_gen::<u16, u8>(1_000, 8, 0);
        bit_vec_gen::<u16, u8>(1_000, 16, 0);
        bit_vec_gen::<u32, u8>(1_000, 1, 0);
        bit_vec_gen::<u32, u8>(1_000, 2, 0);
        bit_vec_gen::<u32, u8>(1_000, 4, 0);
        bit_vec_gen::<u32, u8>(1_000, 8, 0);
        bit_vec_gen::<u32, u8>(1_000, 16, 0);
        bit_vec_gen::<u32, u8>(10_000, 32, 0);
        bit_vec_gen::<u32, u8>(10_000, 64, 0);
        bit_vec_gen::<u32, u8>(10_000, 128, 0);
        bit_vec_gen::<u32, u8>(100_000, 32, 0);
        bit_vec_gen::<u32, u8>(100_000, 64, 0);
        bit_vec_gen::<u32, u8>(100_000, 128, 0);
        bit_vec_gen::<u32, u8>(1_000_000, 32, 0);
        bit_vec_gen::<u32, u8>(1_000_000, 64, 0);
        bit_vec_gen::<u32, u8>(1_000_000, 128, 0);
        bit_vec_gen::<u32, u16>(1_000, 1, 0);
        bit_vec_gen::<u32, u16>(1_000, 2, 0);
        bit_vec_gen::<u32, u16>(1_000, 4, 0);
        bit_vec_gen::<u32, u16>(1_000, 8, 0);
        bit_vec_gen::<u32, u16>(1_000, 16, 0);
        bit_vec_gen::<u32, u16>(10_000, 32, 0);
        bit_vec_gen::<u32, u16>(10_000, 64, 0);
        bit_vec_gen::<u32, u16>(10_000, 128, 0);
        bit_vec_gen::<u32, u16>(100_000, 32, 0);
        bit_vec_gen::<u32, u16>(100_000, 64, 0);
        bit_vec_gen::<u32, u16>(100_000, 128, 0);
        bit_vec_gen::<u32, u16>(1_000_000, 32, 0);
        bit_vec_gen::<u32, u16>(1_000_000, 64, 0);
        bit_vec_gen::<u32, u16>(1_000_000, 128, 0);
        bit_vec_gen::<i16, u8>(1_000, 1, -500);
        bit_vec_gen::<i16, u8>(1_000, 2, -500);
        bit_vec_gen::<i16, u8>(1_000, 4, -500);
        bit_vec_gen::<i16, u8>(1_000, 8, -500);
        bit_vec_gen::<i16, u8>(1_000, 16, -500);
        bit_vec_gen::<i32, u8>(1_000, 1, -500);
        bit_vec_gen::<i32, u8>(1_000, 2, -500);
        bit_vec_gen::<i32, u8>(1_000, 4, -500);
        bit_vec_gen::<i32, u8>(1_000, 8, -500);
        bit_vec_gen::<i32, u8>(1_000, 16, -500);
        bit_vec_gen::<i32, u8>(10_000, 32, -5_000);
        bit_vec_gen::<i32, u8>(10_000, 64, -5_000);
        bit_vec_gen::<i32, u8>(10_000, 128, -5_000);
        bit_vec_gen::<i32, u8>(100_000, 32, -50_000);
        bit_vec_gen::<i32, u8>(100_000, 64, -50_000);
        bit_vec_gen::<i32, u8>(100_000, 128, -50_000);
        bit_vec_gen::<i32, u8>(1_000_000, 32, -500_000);
        bit_vec_gen::<i32, u8>(1_000_000, 64, -500_000);
        bit_vec_gen::<i32, u8>(1_000_000, 128, -500_000);
        bit_vec_gen::<i32, u16>(1_000, 1, -500);
        bit_vec_gen::<i32, u16>(1_000, 2, -500);
        bit_vec_gen::<i32, u16>(1_000, 4, -500);
        bit_vec_gen::<i32, u16>(1_000, 8, -500);
        bit_vec_gen::<i32, u16>(1_000, 16, -500);
        bit_vec_gen::<i32, u16>(10_000, 32, -5_000);
        bit_vec_gen::<i32, u16>(10_000, 64, -5_000);
        bit_vec_gen::<i32, u16>(10_000, 128, -5_000);
        bit_vec_gen::<i32, u16>(100_000, 32, -50_000);
        bit_vec_gen::<i32, u16>(100_000, 64, -50_000);
        bit_vec_gen::<i32, u16>(100_000, 128, -50_000);
        bit_vec_gen::<i32, u16>(1_000_000, 32, -500_000);
        bit_vec_gen::<i32, u16>(1_000_000, 64, -500_000);
        bit_vec_gen::<i32, u16>(1_000_000, 128, -500_000);
        bit_vec_gen::<usize, u8>(1_000, 1, 0);
        bit_vec_gen::<usize, u8>(1_000, 2, 0);
        bit_vec_gen::<usize, u8>(1_000, 4, 0);
        bit_vec_gen::<usize, u8>(1_000, 8, 0);
        bit_vec_gen::<usize, u8>(1_000, 16, 0);
        bit_vec_gen::<usize, u8>(10_000, 32, 0);
        bit_vec_gen::<usize, u8>(10_000, 64, 0);
        bit_vec_gen::<usize, u8>(10_000, 128, 0);
        bit_vec_gen::<usize, u8>(100_000, 32, 0);
        bit_vec_gen::<usize, u8>(100_000, 64, 0);
        bit_vec_gen::<usize, u8>(100_000, 128, 0);
        bit_vec_gen::<usize, u8>(1_000_000, 32, 0);
        bit_vec_gen::<usize, u8>(1_000_000, 64, 0);
        bit_vec_gen::<usize, u8>(1_000_000, 128, 0);
        bit_vec_gen::<usize, u16>(1_000, 1, 0);
        bit_vec_gen::<usize, u16>(1_000, 2, 0);
        bit_vec_gen::<usize, u16>(1_000, 4, 0);
        bit_vec_gen::<usize, u16>(1_000, 8, 0);
        bit_vec_gen::<usize, u16>(1_000, 16, 0);
        bit_vec_gen::<usize, u16>(10_000, 32, 0);
        bit_vec_gen::<usize, u16>(10_000, 64, 0);
        bit_vec_gen::<usize, u16>(10_000, 128, 0);
        bit_vec_gen::<usize, u16>(100_000, 32, 0);
        bit_vec_gen::<usize, u16>(100_000, 64, 0);
        bit_vec_gen::<usize, u16>(100_000, 128, 0);
        bit_vec_gen::<usize, u16>(1_000_000, 32, 0);
        bit_vec_gen::<usize, u16>(1_000_000, 64, 0);
        bit_vec_gen::<usize, u16>(1_000_000, 128, 0);
        bit_vec_gen::<isize, u8>(1_000, 1, -500);
        bit_vec_gen::<isize, u8>(1_000, 2, -500);
        bit_vec_gen::<isize, u8>(1_000, 4, -500);
        bit_vec_gen::<isize, u8>(1_000, 8, -500);
        bit_vec_gen::<isize, u8>(1_000, 16, -500);
        bit_vec_gen::<isize, u8>(10_000, 32, -5_000);
        bit_vec_gen::<isize, u8>(10_000, 64, -5_000);
        bit_vec_gen::<isize, u8>(10_000, 128, -5_000);
        bit_vec_gen::<isize, u8>(100_000, 32, -50_000);
        bit_vec_gen::<isize, u8>(100_000, 64, -50_000);
        bit_vec_gen::<isize, u8>(100_000, 128, -50_000);
        bit_vec_gen::<isize, u8>(1_000_000, 32, -500_000);
        bit_vec_gen::<isize, u8>(1_000_000, 64, -500_000);
        bit_vec_gen::<isize, u8>(1_000_000, 128, -500_000);
        bit_vec_gen::<isize, u16>(1_000, 1, -500);
        bit_vec_gen::<isize, u16>(1_000, 2, -500);
        bit_vec_gen::<isize, u16>(1_000, 4, -500);
        bit_vec_gen::<isize, u16>(1_000, 8, -500);
        bit_vec_gen::<isize, u16>(1_000, 16, -500);
        bit_vec_gen::<isize, u16>(10_000, 32, -5_000);
        bit_vec_gen::<isize, u16>(10_000, 64, -5_000);
        bit_vec_gen::<isize, u16>(10_000, 128, -5_000);
        bit_vec_gen::<isize, u16>(100_000, 32, -50_000);
        bit_vec_gen::<isize, u16>(100_000, 64, -50_000);
        bit_vec_gen::<isize, u16>(100_000, 128, -50_000);
        bit_vec_gen::<isize, u16>(1_000_000, 32, -500_000);
        bit_vec_gen::<isize, u16>(1_000_000, 64, -500_000);
        bit_vec_gen::<isize, u16>(1_000_000, 128, -500_000);
        bit_vec_gen::<u64, u8>(1_000, 1, 0);
        bit_vec_gen::<u64, u8>(1_000, 2, 0);
        bit_vec_gen::<u64, u8>(1_000, 4, 0);
        bit_vec_gen::<u64, u8>(1_000, 8, 0);
        bit_vec_gen::<u64, u8>(1_000, 16, 0);
        bit_vec_gen::<u64, u8>(10_000, 32, 0);
        bit_vec_gen::<u64, u8>(10_000, 64, 0);
        bit_vec_gen::<u64, u8>(10_000, 128, 0);
        bit_vec_gen::<u64, u8>(100_000, 32, 0);
        bit_vec_gen::<u64, u8>(100_000, 64, 0);
        bit_vec_gen::<u64, u8>(100_000, 128, 0);
        bit_vec_gen::<u64, u8>(1_000_000, 32, 0);
        bit_vec_gen::<u64, u8>(1_000_000, 64, 0);
        bit_vec_gen::<u64, u8>(1_000_000, 128, 0);
        bit_vec_gen::<u64, u16>(1_000, 1, 0);
        bit_vec_gen::<u64, u16>(1_000, 2, 0);
        bit_vec_gen::<u64, u16>(1_000, 4, 0);
        bit_vec_gen::<u64, u16>(1_000, 8, 0);
        bit_vec_gen::<u64, u16>(1_000, 16, 0);
        bit_vec_gen::<u64, u16>(10_000, 32, 0);
        bit_vec_gen::<u64, u16>(10_000, 64, 0);
        bit_vec_gen::<u64, u16>(10_000, 128, 0);
        bit_vec_gen::<u64, u16>(100_000, 32, 0);
        bit_vec_gen::<u64, u16>(100_000, 64, 0);
        bit_vec_gen::<u64, u16>(100_000, 128, 0);
        bit_vec_gen::<u64, u16>(1_000_000, 32, 0);
        bit_vec_gen::<u64, u16>(1_000_000, 64, 0);
        bit_vec_gen::<u64, u16>(1_000_000, 128, 0);
        bit_vec_gen::<u64, u32>(1_000, 1, 0);
        bit_vec_gen::<u64, u32>(1_000, 2, 0);
        bit_vec_gen::<u64, u32>(1_000, 4, 0);
        bit_vec_gen::<u64, u32>(1_000, 8, 0);
        bit_vec_gen::<u64, u32>(1_000, 16, 0);
        bit_vec_gen::<u64, u32>(10_000, 32, 0);
        bit_vec_gen::<u64, u32>(10_000, 64, 0);
        bit_vec_gen::<u64, u32>(10_000, 128, 0);
        bit_vec_gen::<u64, u32>(100_000, 32, 0);
        bit_vec_gen::<u64, u32>(100_000, 64, 0);
        bit_vec_gen::<u64, u32>(100_000, 128, 0);
        bit_vec_gen::<u64, u32>(1_000_000, 32, 0);
        bit_vec_gen::<u64, u32>(1_000_000, 64, 0);
        bit_vec_gen::<u64, u32>(1_000_000, 128, 0);
        bit_vec_gen::<i64, u8>(1_000, 1, 0);
        bit_vec_gen::<i64, u8>(1_000, 2, 0);
        bit_vec_gen::<i64, u8>(1_000, 4, 0);
        bit_vec_gen::<i64, u8>(1_000, 8, 0);
        bit_vec_gen::<i64, u8>(1_000, 16, 0);
        bit_vec_gen::<i64, u8>(10_000, 32, 0);
        bit_vec_gen::<i64, u8>(10_000, 64, 0);
        bit_vec_gen::<i64, u8>(10_000, 128, 0);
        bit_vec_gen::<i64, u8>(100_000, 32, 0);
        bit_vec_gen::<i64, u8>(100_000, 64, 0);
        bit_vec_gen::<i64, u8>(100_000, 128, 0);
        bit_vec_gen::<i64, u8>(1_000_000, 32, 0);
        bit_vec_gen::<i64, u8>(1_000_000, 64, 0);
        bit_vec_gen::<i64, u8>(1_000_000, 128, 0);
        bit_vec_gen::<i64, u16>(1_000, 1, 0);
        bit_vec_gen::<i64, u16>(1_000, 2, 0);
        bit_vec_gen::<i64, u16>(1_000, 4, 0);
        bit_vec_gen::<i64, u16>(1_000, 8, 0);
        bit_vec_gen::<i64, u16>(1_000, 16, 0);
        bit_vec_gen::<i64, u16>(10_000, 32, 0);
        bit_vec_gen::<i64, u16>(10_000, 64, 0);
        bit_vec_gen::<i64, u16>(10_000, 128, 0);
        bit_vec_gen::<i64, u16>(100_000, 32, 0);
        bit_vec_gen::<i64, u16>(100_000, 64, 0);
        bit_vec_gen::<i64, u16>(100_000, 128, 0);
        bit_vec_gen::<i64, u16>(1_000_000, 32, 0);
        bit_vec_gen::<i64, u16>(1_000_000, 64, 0);
        bit_vec_gen::<i64, u16>(1_000_000, 128, 0);
        bit_vec_gen::<i64, u32>(1_000, 1, 0);
        bit_vec_gen::<i64, u32>(1_000, 2, 0);
        bit_vec_gen::<i64, u32>(1_000, 4, 0);
        bit_vec_gen::<i64, u32>(1_000, 8, 0);
        bit_vec_gen::<i64, u32>(1_000, 16, 0);
        bit_vec_gen::<i64, u32>(10_000, 32, 0);
        bit_vec_gen::<i64, u32>(10_000, 64, 0);
        bit_vec_gen::<i64, u32>(10_000, 128, 0);
        bit_vec_gen::<i64, u32>(100_000, 32, 0);
        bit_vec_gen::<i64, u32>(100_000, 64, 0);
        bit_vec_gen::<i64, u32>(100_000, 128, 0);
        bit_vec_gen::<i64, u32>(1_000_000, 32, 0);
        bit_vec_gen::<i64, u32>(1_000_000, 64, 0);
        bit_vec_gen::<i64, u32>(1_000_000, 128, 0);
    }
}
