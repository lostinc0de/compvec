use crate::veclike::VecLike;

/// A vector of booleans stored bitwise.
#[derive(Clone, Debug)]
pub struct BoolVec {
    bytes: Vec<u8>,
    n_bools: usize,
}

impl BoolVec {
    /// Look up tables for masking single bits.
    const MASK_BIT: [u8; 8] = [
        1 << 0,
        1 << 1,
        1 << 2,
        1 << 3,
        1 << 4,
        1 << 5,
        1 << 6,
        1 << 7,
    ];
    const MASK_BIT_NEG: [u8; 8] = [
        !Self::MASK_BIT[0],
        !Self::MASK_BIT[1],
        !Self::MASK_BIT[2],
        !Self::MASK_BIT[3],
        !Self::MASK_BIT[4],
        !Self::MASK_BIT[5],
        !Self::MASK_BIT[6],
        !Self::MASK_BIT[7],
    ];

    /// Returns the position of the i-th boolean entry in the bytes vec
    /// and the position of the bit.
    fn pos_byte_and_bit(i: usize) -> (usize, usize) {
        let pos_byte = i / 8;
        let pos_bit = i - pos_byte * 8;
        (pos_byte, pos_bit)
    }

    fn set_byte_bit(&mut self, pos_byte: usize, pos_bit: usize, val: bool) {
        if val {
            self.bytes[pos_byte] |= Self::MASK_BIT[pos_bit];
        } else {
            self.bytes[pos_byte] &= Self::MASK_BIT_NEG[pos_bit];
        }
    }
}

impl VecLike for BoolVec {
    type Type = bool;

    fn with_capacity(c: usize) -> Self {
        Self {
            bytes: Vec::<u8>::with_capacity(c / 8 + 1),
            n_bools: 0,
        }
    }

    fn get(&self, i: usize) -> bool {
        let (pos_byte, pos_bit) = Self::pos_byte_and_bit(i);
        let ret = self.bytes[pos_byte] & Self::MASK_BIT[pos_bit];
        ret > 0
    }

    fn set(&mut self, i: usize, val: bool) {
        let (pos_byte, pos_bit) = Self::pos_byte_and_bit(i);
        self.set_byte_bit(pos_byte, pos_bit, val);
    }

    fn len(&self) -> usize {
        self.n_bools
    }

    fn push(&mut self, val: bool) {
        self.n_bools += 1;
        let (pos_byte, pos_bit) = Self::pos_byte_and_bit(self.n_bools - 1);
        if pos_byte >= self.bytes.len() {
            let byte_new = if val { 1u8 } else { 0u8 };
            self.bytes.push(byte_new);
        } else {
            self.set_byte_bit(pos_byte, pos_bit, val);
        }
    }

    fn pop(&mut self) -> Option<bool> {
        if self.n_bools > 0 {
            let (pos_byte, pos_bit) = Self::pos_byte_and_bit(self.n_bools - 1);
            self.n_bools -= 1;
            if pos_bit == 0 {
                let ret = self.bytes[pos_byte] & 1;
                let len_new = self.bytes.len() - 1;
                self.bytes.resize(len_new, 0u8);
                return Some(ret > 0);
            } else {
                let ret = self.bytes[pos_byte] & (1 << pos_bit);
                return Some(ret > 0);
            }
        }
        None
    }

    fn iter_values(&self) -> impl Iterator<Item = bool> {
        (0..self.n_bools).map(|i| self.get(i))
    }

    fn clear(&mut self) {
        self.bytes.clear();
        self.n_bools = 0;
    }

    fn resize(&mut self, new_len: usize, value: bool) {
        if new_len < self.len() {
            let len_bytes = new_len / 8 + 1;
            self.bytes.resize(len_bytes, 0u8);
            self.n_bools = new_len;
        } else {
            for _i in self.len()..new_len {
                self.push(value);
            }
        }
    }

    fn reserve(&mut self, additional: usize) {
        let add = additional / 8 + 1;
        self.bytes.reserve(add);
    }
}

impl From<Vec<bool>> for BoolVec {
    fn from(v: Vec<bool>) -> Self {
        let mut bytes = vec![];
        for c in v.chunks(8) {
            let mut byte = 0u8;
            for (ind, &b) in c.iter().enumerate() {
                if b {
                    byte |= 1 << ind;
                }
            }
            bytes.push(byte);
        }
        let n_bools = v.len();
        Self { bytes, n_bools }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bool_vec() {
        const N: usize = 1000;
        let mut vec_ref = vec![false; N];
        let mut bool_vec = BoolVec::from(vec_ref.clone());
        assert_eq!(vec_ref.len(), bool_vec.len());
        assert_eq!(vec_ref.len(), N);
        for i in 0..N {
            assert_eq!(vec_ref[i], bool_vec.get(i));
            assert_eq!(bool_vec.get(i), false);
        }
        for i in 0..N {
            if i % 2 == 0 {
                vec_ref[i] = true;
                bool_vec.set(i, true);
            }
        }
        for i in 0..N {
            assert_eq!(vec_ref[i], bool_vec.get(i));
            vec_ref.push(i % 3 == 0);
            bool_vec.push(i % 3 == 0);
        }
        for i in 0..(2 * N) {
            assert_eq!(vec_ref[i], bool_vec.get(i));
        }
        // Test inserting values
        for i in (0..N).step_by(N / 16) {
            vec_ref.insert(i, vec_ref[i]);
            bool_vec.insert(i, bool_vec.get(i));
        }
        assert_eq!(vec_ref.len(), bool_vec.len());
        for i in 0..vec_ref.len() {
            assert_eq!(vec_ref[i], bool_vec.get(i));
        }
        // Test removing values
        for i in (0..N).step_by(N / 16) {
            let v0 = vec_ref.remove(i);
            let v1 = bool_vec.remove(i);
            assert_eq!(v0, v1);
        }
        assert_eq!(vec_ref.len(), bool_vec.len());
        for i in 0..vec_ref.len() {
            assert_eq!(vec_ref[i], bool_vec.get(i));
        }
        // Test extending from a slice
        let slc = &[vec_ref[0], vec_ref[1], vec_ref[N - 1]];
        vec_ref.extend_from_slice(slc);
        bool_vec.extend_from_slice(slc);
        assert_eq!(vec_ref.len(), bool_vec.len());
        let offset = vec_ref.len() - slc.len();
        for i in 0..slc.len() {
            assert_eq!(vec_ref.get(offset + i), slc[i]);
            assert_eq!(bool_vec.get(offset + i), slc[i]);
        }
        // Test splitting off
        let at = vec_ref.len() - slc.len();
        let splt_ref = vec_ref.split_off(at);
        let splt_bool_vec = bool_vec.split_off(at);
        assert_eq!(splt_ref.len(), splt_bool_vec.len());
        assert_eq!(vec_ref.len(), bool_vec.len());
        for i in 0..slc.len() {
            assert_eq!(splt_ref.get(i), slc[i]);
            assert_eq!(splt_bool_vec.get(i), slc[i]);
        }
        // Test removing consecutive duplicate elements
        let slc = &[vec_ref[0], vec_ref[0], vec_ref[0], vec_ref[1]];
        vec_ref.extend_from_slice(slc);
        bool_vec.extend_from_slice(slc);
        vec_ref.dedup();
        bool_vec.dedup();
        assert_eq!(vec_ref.len(), bool_vec.len());
        assert_eq!(vec_ref[vec_ref.len() - 3], bool_vec.get(bool_vec.len() - 3));
        assert_eq!(vec_ref[vec_ref.len() - 2], bool_vec.get(bool_vec.len() - 2));
        assert_eq!(vec_ref[vec_ref.len() - 1], bool_vec.get(bool_vec.len() - 1));
        // Test swap remove
        vec_ref.swap_remove(N / 2);
        bool_vec.swap_remove(N / 2);
        vec_ref.swap_remove(N / 4);
        bool_vec.swap_remove(N / 4);
        vec_ref.swap_remove(N / 6 + 1);
        bool_vec.swap_remove(N / 6 + 1);
        assert_eq!(vec_ref.len(), bool_vec.len());
        for i in 0..vec_ref.len() {
            assert_eq!(vec_ref[i], bool_vec.get(i));
        }
        // Pop until the containers are empty
        while !vec_ref.is_empty() {
            assert_eq!(vec_ref.pop(), bool_vec.pop());
            assert_eq!(vec_ref.len(), bool_vec.len());
        }
    }
}
