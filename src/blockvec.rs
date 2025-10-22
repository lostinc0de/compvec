use crate::veclike::VecLike;

/// Holds a Vec of Veclike structures, where each element has a fixed block length.
#[derive(Clone, Debug)]
pub struct BlockVec<V: VecLike> {
    n: usize,
    len_block: usize,
    blocks: Vec<V>,
}

impl<V: VecLike> BlockVec<V> {
    /// Computes the index of the block and the local index from the global index.
    fn block_id_and_local_id(&self, ind: usize) -> (usize, usize) {
        let block_id = ind / self.len_block;
        let loc_id = ind - block_id * self.len_block;
        (block_id, loc_id)
    }

    /// Constructs an empty BlockVec with specified block length and reserves space for c values.
    fn with_block_len(c: usize, len_block: usize) -> Self {
        let n_blocks = c / len_block + 1;
        let blocks = Vec::<V>::with_capacity(n_blocks);
        Self {
            n: 0,
            len_block,
            blocks,
        }
    }
}

impl<V: VecLike> From<Vec<V::Type>> for BlockVec<V> {
    fn from(v: Vec<V::Type>) -> Self {
        let mut ret = BlockVec::<V>::with_block_len(v.len(), 1024);
        for &x in v.iter() {
            ret.push(x);
        }
        ret
    }
}

impl<V: VecLike> VecLike for BlockVec<V> {
    type Type = V::Type;

    fn with_capacity(c: usize) -> Self {
        Self::with_block_len(c, 64)
    }

    fn get(&self, i: usize) -> Self::Type {
        let (block_id, loc_id) = self.block_id_and_local_id(i);
        self.blocks[block_id].get(loc_id)
    }

    fn set(&mut self, i: usize, val: Self::Type) {
        let (block_id, loc_id) = self.block_id_and_local_id(i);
        self.blocks[block_id].set(loc_id, val);
    }

    fn len(&self) -> usize {
        self.n
    }

    fn push(&mut self, val: Self::Type) {
        if self.blocks.len() == 0 {
            self.blocks.push(V::with_capacity(self.len_block));
        }
        let ind_last_block = {
            // Check if the last block has enough capacity
            let ind_last_bl = self.blocks.len() - 1;
            if self.blocks[ind_last_bl].len() == self.len_block {
                self.blocks.push(V::with_capacity(self.len_block));
            }
            self.blocks.len() - 1
        };
        self.blocks[ind_last_block].push(val);
        self.n += 1;
    }

    fn pop(&mut self) -> Option<Self::Type> {
        if self.len() > 0 {
            let (block_id, loc_id) = self.block_id_and_local_id(self.len() - 1);
            let ret = self.blocks[block_id].pop();
            if loc_id == 0 {
                self.blocks.resize(self.blocks.len() - 1, V::new());
            }
            self.n -= 1;
            return ret;
        }
        None
    }

    fn iter_values(&self) -> impl Iterator<Item = Self::Type> {
        self.blocks.iter().map(|v| v.iter_values()).flatten()
    }

    fn clear(&mut self) {
        self.blocks.clear();
        self.n = 0;
    }

    fn resize(&mut self, new_len: usize, value: Self::Type) {
        let n_blocks_new = new_len / self.len_block;
        let res = new_len - n_blocks_new * self.len_block;
        if res > 0 {
            // We need an additional block for the remaining values
            self.blocks
                .resize(n_blocks_new + 1, V::from(vec![value; self.len_block]));
            let ind_last_block = self.blocks.len() - 1;
            self.blocks[ind_last_block].resize(res, value);
        } else {
            // All values fit into the blocks
            self.blocks
                .resize(n_blocks_new, V::from(vec![value; self.len_block]));
        }
        self.n = new_len;
    }

    fn reserve(&mut self, additional: usize) {
        if additional > 0 {
            let n_blocks_add = additional / self.len_block;
            // Reserve additional blocks needed
            self.blocks.reserve(n_blocks_add);
            // Check if the remaining reserved values can be pushed to the last block
            let res = additional - n_blocks_add * self.len_block;
            let ind_last_block = self.blocks.len() - 1;
            if res < (self.len_block - self.blocks[ind_last_block].len()) {
                self.blocks[ind_last_block].reserve(res);
            } else {
                // We need another block
                self.blocks.reserve(1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitvec::BitVec;
    use crate::bytevec::ByteVec;
    use crate::types::Integer;

    fn block_vec_gen<V: VecLike>(n: usize, len_block: usize, mul: V::Type, offset: V::Type)
    where
        V::Type: Integer + From<u8>,
    {
        let mut block_vec = BlockVec::<V>::with_block_len(n, len_block);
        let mut vec_ref = vec![];
        for i in 0..n {
            let val = V::Type::from((i % u8::MAX as usize) as u8) * mul + offset;
            block_vec.push(val);
            vec_ref.push(val);
        }
        assert_eq!(vec_ref.len(), block_vec.len());
        for i in 0..n {
            assert_eq!(vec_ref.get(i), block_vec.get(i));
        }
        block_vec.clear();
        block_vec = BlockVec::<V>::from(vec_ref.clone());
        assert_eq!(vec_ref.len(), block_vec.len());
        for i in 0..n {
            assert_eq!(vec_ref.get(i), block_vec.get(i));
        }
        // Test extending from a slice
        let slc = &[vec_ref[0], vec_ref[1], vec_ref[n - 1]];
        vec_ref.extend_from_slice(slc);
        block_vec.extend_from_slice(slc);
        let offset = vec_ref.len() - slc.len();
        for i in 0..slc.len() {
            assert_eq!(vec_ref.get(offset + i), slc[i]);
            assert_eq!(block_vec.get(offset + i), slc[i]);
        }
        // Test splitting off
        let at = vec_ref.len() - slc.len();
        let splt_ref = vec_ref.split_off(at);
        let splt_byte_vec = block_vec.split_off(at);
        assert_eq!(splt_ref.len(), splt_byte_vec.len());
        assert_eq!(vec_ref.len(), block_vec.len());
        for i in 0..slc.len() {
            assert_eq!(splt_ref.get(i), slc[i]);
            assert_eq!(splt_byte_vec.get(i), slc[i]);
        }
        // Test removing consecutive duplicate elements
        let slc = &[vec_ref[0], vec_ref[0], vec_ref[0], vec_ref[1]];
        vec_ref.extend_from_slice(slc);
        block_vec.extend_from_slice(slc);
        vec_ref.dedup();
        block_vec.dedup();
        assert_eq!(vec_ref.len(), block_vec.len());
        assert_eq!(vec_ref[vec_ref.len() - 3], block_vec.get(block_vec.len() - 3));
        assert_eq!(vec_ref[vec_ref.len() - 2], block_vec.get(block_vec.len() - 2));
        assert_eq!(vec_ref[vec_ref.len() - 1], block_vec.get(block_vec.len() - 1));
    }

    #[test]
    fn block_vec_byte_vec_unsigned() {
        block_vec_gen::<ByteVec<u16, u8>>(1_000, 64, 1, 0);
        block_vec_gen::<ByteVec<u16, u8>>(1_000, 200, 1, 400);
        block_vec_gen::<ByteVec<u16, u8>>(10_000, 200, 1, 400);
        block_vec_gen::<ByteVec<u32, u8>>(1_000, 64, 1, 0);
        block_vec_gen::<ByteVec<u32, u8>>(1_000, 128, 12, 800);
        block_vec_gen::<ByteVec<u32, u8>>(10_000, 200, 2_400, 80_000);
        block_vec_gen::<ByteVec<u32, u8>>(100_000, 4_000, 4_800, 800_000);
        block_vec_gen::<ByteVec<u32, u16>>(1_000, 64, 1, 0);
        block_vec_gen::<ByteVec<u32, u16>>(1_000, 128, 12, 800);
        block_vec_gen::<ByteVec<u32, u16>>(10_000, 200, 2_400, 80_000);
        block_vec_gen::<ByteVec<u32, u16>>(100_000, 4_000, 4_800, 800_000);
        block_vec_gen::<ByteVec<u64, u8>>(1_000, 64, 1, 0);
        block_vec_gen::<ByteVec<u64, u8>>(1_000, 128, 12, 800);
        block_vec_gen::<ByteVec<u64, u8>>(10_000, 200, 2_400, 80_000);
        block_vec_gen::<ByteVec<u64, u8>>(100_000, 4_000, 4_800, 800_000);
        block_vec_gen::<ByteVec<u64, u8>>(100_000, 8_000, 9_600, 8_000_000);
        block_vec_gen::<ByteVec<u64, u16>>(1_000, 64, 1, 0);
        block_vec_gen::<ByteVec<u64, u16>>(1_000, 128, 12, 800);
        block_vec_gen::<ByteVec<u64, u16>>(10_000, 200, 2_400, 80_000);
        block_vec_gen::<ByteVec<u64, u16>>(100_000, 4_000, 4_800, 800_000);
        block_vec_gen::<ByteVec<u64, u16>>(100_000, 8_000, 9_600, 8_000_000);
        block_vec_gen::<ByteVec<u64, u32>>(1_000, 64, 1, 0);
        block_vec_gen::<ByteVec<u64, u32>>(1_000, 128, 12, 800);
        block_vec_gen::<ByteVec<u64, u32>>(10_000, 200, 2_400, 80_000);
        block_vec_gen::<ByteVec<u64, u32>>(100_000, 4_000, 4_800, 800_000);
        block_vec_gen::<ByteVec<u64, u32>>(100_000, 8_000, 9_600, 8_000_000);
    }

    #[test]
    fn block_vec_byte_vec_signed() {
        block_vec_gen::<ByteVec<i16, u8>>(1_000, 64, 1, 0);
        block_vec_gen::<ByteVec<i16, u8>>(1_000, 200, 1, -400);
        block_vec_gen::<ByteVec<i16, u8>>(10_000, 200, 1, -400);
        block_vec_gen::<ByteVec<i32, u8>>(1_000, 64, 1, 0);
        block_vec_gen::<ByteVec<i32, u8>>(1_000, 128, 12, -800);
        block_vec_gen::<ByteVec<i32, u8>>(10_000, 200, 2_400, -80_000);
        block_vec_gen::<ByteVec<i32, u8>>(100_000, 4_000, 4_800, -800_000);
        block_vec_gen::<ByteVec<i32, u16>>(1_000, 64, 1, 0);
        block_vec_gen::<ByteVec<i32, u16>>(1_000, 128, 12, -800);
        block_vec_gen::<ByteVec<i32, u16>>(10_000, 200, 2_400, -80_000);
        block_vec_gen::<ByteVec<i32, u16>>(100_000, 4_000, 4_800, -800_000);
        block_vec_gen::<ByteVec<i64, u8>>(1_000, 64, 1, 0);
        block_vec_gen::<ByteVec<i64, u8>>(1_000, 128, 12, -800);
        block_vec_gen::<ByteVec<i64, u8>>(10_000, 200, 2_400, -80_000);
        block_vec_gen::<ByteVec<i64, u8>>(100_000, 4_000, 4_800, -800_000);
        block_vec_gen::<ByteVec<i64, u8>>(100_000, 8_000, 9_600, -8_000_000);
        block_vec_gen::<ByteVec<i64, u16>>(1_000, 64, 1, 0);
        block_vec_gen::<ByteVec<i64, u16>>(1_000, 128, 12, -800);
        block_vec_gen::<ByteVec<i64, u16>>(10_000, 200, 2_400, -80_000);
        block_vec_gen::<ByteVec<i64, u16>>(100_000, 4_000, 4_800, -800_000);
        block_vec_gen::<ByteVec<i64, u16>>(100_000, 8_000, 9_600, -8_000_000);
        block_vec_gen::<ByteVec<i64, u32>>(1_000, 64, 1, 0);
        block_vec_gen::<ByteVec<i64, u32>>(1_000, 128, 12, -800);
        block_vec_gen::<ByteVec<i64, u32>>(10_000, 200, 2_400, -80_000);
        block_vec_gen::<ByteVec<i64, u32>>(100_000, 4_000, 4_800, -800_000);
        block_vec_gen::<ByteVec<i64, u32>>(100_000, 8_000, 9_600, -8_000_000);
    }

    #[test]
    fn block_vec_bit_vec_unsigned() {
        block_vec_gen::<BitVec<u16, u8>>(1_000, 64, 1, 0);
        block_vec_gen::<BitVec<u16, u8>>(1_000, 200, 1, 400);
        block_vec_gen::<BitVec<u16, u8>>(10_000, 200, 1, 400);
        block_vec_gen::<BitVec<u32, u8>>(1_000, 64, 1, 0);
        block_vec_gen::<BitVec<u32, u8>>(1_000, 128, 12, 800);
        block_vec_gen::<BitVec<u32, u8>>(10_000, 200, 2_400, 80_000);
        block_vec_gen::<BitVec<u32, u8>>(100_000, 4_000, 4_800, 800_000);
        block_vec_gen::<BitVec<u32, u16>>(1_000, 64, 1, 0);
        block_vec_gen::<BitVec<u32, u16>>(1_000, 128, 12, 800);
        block_vec_gen::<BitVec<u32, u16>>(10_000, 200, 2_400, 80_000);
        block_vec_gen::<BitVec<u32, u16>>(100_000, 4_000, 4_800, 800_000);
        block_vec_gen::<BitVec<u64, u8>>(1_000, 64, 1, 0);
        block_vec_gen::<BitVec<u64, u8>>(1_000, 128, 12, 800);
        block_vec_gen::<BitVec<u64, u8>>(10_000, 200, 2_400, 80_000);
        block_vec_gen::<BitVec<u64, u8>>(100_000, 4_000, 4_800, 800_000);
        block_vec_gen::<BitVec<u64, u8>>(100_000, 8_000, 9_600, 8_000_000);
        block_vec_gen::<BitVec<u64, u16>>(1_000, 64, 1, 0);
        block_vec_gen::<BitVec<u64, u16>>(1_000, 128, 12, 800);
        block_vec_gen::<BitVec<u64, u16>>(10_000, 200, 2_400, 80_000);
        block_vec_gen::<BitVec<u64, u16>>(100_000, 4_000, 4_800, 800_000);
        block_vec_gen::<BitVec<u64, u16>>(100_000, 8_000, 9_600, 8_000_000);
        block_vec_gen::<BitVec<u64, u32>>(1_000, 64, 1, 0);
        block_vec_gen::<BitVec<u64, u32>>(1_000, 128, 12, 800);
        block_vec_gen::<BitVec<u64, u32>>(10_000, 200, 2_400, 80_000);
        block_vec_gen::<BitVec<u64, u32>>(100_000, 4_000, 4_800, 800_000);
        block_vec_gen::<BitVec<u64, u32>>(100_000, 8_000, 9_600, 8_000_000);
    }

    #[test]
    fn block_vec_bit_vec_signed() {
        block_vec_gen::<BitVec<i16, u8>>(1_000, 64, 1, 0);
        block_vec_gen::<BitVec<i16, u8>>(1_000, 200, 1, -400);
        block_vec_gen::<BitVec<i16, u8>>(10_000, 200, 1, -400);
        block_vec_gen::<BitVec<i32, u8>>(1_000, 64, 1, 0);
        block_vec_gen::<BitVec<i32, u8>>(1_000, 128, 12, -800);
        block_vec_gen::<BitVec<i32, u8>>(10_000, 200, 2_400, -80_000);
        block_vec_gen::<BitVec<i32, u8>>(100_000, 4_000, 4_800, -800_000);
        block_vec_gen::<BitVec<i32, u16>>(1_000, 64, 1, 0);
        block_vec_gen::<BitVec<i32, u16>>(1_000, 128, 12, -800);
        block_vec_gen::<BitVec<i32, u16>>(10_000, 200, 2_400, -80_000);
        block_vec_gen::<BitVec<i32, u16>>(100_000, 4_000, 4_800, -800_000);
        block_vec_gen::<BitVec<i64, u8>>(1_000, 64, 1, 0);
        block_vec_gen::<BitVec<i64, u8>>(1_000, 128, 12, -800);
        block_vec_gen::<BitVec<i64, u8>>(10_000, 200, 2_400, -80_000);
        block_vec_gen::<BitVec<i64, u8>>(100_000, 4_000, 4_800, -800_000);
        block_vec_gen::<BitVec<i64, u8>>(100_000, 8_000, 9_600, -8_000_000);
        block_vec_gen::<BitVec<i64, u16>>(1_000, 64, 1, 0);
        block_vec_gen::<BitVec<i64, u16>>(1_000, 128, 12, -800);
        block_vec_gen::<BitVec<i64, u16>>(10_000, 200, 2_400, -80_000);
        block_vec_gen::<BitVec<i64, u16>>(100_000, 4_000, 4_800, -800_000);
        block_vec_gen::<BitVec<i64, u16>>(100_000, 8_000, 9_600, -8_000_000);
        block_vec_gen::<BitVec<i64, u32>>(1_000, 64, 1, 0);
        block_vec_gen::<BitVec<i64, u32>>(1_000, 128, 12, -800);
        block_vec_gen::<BitVec<i64, u32>>(10_000, 200, 2_400, -80_000);
        block_vec_gen::<BitVec<i64, u32>>(100_000, 4_000, 4_800, -800_000);
        block_vec_gen::<BitVec<i64, u32>>(100_000, 8_000, 9_600, -8_000_000);
    }
}
