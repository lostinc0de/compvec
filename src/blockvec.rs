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
        let blocks = vec![V::with_capacity(len_block); n_blocks];
        Self { n: 0, len_block, blocks }
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
        let ind_last_block = self.blocks.len() - 1;
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

    fn iterate(&self) -> impl Iterator<Item = Self::Type> {
        self.blocks.iter().map(|v| v.iterate()).flatten()
    }

    fn clear(&mut self) {
        self.blocks.clear();
        self.n = 0;
    }

    fn resize(&mut self, new_len: usize, value: Self::Type) {
        let n_blocks_new = new_len / self.len_block;
        let res = new_len - n_blocks_new * self.len_block;
        if res > 0 {
            self.blocks.resize(n_blocks_new + 1, V::from(vec![value; self.len_block]));
            let ind_last_block = self.blocks.len() - 1;
            self.blocks[ind_last_block].resize(res, value);
        } else {
            self.blocks.resize(n_blocks_new, V::from(vec![value; self.len_block]));
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
}
