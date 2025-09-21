use std::fmt::Display;

/// Defines functionality for an object, which behaves almost like a Vec.
pub trait VecLike: Sized + From<Vec<Self::Type>> + Clone {
    type Type: Copy + Default + Display;

    /// Creates a vector like object with capacity.
    fn with_capacity(c: usize) -> Self;

    /// Creates an empty vector like object.
    fn new() -> Self {
        Self::with_capacity(0)
    }

    /// Returns entry at position i as value.
    fn get(&self, i: usize) -> Self::Type;

    /// Sets entry at position i with value val.
    fn set(&mut self, i: usize, val: Self::Type);

    /// Returns the number of stored objects.
    fn len(&self) -> usize;
    
    /// Checks, if the container is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Appends an entry.
    fn push(&mut self, val: Self::Type);

    /// Returns the last entry and deletes it.
    fn pop(&mut self) -> Option<Self::Type>;

    /// Returns an iterator over all values.
    fn iterate(&self) -> impl Iterator<Item = Self::Type>;

    /// Moves all elements from another Vec-like structure.
    fn append(&mut self, other: &mut Self) {
        for v in other.iterate() {
            self.push(v);
        }
        other.clear();
    }

    /// Removes all values, but capacity may be unaffected.
    fn clear(&mut self);

    /// Inserts an element at index.
    /// This operation may be inefficient, since the vector needs to be restructured.
    fn insert(&mut self, index: usize, element: Self::Type) {
        let len_old = self.len();
        if index > len_old {
            panic!("insertion index (is {index}) should be <= len (is {len_old})");
        }
        if len_old == 1 {
            if index == 1 {
                self.push(element);
            } else {
                let last = self.get(0);
                self.push(last);
                self.set(index, element);
            }
        } else {
            // Copy the last value and append it
            let last = self.get(len_old - 1);
            self.push(last);
            // Store the previous value at index
            let mut prev = self.get(index);
            self.set(index, element);
            // Repeat for all remaining values
            for i in (index + 1)..self.len() {
                let tmp = prev;
                prev = self.get(i);
                self.set(i, tmp);
            }
        }
    }

    /// Removes an element at index.
    /// This operation may be inefficient, since the vector needs to be restructured.
    fn remove(&mut self, index: usize) -> Self::Type {
        let len = self.len();
        if index >= len {
            panic!("removal index (is {index}) should be < len (is {len})");
        }
        // Store the deleted value
        let ret = self.get(index);
        // Move all values beyond the index
        for i in index..(len - 1) {
            let next = self.get(i + 1);
            self.set(i, next);
        }
        // Remove the last one by resizing
        self.resize(len - 1, Self::Type::default());
        ret
    }

    /// Resizes the vector to a given length.
    fn resize(&mut self, new_len: usize, value: Self::Type);
    
    /// Truncates the vector to len.
    fn truncate(&mut self, len: usize) {
        if len < self.len() {
            self.resize(len, Self::Type::default());
        }
    }
    
    /// Reserves additional space for push values.
    fn reserve(&mut self, additional: usize);
}

/// Implement VecLike for Vec in the std-lib.
impl<T: Copy + Default + Display> VecLike for Vec<T> {
    type Type = T;

    fn with_capacity(c: usize) -> Self {
        Vec::<T>::with_capacity(c)
    }

    fn get(&self, i: usize) -> T {
        self[i]
    }

    fn set(&mut self, i: usize, val: T) {
        self[i] = val;
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn push(&mut self, val: T) {
        self.push(val);
    }

    fn pop(&mut self) -> Option<T> {
        self.pop()
    }

    fn iterate(&self) -> impl Iterator<Item = T> {
        self.iter().cloned()
    }

    // Overwrite implementation of VecLike.
    fn append(&mut self, other: &mut Self) {
        self.append(other);
    }

    fn clear(&mut self) {
        self.clear();
    }

    // Overwrite implementation of VecLike.
    fn insert(&mut self, index: usize, element: Self::Type) {
        self.insert(index, element);
    }

    // Overwrite implementation of VecLike.
    fn remove(&mut self, index: usize) -> Self::Type {
        self.remove(index)
    }

    fn resize(&mut self, new_len: usize, value: Self::Type) {
        self.resize(new_len, value);
    }

    fn reserve(&mut self, additional: usize) {
        self.reserve(additional);
    }
}

// TODO
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

    fn with_block_len(c: usize, len_block: usize) -> Self {
        let n_blocks = c / len_block + 1;
        let blocks = vec![V::with_capacity(len_block); n_blocks];
        Self { n: 0, len_block, blocks }
    }
}

/*
impl<T: Copy + Default + Display> VecLike for BlockVec<T> {
    type Type = V::Type;

    fn with_capacity(c: usize) -> Self {
        Self::with_block_len(c, 64)
    }

    fn get(&self, i: usize) -> T {
        let (block_id, loc_id) = self.block_id_and_local_id(i);
        self.blocks[block_id].get(loc_id);
    }

    fn set(&mut self, i: usize, val: T) {
        let (block_id, loc_id) = self.block_id_and_local_id(i);
        self.blocks[block_id].set(loc_id, val);
    }

    fn len(&self) -> usize {
        self.n
    }

    fn push(&mut self, val: T) {
        if self.blocks.len() == 0 {
            self.blocks.push(V::with_capacity(self.len_block));
        } else {
            let last_block_id = self.blocks.len() - 1;
        }
        self.n += 1;
    }

    fn pop(&mut self) -> Option<T> {
        if self.len() > 0 {
            let (block_id, loc_id) = self.block_id_and_local_id(self.len() - 1);
            self.n -= 1;
        }
    }

    fn iterate(&self) -> impl Iterator<Item = T> {
    }

    fn clear(&mut self) {
        self.blocks.clear();
        self.n = 0;
    }

    fn resize(&mut self, new_len: usize, value: Self::Type) {
        self.n = new_len;
    }

    fn reserve(&mut self, additional: usize) {
    }
}
*/
