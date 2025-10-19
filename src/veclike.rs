use std::cmp::PartialEq;
use std::fmt::Display;

/// Defines functionality for an object, which behaves almost like a Vec.
pub trait VecLike: Sized + From<Vec<Self::Type>> + Clone {
    type Type: Copy + Default + Display + PartialEq;

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

    /// Appends all elements from the slice.
    fn extend_from_slice(&mut self, other: &[Self::Type]) {
        for &v in other.iter() {
            self.push(v);
        }
    }

    /// Removes consecutive duplicates.
    fn dedup(&mut self) {
        let len = self.len();
        if len > 0 {
            let mut v = Self::with_capacity(len / 2);
            let mut prev = self.get(0);
            for i in 1..len {
                let val = self.get(i);
                if val != prev {
                    v.push(val);
                    prev = val;
                }
            }
            *self = v;
        }
    }

    /// Swaps the last element with index and resizes vec.
    /// Returns the removed element at index.
    fn swap_remove(&mut self, index: usize) -> Self::Type {
        let len = self.len();
        if index >= len {
            panic!("swap_remove index (is {index}) should be < len (is {len})");
        }
        let last = self.get(len - 1);
        let ret = self.get(index);
        self.set(index, last);
        self.resize(len - 1, Self::Type::default());
        return ret;
    }

    /// Splits the collection into two at the given index.
    fn split_off(&mut self, at: usize) -> Self {
        let len = self.len();
        if at >= len {
            panic!("`at` split index (is {at}) should be <= len (is {len})");
        }
        let mut ret = Self::with_capacity(len - at);
        for i in at..len {
            let val = self.get(i);
            ret.push(val);
        }
        self.resize(at, Self::Type::default());
        ret
    }
}

/// Implement VecLike for Vec in the std-lib.
impl<T: Copy + Default + Display + PartialEq> VecLike for Vec<T> {
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

    // Overwrite implementation of VecLike.
    fn extend_from_slice(&mut self, other: &[Self::Type]) {
        self.extend_from_slice(other);
    }

    // Overwrite implementation of VecLike.
    //fn dedup(&mut self) {
    //    self.dedup();
    //}

    // Overwrite implementation of VecLike.
    fn swap_remove(&mut self, index: usize) -> Self::Type {
        self.swap_remove(index)
    }
}
