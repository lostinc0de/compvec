use std::fmt::Display;

/// Defines functionality for an object, which behaves almost like a Vec.
pub trait VecLike: Sized + From<Vec<Self::Type>> {
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

    /// Appends an entry.
    fn push(&mut self, val: Self::Type);

    /// Returns the last entry and deletes it.
    fn pop(&mut self) -> Option<Self::Type>;

    /// Checks, if the container is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

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
            self.push(element);
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

    fn insert(&mut self, index: usize, element: Self::Type) {
        self.insert(index, element);
    }

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
