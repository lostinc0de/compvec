# compvec - Compressing integer vectors in Rust
## Introduction
This library tries to solve a simple problem: Storing integers in a vector / Vec in a space efficient and still performant way. If you want to store the indices of an array, you usually don't need more than 32 bit for each index. Still a usize wastes eight bytes on a 64 bit system.
One could store the indices in a Vec<u32>, but this will only work, if you can be sure, the maximum value won't exceed 32 bits. compvec solves this issue as well and you don't have to worry about pushing a larger value. The only problem is, all the integers in the sequence have to be realigned.

## Details
With compvec you can store integer values in one or more smaller integers. There are two structs to achieve this: ByteVec and BitVec. As the name intends, integers are compressed either on a byte-level or bit-level.
For example a ByteVec<u32, u8> stores 32 bit values in one, two, three or four bytes, depending on the maximum value in the collection. If the maximum number is 65000, two bytes per entry is enough (16 bit). If you know this beforehand, consider using a ByteVec<u32, u16> instead, since performance should be better with one u16.

## Examples
### ByteVec
ByteVec encodes integers in sub-elements (not necessarily bytes). 

```rust
// Two bytes are needed to represent these 32 bit values
let mut bytevec = ByteVec::<u32, u8>::from(vec![32, 64, 999, 65_000]);
bytevec.push(1_000);
// This will increase the number of bytes per value by one
bytevec.push(100_000);
// Prints "ByteVec { values: [32, 0, 0, 64, 0, 0, 231, 3, 0, 232, 253, 0, 232, 3, 0, 160, 134, 1], max: 16581375, stride: 3, n: 6 }"
println!("{:?}", bytevec);
```

So let's have a look at the third value 999: It's encoded in the three bytes 231, 3 and 0.

|            | 231   | 3     | 0     |
| -----      | ----- | ----- | ----- |
| Multiplier | 1     | 256   | 65536 |
| Index      | 2     | 2     | 2     |

If we decode it using the get() function, these three bytes are multiplied and summed up: 231 + 3 * 2^8 + 0 * 2^16 = 231 + 3 * 256 + 0 * 65536 = 999.

### BitVec
For storing values in a more space-efficient way, we can use the BitVec. Every value will only use the number of bits of the maximum value in the set. If you don't have values larger than 1,000, every integer will consume 10 bits of memory.
If you push a larger value to the set, the whole structure needs to be realigned. At best you know the largest value in the Vec beforehand.You can use the function with_max_value() for this purpose.

```rust
// Values will be stored in 10 bits and the maximum value is 1,023
let mut bitvec = BitVec::<usize, u8>::with_max_value(4, 1_000);
bitvec.push(100);
bitvec.push(200);
bitvec.push(400);
// This will increase the number of bits per value to 11 bits
bitvec.push(2_000);
// Prints "BitVec { values: [100, 64, 6, 100, 160, 15, 0], max: 2047, stride_bits: 11, n: 4, ..."
println!("{:?}", bitvec);
```

The first value 100 is encoded in the first byte and takes up three more zero bits in the second byte.
To make this clear, let's have a look at the first three bytes:

|                    | 100      | 64       | 6        |
| --------           | -------- | -------- | -------- |
| Local bit position | 01234567 | 01234567 | 01234567 |
| Binary value       | 00100110 | 00000010 | 01100000 |
| Index              | 00000000 | 00011111 | 11111122 |

So the actual bit pattern for the first and second value is 00100110000 and 00010011000, starting with the lowest bit from the left.

### BoolVec
Additionally *compvec* provides a structure for storing booleans bitwise.
Eight bools are written to a single byte:

```rust
let mut boolvec = BoolVec::from(vec![false, true, false, true, false, true]);
boolvec.push(true);
// Prints "false"
println!("{}", boolvec.get(4));
```

## Notes
### Iterators
There is no way to iterate over the integers by reference or mutably, since they are stored compressed.
But you can use the function *iter_values* for iterating over the integer values.

```rust
let bytevec = ByteVec::<u32, u8>::from(vec![32, 64, 999, 5_000]);
// Prints values 32, 64, 999 and 5000
for v in bytevec.iter_values() {
    println!("{v}");
}
```

### A word on bitpacking
Using the crate *bitpacking* you can compress a sequence of 32 bit unsigned integers into bytes efficiently with SIMD. I was looking for a generic solution for storing any type of signed and unsigned integers. *compvec* accomplishes that in a simple way.
Sure, one could use a wrapper, such that any integer type is first encoded into a Vec of u32. But I don't think it's applicable.

### Running tests
There are a lot of tests and running all of them may take some time, depending on your hardware. It is recommended to run the tests in release mode:

```bash
cargo t --release
```
Or run a specific test function:

```bash
cargo t byte_vec_u16_u8
```
