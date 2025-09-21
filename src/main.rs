use compvec::boolvec::*;
use compvec::bytevec::*;
use compvec::bitvec::*;
use compvec::types::*;
use compvec::veclike::*;
use std::time::SystemTime;

/// A simple performance test.
fn perf<V: VecLike>(v: &V)
where
    V::Type: Integer,
{
    let mut sum = V::Type::ZERO;
    let n = v.len();
    let now = SystemTime::now();
    for i in 0..n {
        if i % 4 == 0 {
            sum += v.get(i);
        }
    }
    println!("Every 4th {sum} {} ms", now.elapsed().unwrap().as_millis());
    let now = SystemTime::now();
    let sum = v.iterate().step_by(4).sum::<V::Type>();
    println!("Every 4th iter {sum} {} ms", now.elapsed().unwrap().as_millis());
    let now = SystemTime::now();
    let sum = v.iterate().sum::<V::Type>();
    println!("All {sum} {} ms", now.elapsed().unwrap().as_millis());
}

fn main() {
    {
// Values will be stored in 10 bits and the maximum value is 1,023
let mut bitvec = BitVec::<usize, u8>::with_max_value(4, 1_000);
bitvec.push(100);
bitvec.push(200);
bitvec.push(400);
// This will increase the number of bits per value to 11 bits
bitvec.push(2_000);
println!("{:?}", bitvec); // Prints "400"
    }
    {
        const N: usize = 200_000_000;
        let mut bool_vec_ref = vec![false; N];
        for i in 0..N {
            bool_vec_ref[i] = i % 4 == 0;
        }
        let bool_vec = BoolVec::from(bool_vec_ref.clone());
        let now = SystemTime::now();
        let sum = bool_vec_ref.iter().filter(|&&x| x).count();
        println!("Vec<bool> {sum} {} ms", now.elapsed().unwrap().as_millis());
        let now = SystemTime::now();
        let sum = bool_vec.iterate().filter(|&x| x).count();
        println!("BoolVec {sum} {} ms", now.elapsed().unwrap().as_millis());
    }
    {
        let n = 120_000_000;
        let mut byte_vec_ref = vec![0; n];
        for i in 0..n {
            byte_vec_ref[i] = i as u64;
        }
        println!("Performance test Vec");
        perf(&byte_vec_ref);
        let byte_vec = ByteVec::<u64, u8>::from(byte_vec_ref.clone());
        println!("Performance test ByteVec");
        println!("Stride elements {}", byte_vec.stride());
        perf(&byte_vec);
        let bit_vec = BitVec::<u64, u8>::from(byte_vec_ref.clone());
        println!("Performance test BitVec");
        println!("Stride bits {}", bit_vec.stride_bits());
        perf(&bit_vec);
    }
}
