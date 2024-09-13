// #![feature(portable_simd)]

pub mod intrinsics;
pub mod minimizer;
pub mod ringbuf;

use biotest::Format;

pub fn gen_seq(len: usize) -> Vec<u8> {
    let mut rng = biotest::rand();
    let mut seq = Vec::with_capacity(len);
    let generator = biotest::Sequence::builder()
        .sequence_len(len)
        .build()
        .unwrap();
    generator.record(&mut seq, &mut rng).unwrap();
    seq
}
