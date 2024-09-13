#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use core::arch::aarch64::{uint32x4_t, vuzp1q_u32, vuzp2q_u32};
use core::mem::transmute;
use wide::u32x8;

// use std::simd::Simd;
// #[inline(never)]
// pub fn simd_deinterleave(a: Simd<u32, 8>, b: Simd<u32, 8>) -> (Simd<u32, 8>, Simd<u32, 8>) {
//     a.deinterleave(b)
// }

#[inline(always)]
pub fn wide_deinterleave(a: u32x8, b: u32x8) -> (u32x8, u32x8) {
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx"
    ))]
    {
        unimplemented!()
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe {
        let (a1, a2): (uint32x4_t, uint32x4_t) = transmute(a);
        let (b1, b2): (uint32x4_t, uint32x4_t) = transmute(b);
        let a_even = vuzp1q_u32(a1, a2);
        let a_odd = vuzp2q_u32(a1, a2);
        let b_even = vuzp1q_u32(b1, b2);
        let b_odd = vuzp2q_u32(b1, b2);
        let ab_even: u32x8 = transmute((a_even, b_even));
        let ab_odd: u32x8 = transmute((a_odd, b_odd));
        return (ab_even, ab_odd);
    }
    unsafe {
        let a = a.as_array_ref();
        let b = b.as_array_ref();
        (
            u32x8::new([
                *a.get_unchecked(0),
                *a.get_unchecked(2),
                *a.get_unchecked(4),
                *a.get_unchecked(6),
                *b.get_unchecked(0),
                *b.get_unchecked(2),
                *b.get_unchecked(4),
                *b.get_unchecked(6),
            ]),
            u32x8::new([
                *a.get_unchecked(1),
                *a.get_unchecked(3),
                *a.get_unchecked(5),
                *a.get_unchecked(7),
                *b.get_unchecked(1),
                *b.get_unchecked(3),
                *b.get_unchecked(5),
                *b.get_unchecked(7),
            ]),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deinterleave() {
        let a = u32x8::new([0, 1, 2, 3, 4, 5, 6, 7]);
        let b = u32x8::new([8, 9, 10, 11, 12, 13, 14, 15]);
        let (c, d) = wide_deinterleave(a, b);

        assert_eq!(c.to_array(), [0, 2, 4, 6, 8, 10, 12, 14]);
        assert_eq!(d.to_array(), [1, 3, 5, 7, 9, 11, 13, 15]);
    }
}
