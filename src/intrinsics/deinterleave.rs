use core::mem::transmute;
use wide::u32x8;

// use std::simd::Simd;
// #[inline(never)]
// pub fn simd_deinterleave(a: Simd<u32, 8>, b: Simd<u32, 8>) -> (Simd<u32, 8>, Simd<u32, 8>) {
//     a.deinterleave(b)
// }

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn deinterleave_avx(a: u32x8, b: u32x8) -> (u32x8, u32x8) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{__m256, _mm256_permute_ps, _mm256_shuffle_ps};
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{__m256, _mm256_permute_ps, _mm256_shuffle_ps};

    let a = transmute(a);
    let b = transmute(b);
    let abab_even = _mm256_shuffle_ps(a, b, 0b10_00_10_00);
    let abab_odd = _mm256_shuffle_ps(a, b, 0b11_01_11_01);
    let ab_even = _mm256_permute_ps(abab_even, 0b11_01_10_00);
    let ab_odd = _mm256_permute_ps(abab_odd, 0b11_01_10_00);
    (transmute(a), transmute(b))
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn deinterleave_neon(a: u32x8, b: u32x8) -> (u32x8, u32x8) {
    #[cfg(target_arch = "aarch64")]
    use core::arch::aarch64::{uint32x4_t, vuzp1q_u32, vuzp2q_u32};

    let (a1, a2): (uint32x4_t, uint32x4_t) = transmute(a);
    let (b1, b2): (uint32x4_t, uint32x4_t) = transmute(b);
    let a_even = vuzp1q_u32(a1, a2);
    let a_odd = vuzp2q_u32(a1, a2);
    let b_even = vuzp1q_u32(b1, b2);
    let b_odd = vuzp2q_u32(b1, b2);
    let ab_even: u32x8 = transmute((a_even, b_even));
    let ab_odd: u32x8 = transmute((a_odd, b_odd));
    (ab_even, ab_odd)
}

#[inline(always)]
pub fn deinterleave(a: u32x8, b: u32x8) -> (u32x8, u32x8) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("avx") {
        return unsafe { deinterleave_avx(a, b) };
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return unsafe { deinterleave_neon(a, b) };
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
        let (c, d) = deinterleave(a, b);

        assert_eq!(c.to_array(), [0, 2, 4, 6, 8, 10, 12, 14]);
        assert_eq!(d.to_array(), [1, 3, 5, 7, 9, 11, 13, 15]);
    }
}
