use core::mem::transmute;
use wide::u32x8;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn lookup_avx(t: u32x8, idx: u32x8) -> u32x8 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::_mm256_permutevar_ps;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::_mm256_permutevar_ps;

    transmute(_mm256_permutevar_ps(transmute(t), transmute(idx)))
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn lookup_neon(t: u32x8, idx: u32x8) -> u32x8 {
    #[cfg(target_arch = "aarch64")]
    use core::arch::aarch64::{uint8x16_t, vqtbl1q_u8};

    const OFFSET: u32 = if cfg!(target_endian = "little") {
        0x03_02_01_00
    } else {
        0x00_01_02_03
    };

    let idx = idx * u32x8::splat(0x04_04_04_04) + u32x8::splat(OFFSET);
    let (t1, t2): (uint8x16_t, uint8x16_t) = transmute(t);
    let (i1, i2): (uint8x16_t, uint8x16_t) = transmute(idx);
    let r1 = vqtbl1q_u8(t1, i1);
    let r2 = vqtbl1q_u8(t2, i2);
    transmute((r1, r2))
}

#[inline(always)]
#[deprecated(
    note = "This function does not use SIMD, make sure you are compiling using `target=native`."
)]
unsafe fn lookup_fallback(t: u32x8, idx: u32x8) -> u32x8 {
    let t = t.as_array_ref();
    u32x8::new(idx.to_array().map(|i| *t.get_unchecked(i as usize)))
}

#[inline(always)]
pub fn lookup(t: u32x8, idx: u32x8) -> u32x8 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("avx") {
        return unsafe { lookup_avx(t, idx) };
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return unsafe { lookup_neon(t, idx) };
    }
    unsafe { lookup_fallback(t, idx) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lookup() {
        let t = u32x8::new([1000, 1001, 1002, 1003, 1000, 1001, 1002, 1003]);
        let idx = u32x8::new([2, 0, 3, 1, 0, 2, 1, 0]);
        let res = lookup(t, idx);

        assert_eq!(
            res.to_array(),
            [1002, 1000, 1003, 1001, 1000, 1002, 1001, 1000]
        );
    }
}
