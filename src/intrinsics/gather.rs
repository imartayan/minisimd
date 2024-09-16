use wide::u64x4;

// use std::simd::Simd;
// #[inline(never)]
// pub fn simd_gather(source: Simd<*const u64, 2>) -> Simd<u64, 2> {
//     unsafe { Simd::<u64, 2>::gather_ptr(source) }
// }

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn gather_avx(ptr: *const u8, offsets: u64x4) -> u64x4 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::_mm512_i64gather_epi64;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::_mm512_i64gather_epi64;

    _mm512_i64gather_epi64(offsets, ptr, 1)
}

#[inline(always)]
pub fn gather(ptr: *const u8, offsets: u64x4) -> u64x4 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("avx512f") {
        return unsafe { gather_avx(source) };
    }
    unsafe {
        let source = u64x4::splat(ptr as u64) + offsets;
        u64x4::new(source.to_array().map(|p| *(p as *const u64)))
    }
}
