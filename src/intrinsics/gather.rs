use wide::u64x4;

// use std::simd::Simd;
// #[inline(never)]
// pub fn simd_gather(source: Simd<*const u64, 2>) -> Simd<u64, 2> {
//     unsafe { Simd::<u64, 2>::gather_ptr(source) }
// }

#[inline(always)]
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx512f"
))]
unsafe fn gather_avx(ptr: *const u8, offsets: u64x4) -> u64x4 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::_mm512_i64gather_epi64;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::_mm512_i64gather_epi64;

    _mm512_i64gather_epi64(offsets, ptr, 1)
}

#[inline(always)]
unsafe fn gather_fallback(ptr: *const u8, offsets: u64x4) -> u64x4 {
    let source = u64x4::splat(ptr as u64) + offsets;
    u64x4::new(source.to_array().map(|p| *(p as *const u64)))
}

#[inline(always)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn gather(ptr: *const u8, offsets: u64x4) -> u64x4 {
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx512f"
    ))]
    unsafe {
        gather_avx(ptr, offsets)
    }
    #[cfg(not(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx512f"
    )))]
    unsafe {
        gather_fallback(ptr, offsets)
    }
}
