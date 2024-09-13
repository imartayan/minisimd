use wide::u64x4;

// use std::simd::Simd;
// #[inline(never)]
// pub fn simd_gather(source: Simd<*const u64, 2>) -> Simd<u64, 2> {
//     unsafe { Simd::<u64, 2>::gather_ptr(source) }
// }

#[inline(always)]
pub fn wide_gather(source: u64x4) -> u64x4 {
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx"
    ))]
    {
        unimplemented!()
    }
    unsafe { u64x4::new(source.to_array().map(|p| *(p as *const u64))) }
}
