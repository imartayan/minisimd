use wide::u64x4;

// use std::simd::Simd;
// #[inline(never)]
// pub fn simd_gather(source: Simd<*const u64, 2>) -> Simd<u64, 2> {
//     unsafe { Simd::<u64, 2>::gather_ptr(source) }
// }

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn gather_avx(source: u64x4) -> u64x4 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    unimplemented!()
}

#[inline(always)]
pub fn gather(source: u64x4) -> u64x4 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("avx") {
        return unsafe { gather_avx(a, b) };
    }
    unsafe { u64x4::new(source.to_array().map(|p| *(p as *const u64))) }
}
