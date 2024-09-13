#![allow(unreachable_code)]

mod deinterleave;
mod gather;
mod lookup;

pub use deinterleave::wide_deinterleave;
pub use gather::wide_gather;
pub use lookup::wide_lookup;
