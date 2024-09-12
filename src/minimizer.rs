use crate::ringbuf::RingBuf;
use core::array::from_fn;
use wide::u32x8;

type S = u32x8;

#[inline(always)]
fn sliding_min_par_it(
    it: impl ExactSizeIterator<Item = S>,
    w: usize,
) -> impl ExactSizeIterator<Item = S> {
    assert!(w > 0);
    assert!(w < (1 << 15), "This method is not tested for large w.");
    assert!(it.size_hint().0 * 8 < (1 << 32));

    let mut prefix_min = S::splat(u32::MAX);
    let mut ring_buf = RingBuf::new(w, prefix_min);
    // We only compare the upper 16 bits of each hash.
    // Ties are broken automatically in favour of lower pos.
    let val_mask = S::splat(0xffff_0000);
    let pos_mask = S::splat(0x0000_ffff);
    let max_pos = S::splat((1 << 16) - 1);
    let mut pos = S::splat(0);
    let mut pos_offset = S::new(from_fn(|l| {
        (l * (it.size_hint().0.saturating_sub(w - 1))) as u32
    }));

    let mut it = it.map(
        #[inline(always)]
        move |val| {
            // Make sure the position does not interfere with the hash value.
            if pos == max_pos {
                let delta = S::splat((1 << 16) - 2 - w as u32);
                pos -= delta;
                prefix_min -= delta;
                pos_offset += delta;
                for x in &mut *ring_buf {
                    *x -= delta;
                }
            }

            let elem = (val & val_mask) | pos;
            pos += S::splat(1);
            ring_buf.push(elem);
            prefix_min = prefix_min.min(elem);

            // After a chunk has been filled, compute suffix minima.
            if ring_buf.idx() == 0 {
                let mut suffix_min = ring_buf[w - 1];
                for i in (0..w - 1).rev() {
                    suffix_min = suffix_min.min(ring_buf[i]);
                    ring_buf[i] = suffix_min;
                }
                prefix_min = elem; // slightly faster than assigning S::splat(u32::MAX)
            }
            let suffix_min = unsafe { *ring_buf.get_unchecked(ring_buf.idx()) };
            (prefix_min.min(suffix_min) & pos_mask) + pos_offset
        },
    );

    // This optimizes better than it.skip(w-1).
    it.by_ref().take(w - 1).for_each(drop);
    it
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::array::from_fn;

    const LANES: usize = 8;

    #[test]
    fn test_sliding_min() {
        let w = 3;
        let hashes = [7, 5, 9, 8, 6, 3].map(|x| x << 16);
        let hashes: [u32; 6 * LANES] = from_fn(|i| hashes[i / LANES]);
        let expected = [1, 1, 4, 5]; // positions of the minimizers
        let expected: [u32; 4 * LANES] = from_fn(|i| expected[i / LANES] + 4 * (i % LANES) as u32);

        let chunks = hashes.as_chunks::<8>().0;
        let it = chunks.iter().map(|&t| t.into());

        let res: Vec<u32> = sliding_min_par_it(it, w)
            .flat_map(|t| t.to_array())
            .collect();
        assert_eq!(res, expected);
    }
}
