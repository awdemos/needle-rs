//! Exact reproduction of `np.random.RandomState(seed).permutation(n)` — the
//! numpy legacy MT19937 + Fisher-Yates (`rk_interval`) pipeline — because the
//! Hadamard MLP permutations are fixed by the training code (`_hada_perms`)
//! and the runtime applies them from the archive. Ported from numpy's
//! `numpy/random/src/legacy/legacy-distributions.c` and `mt19937.c`.

/// numpy legacy MT19937 (init_genrand).
pub struct Mt19937 {
    state: [u32; 624],
    index: usize,
}

impl Mt19937 {
    pub fn new(seed: u32) -> Mt19937 {
        let mut state = [0u32; 624];
        state[0] = seed;
        for i in 1..624 {
            // state[i] = 1812433253 * (state[i-1] ^ (state[i-1] >> 30)) + i
            let prev = state[i - 1];
            state[i] = 1812433253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        Mt19937 { state, index: 624 }
    }

    fn twist(&mut self) {
        const UPPER_MASK: u32 = 0x80000000;
        const LOWER_MASK: u32 = 0x7fffffff;
        const MATRIX_A: u32 = 0x9908b0df;
        for i in 0..624 {
            let y = (self.state[i] & UPPER_MASK) | (self.state[(i + 1) % 624] & LOWER_MASK);
            let mut next = self.state[(i + 397) % 624] ^ (y >> 1);
            if y & 1 != 0 {
                next ^= MATRIX_A;
            }
            self.state[i] = next;
        }
        self.index = 0;
    }

    /// One 32-bit draw (`rk_random` returns buffered words one at a time).
    pub fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            self.twist();
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        y
    }

    /// `rk_interval`: uniform int in [0, max] via masked rejection on one draw.
    pub fn interval(&mut self, max: u32) -> u32 {
        if max == 0 {
            return 0;
        }
        let mut mask = max;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        loop {
            let value = self.next_u32() & mask;
            if value <= max {
                return value;
            }
        }
    }

    /// `RandomState.permutation(n)` for scalar n: Fisher-Yates with rk_interval.
    pub fn permutation(&mut self, n: usize) -> Vec<u32> {
        let mut arr: Vec<u32> = (0..n as u32).collect();
        for i in (1..n).rev() {
            let j = self.interval(i as u32) as usize;
            arr.swap(i, j);
        }
        arr
    }
}

/// `_hada_perms(n, split)`: seeds (11, 13); split permutes each half with
/// seed+977 offset added to the upper half.
pub fn hada_perms(n: usize, split: bool) -> (Vec<u32>, Vec<u32>) {
    let mut perms = Vec::new();
    for seed in [11u32, 13u32] {
        if split {
            let h = n / 2;
            let mut lower = Mt19937::new(seed).permutation(h);
            let mut upper = Mt19937::new(seed + 977).permutation(h);
            for v in upper.iter_mut() {
                *v += h as u32;
            }
            lower.extend(upper);
            perms.push(lower);
        } else {
            perms.push(Mt19937::new(seed).permutation(n));
        }
    }
    (perms[0].clone(), perms[1].clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_numpy_random_state() {
        // expected values generated with numpy:
        //   np.random.RandomState(11).permutation(16)
        //   np.random.RandomState(13).permutation(16)
        let expect11 = [8u32, 3, 6, 10, 15, 4, 5, 14, 2, 13, 12, 7, 1, 11, 0, 9];
        let expect13 = [8u32, 11, 12, 1, 5, 7, 9, 13, 3, 4, 15, 6, 14, 10, 0, 2];
        assert_eq!(Mt19937::new(11).permutation(16), expect11.to_vec());
        assert_eq!(Mt19937::new(13).permutation(16), expect13.to_vec());
        // 1024-element head: np.random.RandomState(11).permutation(1024)[:8]
        let head1024 = Mt19937::new(11).permutation(1024);
        assert_eq!(&head1024[..8], &[177, 777, 1006, 410, 709, 670, 932, 538]);
    }
}
