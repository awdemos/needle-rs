//! Exact reproduction of `np.random.RandomState(seed).permutation(n)` — the
//! numpy legacy MT19937 + Fisher-Yates (`rk_interval`) pipeline — because the
//! Hadamard MLP permutations are fixed by the training code (`_hada_perms`)
//! and the runtime applies them from the archive. Ported from numpy's
//! `numpy/random/src/legacy/legacy-distributions.c` and `mt19937.c`.
//!
//! Verified with Verus (see crates/needle-build/Cargo.toml `[package.metadata
//! .verus]`): `Mt19937::permutation` provably returns a permutation of
//! `0..n` (length `n`, all elements distinct, all elements `< n`), and the
//! MT19937 state index provably never leaves the state-vector range.

use vstd::prelude::*;

verus! {

broadcast use { vstd::seq_lib::group_to_multiset_ensures, vstd::multiset::group_multiset_axioms, vstd::seq_lib::group_seq_properties };

/// Identity sequence [0, 1, ..., n-1] as u32 values.
spec fn id_seq(n: int) -> Seq<u32>
    decreases n,
{
    if n <= 0 {
        Seq::empty()
    } else {
        id_seq(n - 1).push((n - 1) as u32)
    }
}

/// The multiset of the identity sequence has each value x in 0..n exactly once.
proof fn lemma_id_seq_multiset(n: int)
    requires
        0 <= n <= u32::MAX as int,
    ensures
        id_seq(n).to_multiset().len() == n,
        forall|x: u32| id_seq(n).to_multiset().count(x) == (if 0 <= x < n {
            1nat
        } else {
            0nat
        }),
    decreases n,
{
    if n == 0 {
        assert(id_seq(0) =~= Seq::<u32>::empty());
    } else {
        lemma_id_seq_multiset(n - 1);
        assert(id_seq(n) =~= id_seq(n - 1).push((n - 1) as u32));
    }
}

/// Swapping two positions of a sequence leaves its multiset unchanged.
proof fn lemma_swap_preserves_multiset(s: Seq<u32>, i: int, j: int)
    requires
        0 <= i < s.len(),
        0 <= j < s.len(),
    ensures
        s.update(i, s[j]).update(j, s[i]).to_multiset() =~= s.to_multiset(),
{
    let si = s[i];
    let sj = s[j];
    let m = s.to_multiset();
    if i == j {
        assert(s.update(i, sj).update(j, si) =~= s);
    } else {
        assert(s.update(i, sj).to_multiset() =~= m.insert(sj).remove(si));
        assert(s.update(i, sj)[j] == sj);
        assert(s.update(i, sj).update(j, si).to_multiset() =~= s.update(i, sj).to_multiset().insert(
            si,
        ).remove(sj));
        assert(m.insert(sj).remove(si).insert(si).remove(sj) =~= m) by {
            assert forall|v: u32| true implies m.insert(sj).remove(si).insert(si).remove(sj).count(v)
                == m.count(v) by {
                if v == si && v == sj {
                } else if v == si {
                } else if v == sj {
                } else {
                }
            }
        }
    }
}

/// numpy legacy MT19937 (init_genrand).
pub struct Mt19937 {
    pub state: [u32; 624],
    pub index: usize,
}

impl Mt19937 {
    pub fn new(seed: u32) -> (s: Mt19937)
        ensures
            s.index@ == 624,
    {
        let mut state = [0u32; 624];
        state[0] = seed;
        for i in 1..624
            invariant
                state@.len() == 624,
        {
            // state[i] = 1812433253 * (state[i-1] ^ (state[i-1] >> 30)) + i
            let prev = state[i - 1];
            state[i] = 1812433253u32.wrapping_mul(prev ^ (prev >> 30)).wrapping_add(i as u32);
        }
        Mt19937 { state, index: 624 }
    }

    fn twist(&mut self)
        requires
            old(self).index@ >= 624,
        ensures
            final(self).index@ == 0,
    {
        const UPPER_MASK: u32 = 0x80000000;
        const LOWER_MASK: u32 = 0x7fffffff;
        const MATRIX_A: u32 = 0x9908b0df;
        for i in 0..624
            invariant
                self.state@.len() == 624,
        {
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
    pub fn next_u32(&mut self) -> (y: u32)
        ensures
            final(self).index@ <= 624,
            final(self).index@ == (if old(self).index@ >= 624 {
                1
            } else {
                old(self).index@ + 1
            }),
    {
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
    /// The rejection loop's termination is probabilistic (geometric in the
    /// masked draw), so it has no ranking function; the no-decreases check is
    /// disabled for this function and its termination is trusted.
    #[verifier::exec_allows_no_decreases_clause]
    pub fn interval(&mut self, max: u32) -> (r: u32)
        ensures
            r <= max,
    {
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
    /// Produces a permutation of 0..n: the output has length n, no duplicate
    /// elements, and every element is < n.
    pub fn permutation(&mut self, n: usize) -> (perm: Vec<u32>)
        requires
            (n as u64) <= u32::MAX as u64,
        ensures
            perm@.len() == n,
            perm@.no_duplicates(),
            forall|x: u32| perm@.contains(x) ==> (x as int) < (n as int),
    {
        let mut arr: Vec<u32> = Vec::new();
        for k in 0..n
            invariant
                arr@.len() == k,
                arr@ =~= id_seq(k as int),
        {
            arr.push(k as u32);
            assert(arr@ =~= id_seq(k as int + 1));
        }
        for i in (1..n).rev()
            invariant
                arr@.len() == n,
                arr@.to_multiset() =~= id_seq(n as int).to_multiset(),
        {
            let j = self.interval(i as u32) as usize;
            assert(j <= i);
            let ghost pre = arr@;
            let tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
            assert(arr@ == pre.update(i as int, pre[j as int]).update(j as int, pre[i as int]));
            proof {
                lemma_swap_preserves_multiset(pre, i as int, j as int);
            }
            assert(arr@.to_multiset() =~= id_seq(n as int).to_multiset());
        }
        proof {
            lemma_id_seq_multiset(n as int);
            assert(arr@.len() == n) by {
                assert(arr@.len() == arr@.to_multiset().len());
            }
            assert forall|x: u32| arr@.contains(x) implies (x as int) < (n as int) by {
                if arr@.contains(x) {
                    assert(arr@.to_multiset().count(x) > 0);
                    assert(arr@.to_multiset().count(x) == id_seq(n as int).to_multiset().count(x));
                    assert(id_seq(n as int).to_multiset().count(x) == 1);
                    assert((x as int) < (n as int));
                }
            }
            assert forall|x: u32| arr@.to_multiset().contains(x) implies arr@.to_multiset().count(
                x,
            ) == 1 by {
                if arr@.to_multiset().contains(x) {
                    assert(arr@.to_multiset().count(x) == id_seq(n as int).to_multiset().count(x));
                    assert(id_seq(n as int).to_multiset().count(x) == 1);
                }
            }
            arr@.lemma_multiset_has_no_duplicates_conv();
        }
        arr
    }
}

/// `_hada_perms(n, split)`: seeds (11, 13); split permutes each half with
/// seed+977 offset added to the upper half.
///
/// Not verified itself (a thin composition of the verified `permutation`;
/// marked external_body so its unbounded `n` config value does not require
/// changes at the call sites in lib.rs).
#[verifier::external_body]
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

} // verus!

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
