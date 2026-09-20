//! Regression tests: hand-built malformed archives must return `Err`, never
//! panic, abort, or hang. All archives here are synthetic (no model download).
use needle_format::{
    read_archive, read_archive_from, dequant_cq, Config, CqMatrix, Error, Tensor, DTYPE_CQ,
    DTYPE_FP32, DTYPE_RAW, TAG_V2, TAG_V3,
};

const HEADER_LEN: usize = 196;
const REC_LEN: usize = 44;

fn put_u32(buf: &mut [u8], at: usize, v: u32) {
    buf[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

fn put_u64(buf: &mut [u8], at: usize, v: u64) {
    buf[at..at + 8].copy_from_slice(&v.to_le_bytes());
}

/// A minimal but valid V3 header: 20 layers, sane geometry, 2 orders / 5 sites.
fn header(num_tensors: u32, cb_len: u32) -> Vec<u8> {
    let mut h = vec![0u8; HEADER_LEN];
    put_u32(&mut h, 0, TAG_V3);
    put_u32(&mut h, 4, num_tensors);
    put_u32(&mut h, 8, cb_len);
    put_u32(&mut h, 40, 20); // num_layers
    put_u32(&mut h, 56, 1024); // hada_n
    put_u32(&mut h, 68, 0xFFFF_FFFF); // gmask lo
    put_u32(&mut h, 72, 0); // gmask hi
    put_u32(&mut h, 104, 2); // num_orders
    put_u32(&mut h, 108, 2);
    put_u32(&mut h, 112, 3);
    put_u32(&mut h, 124, 5); // num_sites
    for (i, site) in [3u32, 7, 11, 15, 19].iter().enumerate() {
        put_u32(&mut h, 128 + i * 4, *site);
    }
    h
}

/// One 44-byte directory record (`<BBHIIIIQQII`).
#[allow(clippy::too_many_arguments)]
fn record(
    dtype: u8,
    shape: &[u32],
    offset: u64,
    nbytes: u64,
    group_size: u32,
    bits: u32,
) -> Vec<u8> {
    let mut r = vec![0u8; REC_LEN];
    r[0] = dtype;
    r[1] = shape.len() as u8;
    for (i, &d) in shape.iter().enumerate() {
        put_u32(&mut r, 4 + i * 4, d);
    }
    put_u64(&mut r, 20, offset);
    put_u64(&mut r, 28, nbytes);
    put_u32(&mut r, 36, group_size);
    put_u32(&mut r, 40, bits);
    r
}

fn codebook_bytes(n: u32) -> Vec<u8> {
    let mut b = Vec::new();
    for i in 0..n {
        b.extend_from_slice(&(i as f32).to_le_bytes());
    }
    b
}

/// `Archive` does not implement `Debug`, so `unwrap_err` is unavailable.
fn archive_err(r: needle_format::Result<needle_format::Archive>) -> Error {
    match r {
        Err(e) => e,
        Ok(_) => panic!("expected Err, got a parsed archive"),
    }
}

/// Assemble a full archive: header + codebook + contiguous directory records
/// + tensor blobs (all zeros, `nbytes` each), offsets patched correctly.
fn archive(cb_len: u32, specs: &[(u8, &[u32], u64, u32, u32)]) -> Vec<u8> {
    let mut a = header(specs.len() as u32, cb_len);
    a.extend_from_slice(&codebook_bytes(cb_len));
    let mut off = (a.len() + specs.len() * REC_LEN) as u64;
    for &(dtype, shape, nbytes, group, bits) in specs {
        a.extend_from_slice(&record(dtype, shape, off, nbytes, group, bits));
        off += nbytes;
    }
    for &(_, _, nbytes, _, _) in specs {
        a.extend_from_slice(&vec![0u8; nbytes as usize]);
    }
    a
}

/// Header + codebook + one CQ record `[out, inp]` with a tight zero blob
/// (packed rows + one f16 norm per group).
fn cq_archive(out: u32, inp: u32, group_size: u32, bits: u32, cb_len: u32) -> Vec<u8> {
    let g = group_size as usize;
    let nbytes = if g == 0 {
        0 // the reader must reject group_size == 0 before touching the blob
    } else {
        let in_pad = (inp as usize).div_ceil(g) * g;
        let row_bytes = if bits == 5 { in_pad * 2 / 8 } else { in_pad * bits as usize / 8 };
        ((out as usize * row_bytes) + 2 * out as usize * (in_pad / g)) as u64
    };
    archive(cb_len, &[(DTYPE_CQ, &[out, inp], nbytes, group_size, bits)])
}

// ---------- positive controls: valid archives still parse ----------

#[test]
fn valid_minimal_archive_parses() {
    let a = header(0, 0);
    let ar = read_archive_from(a).unwrap();
    assert_eq!(ar.config.num_layers, 20);
    assert_eq!(ar.config.global_layers, (0..20).collect::<Vec<_>>());
    assert_eq!(ar.config.engram_orders, vec![2, 3]);
    assert_eq!(ar.config.engram_layers, vec![3, 7, 11, 15, 19]);
    assert!(ar.tensors.is_empty());
}

#[test]
fn valid_cq_tensor_parses_and_dequants() {
    let a = cq_archive(2, 8, 8, 5, 28);
    let ar = read_archive_from(a).unwrap();
    assert_eq!(ar.codebook.len(), 28);
    let Tensor::Cq(m) = &ar.tensors[0] else {
        panic!("expected CQ tensor");
    };
    // Ternary crumbs use the analytic codebook; no header book needed.
    let cb = ar.codebook_for(5, m.group_size).unwrap();
    assert_eq!(cb.len(), 3);
    let w = dequant_cq(m, &cb).unwrap();
    assert_eq!(w.len(), 2 * 8);
    // All-zero packed data -> all-zero weights regardless of codebook.
    assert!(w.iter().all(|&v| v == 0.0));
}

#[test]
fn valid_raw_and_fp32_tensors_parse() {
    let a = archive(0, &[(DTYPE_FP32, &[2, 2], 16, 0, 0), (DTYPE_RAW, &[7], 5, 0, 0)]);
    let ar = read_archive_from(a).unwrap();
    assert_eq!(ar.tensors.len(), 2);
    assert!(matches!(&ar.tensors[0], Tensor::Fp32 { data, .. } if data.len() == 4));
    assert!(matches!(&ar.tensors[1], Tensor::Raw(b) if b.len() == 5));
}

// ---------- tags / version ----------

#[test]
fn v2_tag_is_rejected_not_misparsed() {
    let mut a = header(0, 0);
    put_u32(&mut a, 0, TAG_V2);
    let err = archive_err(read_archive_from(a));
    assert!(matches!(err, Error::UnsupportedVersion(t) if t == TAG_V2), "{err}");
}

#[test]
fn unknown_tag_is_rejected() {
    let mut a = header(0, 0);
    put_u32(&mut a, 0, 0xDEAD_BEEF);
    assert!(matches!(read_archive_from(a), Err(Error::BadTag(_))));
}

#[test]
fn missing_file_reports_io_not_truncation() {
    let err = archive_err(read_archive(std::path::Path::new(
        "/nonexistent/needle-definitely-missing.cact",
    )));
    assert!(matches!(err, Error::Io(_)), "{err}");
}

// ---------- header geometry ----------

#[test]
fn oversized_engram_counts_are_capped() {
    // The header has fixed 4-order / 16-site slots; like the Python reader's
    // `orders4[:num_orders]`, excess counts are capped rather than erroring.
    let mut a = header(0, 0);
    put_u32(&mut a, 104, 100); // num_orders
    put_u32(&mut a, 124, 0xFFFF_FFFF); // num_sites
    let ar = read_archive_from(a).unwrap();
    assert_eq!(ar.config.engram_orders.len(), 4);
    assert_eq!(ar.config.engram_layers.len(), 16);
    assert_eq!(ar.config.engram_orders, vec![2, 3, 0, 0]);
}

#[test]
fn num_layers_65_is_rejected() {
    let mut a = header(0, 0);
    put_u32(&mut a, 40, 65);
    assert!(matches!(read_archive_from(a), Err(Error::BadGeometry(_))));
}

#[test]
fn huge_num_tensors_errors_instead_of_aborting() {
    // No directory records follow; the loop must hit the bounds check
    // immediately instead of pre-allocating for 4 billion tensors.
    let a = header(u32::MAX, 0);
    let err = archive_err(read_archive_from(a));
    assert!(matches!(err, Error::Truncated("directory")), "{err}");
    // Even with a well-formed record present, it must still Err once the
    // directory runs out — and do so promptly.
    let a = archive(0, &[(DTYPE_RAW, &[1], 0, 0, 0)]);
    let mut a = a;
    put_u32(&mut a, 4, u32::MAX);
    assert!(matches!(
        archive_err(read_archive_from(a)),
        Error::Truncated("directory")
    ));
}

#[test]
fn hada_blocks_zero_does_not_underflow() {
    // Corrupt hada_n = 0 must not panic (debug underflow) — treated as 1.
    assert_eq!(Config::hada_blocks(0), Config::hada_blocks(1));
    assert_eq!(Config::hada_blocks(0), (1, 1));
    assert_eq!(Config::hada_blocks(1024), (32, 32));
}

// ---------- CQ record validation ----------

#[test]
fn cq_group_size_zero_is_rejected() {
    let a = cq_archive(1, 8, 0, 5, 28);
    assert!(matches!(read_archive_from(a), Err(Error::BadGeometry(_))));
}

#[test]
fn cq_group_size_non_power_of_two_is_rejected() {
    // 96 would reach fwht's `x[j + h]` out-of-bounds panic in dequant.
    let a = cq_archive(1, 96, 96, 5, 28);
    assert!(matches!(read_archive_from(a), Err(Error::BadGeometry(_))));
}

#[test]
fn cq_bits_six_is_rejected() {
    // Valid widths are {1,2,3,4,5}; bits=6 overflows `1u64 << bits` in unpack.
    let a = cq_archive(1, 8, 8, 6, 28);
    assert!(matches!(read_archive_from(a), Err(Error::BadBits(6))));
}

#[test]
fn cq_bits_zero_and_huge_are_rejected() {
    for bits in [0u32, 7, 32, 0xFFFF_FFFF] {
        let a = cq_archive(1, 8, 8, bits, 28);
        assert!(matches!(read_archive_from(a), Err(Error::BadBits(b)) if b == bits), "bits={bits}");
    }
}

#[test]
fn short_codebook_errors_in_codebook_for_and_dequant() {
    // cb_len = 4 (only the cb2 book): reading succeeds, but asking for the
    // 3- or 4-bit book must Err instead of silently truncating.
    let mut a = header(1, 4);
    a.extend_from_slice(&codebook_bytes(4));
    // bits=3 CQ tensor, group 8, out 1, inp 8: packed = 8*3/8 = 3, norms 2.
    // First packed byte 0xFF makes code index 7 (> 4 entries in the book).
    let dir_end = a.len() + REC_LEN;
    a.extend_from_slice(&record(DTYPE_CQ, &[1, 8], dir_end as u64, 5, 8, 3));
    a.extend_from_slice(&[0xFF, 0x00, 0x00, 0x00, 0x00]);
    let ar = read_archive_from(a).unwrap();
    assert!(ar.codebook_for(2, 8).is_ok());
    assert!(matches!(ar.codebook_for(3, 8), Err(Error::BadGeometry(_))));
    assert!(matches!(ar.codebook_for(4, 8), Err(Error::BadGeometry(_))));
    // And dequant with a too-short book is an Err, not an index panic.
    let Tensor::Cq(m) = &ar.tensors[0] else {
        panic!("expected CQ tensor");
    };
    assert!(matches!(dequant_cq(m, &ar.codebook), Err(Error::BadGeometry(_))));
    assert!(matches!(dequant_cq(m, &[]), Err(Error::BadGeometry(_))));
}

#[test]
fn dequant_rejects_bad_matrix_directly() {
    let mat = |group_size: usize, bits: u32| CqMatrix {
        out: 1,
        inp: 8,
        packed: vec![0u8; 4],
        norms: vec![half::f16::ONE; 1],
        group_size,
        bits,
    };
    let cb = needle_format::ternary_codebook(8);
    assert!(matches!(dequant_cq(&mat(8, 6), &cb), Err(Error::BadBits(6))));
    assert!(matches!(dequant_cq(&mat(0, 5), &cb), Err(Error::BadGeometry(_))));
    assert!(matches!(dequant_cq(&mat(96, 5), &cb), Err(Error::BadGeometry(_))));
}

// ---------- truncation at every byte boundary ----------

#[test]
fn truncation_at_every_length_errors() {
    // Tightly packed archive: header | codebook(28) | 3 records | blobs.
    let a = archive(
        28,
        &[
            (DTYPE_FP32, &[2, 2], 16, 0, 0),
            (DTYPE_CQ, &[1, 8], 4, 8, 5),
            (DTYPE_RAW, &[9], 3, 0, 0),
        ],
    );
    let total = a.len();
    assert!(read_archive_from(a.clone()).is_ok());
    for len in 0..total {
        assert!(
            read_archive_from(a[..len].to_vec()).is_err(),
            "truncated archive (len {len}/{total}) must be rejected"
        );
    }
}
