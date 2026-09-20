//! Regression tests: malformed tokenizer blobs and model-generated ids must
//! produce `Err` / lossy output, never a panic or an abort. All blobs here
//! are hand-built (no model download).
use needle_tokenizer::{Error, Tokenizer, TK_BYTE, TK_NORMAL, TK_USER_DEFINED};

/// `<IIIIIBBH` header + `<fBH` records, as written by `parse_tokenizer_blob`.
fn blob(pieces: &[(f32, u8, &str)], unk_id: u32, add_dummy: bool, byte_fb: bool) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&(pieces.len() as u32).to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes()); // pad
    b.extend_from_slice(&1u32.to_le_bytes()); // eos
    b.extend_from_slice(&2u32.to_le_bytes()); // bos
    b.extend_from_slice(&unk_id.to_le_bytes());
    b.push(add_dummy as u8);
    b.push(byte_fb as u8);
    b.extend_from_slice(&0u16.to_le_bytes());
    for (score, t, p) in pieces {
        b.extend_from_slice(&score.to_le_bytes());
        b.push(*t);
        b.extend_from_slice(&(p.len() as u16).to_le_bytes());
        b.extend_from_slice(p.as_bytes());
    }
    b
}

fn load(pieces: &[(f32, u8, &str)]) -> Tokenizer {
    Tokenizer::from_blob(&blob(pieces, 7, false, true)).unwrap()
}

#[test]
fn valid_blob_roundtrips() {
    let tok = load(&[
        (0.0, TK_NORMAL, "a"),
        (0.0, TK_NORMAL, "b"),
        (0.0, TK_NORMAL, "ab"),
        (0.0, TK_BYTE, "<0x41>"),
        (0.0, TK_USER_DEFINED, "<|im_start|>"),
    ]);
    assert_eq!(tok.vocab_size(), 5);
    assert_eq!(tok.piece(4), "<|im_start|>");
    assert_eq!(tok.piece_opt(4), Some("<|im_start|>"));
    assert_eq!(tok.piece_opt(99), None);
    // byte fallback: 'A' is not a piece, so it maps to the <0x41> BYTE piece
    assert_eq!(tok.encode("A"), vec![3]);
    assert_eq!(tok.decode(&[3]), "A");
}

#[test]
fn huge_n_pieces_errors_instead_of_aborting() {
    // n_pieces = 2^32-1 with no records: must hit the bounds check, not
    // pre-allocate ~200 GB of Vec capacity.
    let mut b = Vec::new();
    b.extend_from_slice(&u32::MAX.to_le_bytes());
    b.extend_from_slice(&[0u8; 20]);
    assert!(matches!(
        Tokenizer::from_blob(&b),
        Err(Error::Truncated)
    ));
}

#[test]
fn byte_piece_with_multibyte_surface_is_skipped_on_decode() {
    // "é" is a TK_BYTE piece with a 2-byte surface: `p[3..5]` would panic
    // (non-char-boundary). It must be skipped, not crash decode.
    let tok = load(&[(0.0, TK_BYTE, "é"), (0.0, TK_NORMAL, "z")]);
    assert_eq!(tok.decode(&[0]), "");
    assert_eq!(tok.decode(&[0, 1]), "z");
}

#[test]
fn byte_piece_with_short_surface_is_skipped_on_decode() {
    // "<0x4" (4 bytes): `&p[3..5]` would be out of range. Skipped.
    let tok = load(&[(0.0, TK_BYTE, "<0x4"), (0.0, TK_BYTE, "<0x42>")]);
    assert_eq!(tok.decode(&[0]), "");
    assert_eq!(tok.decode(&[0, 1]), "B");
}

#[test]
fn decode_skips_out_of_range_ids_lossily() {
    let tok = load(&[(0.0, TK_NORMAL, "hi")]);
    assert_eq!(tok.decode(&[0, 12345, u32::MAX]), "hi");
    assert_eq!(tok.decode(&[u32::MAX]), "");
}

#[test]
fn normal_piece_text_is_not_byte_fallback() {
    // Python keys byte_id strictly on type == TK_BYTE; a NORMAL piece that
    // merely looks like "<0x41>" must not capture byte-fallback traffic.
    let tok = load(&[(0.0, TK_NORMAL, "<0x41>"), (0.0, TK_NORMAL, "Z")]);
    // 'A' has no piece and no BYTE fallback entry -> unk id (7), not id 0.
    assert_eq!(tok.encode("A"), vec![7]);
}

#[test]
fn byte_fallback_still_works_for_real_byte_pieces() {
    let tok = load(&[(0.0, TK_BYTE, "<0x41>"), (0.0, TK_NORMAL, "Z")]);
    assert_eq!(tok.encode("A"), vec![0]);
    assert_eq!(tok.decode(&[0]), "A");
}

#[test]
fn truncated_blob_errors_at_record_boundary() {
    let full = blob(&[(1.0f32, TK_NORMAL, "hello"), (2.0, TK_BYTE, "<0x41>")], 7, false, true);
    // Every proper prefix must be Err (never panic).
    for len in 0..full.len() {
        assert!(
            Tokenizer::from_blob(&full[..len]).is_err(),
            "prefix of len {len} must be rejected"
        );
    }
}
