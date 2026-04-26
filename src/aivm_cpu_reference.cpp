// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file aivm_cpu_reference.cpp
/// AIVM CPU reference — deterministic oracle for cross-backend determinism.
///
/// Mirrors what aivm_*.metal / aivm_*.cu / aivm_*.wgsl must produce
/// byte-for-byte:
///   * AttestationApply  : verify quote signature (binding to attesting_key
///                         and tee_quote_digest), check expiry, register
///   * ProvenanceApply   : register/rotate model entry; bump version on
///                         weight rotation; update license commitments
///   * AnchorApply       : append commit anchor with parent-root chain
///   * EpochTransition   : prune expired attestations, compute roots,
///                         compose aivm_state_root
///
/// Determinism contract: identical input -> identical
///   (attestations[], models[], anchors[], epoch) -> identical roots.
///
/// **Note on signature verification.** The CPU reference treats the quote
/// signature as verified iff (a) the attesting_key is non-zero and
/// (b) the tee_quote_digest is non-zero. The four GPU backends apply the
/// same predicate so that cross-backend roots match byte-for-byte without
/// pulling the BLS / Ringtail / Groth16 verifier core into AIVM. The
/// production deployment will link against
/// quasar_bls_verifier / quasar_ringtail_verifier / quasar_groth16_verifier
/// before kernel launch (host-side) and zero out the attesting_key field
/// on any op that fails verification — keeping the on-GPU predicate
/// trivially deterministic. This matches the design used in cevm/quasar.

#include "lux/aivm/aivm_cpu_reference.hpp"

#include <algorithm>
#include <array>
#include <cstring>

namespace aivm::gpu::ref {

namespace {

// =============================================================================
// keccak256 (matches the implementation in cevm/quasar and pvm — bit-identical)
// =============================================================================

constexpr std::array<uint64_t, 24> kKeccakRC = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL,
};

constexpr std::array<uint32_t, 25> kKeccakRot = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14,
};

// rotl64 is called with n=0 in the rho-pi step (kKeccakRot[0] == 0). A
// naive `x >> (64 - n)` is undefined behavior at n=0 (shift width equals
// type width), and clang -O2 happily exploits the UB and produces garbage.
// Mask the shift count to avoid UB; the n==0 case becomes `x | x == x`,
// which is the intended identity.
inline uint64_t rotl64(uint64_t x, uint32_t n) {
    return (x << (n & 63u)) | (x >> ((64u - n) & 63u));
}

void keccak_f1600(uint64_t* s) {
    for (uint32_t round = 0; round < 24u; ++round) {
        uint64_t c[5];
        for (uint32_t x = 0; x < 5u; ++x)
            c[x] = s[x] ^ s[x+5] ^ s[x+10] ^ s[x+15] ^ s[x+20];
        uint64_t d[5];
        for (uint32_t x = 0; x < 5u; ++x)
            d[x] = c[(x + 4u) % 5u] ^ rotl64(c[(x + 1u) % 5u], 1);
        for (uint32_t y = 0; y < 25u; y += 5u)
            for (uint32_t x = 0; x < 5u; ++x)
                s[y + x] ^= d[x];
        uint64_t b[25];
        for (uint32_t y = 0; y < 5u; ++y)
            for (uint32_t x = 0; x < 5u; ++x) {
                uint32_t i = x + 5u * y;
                uint32_t j = y + 5u * ((2u * x + 3u * y) % 5u);
                b[j] = rotl64(s[i], kKeccakRot[i]);
            }
        for (uint32_t y = 0; y < 25u; y += 5u) {
            uint64_t t0 = b[y+0], t1 = b[y+1], t2 = b[y+2], t3 = b[y+3], t4 = b[y+4];
            s[y+0] = t0 ^ ((~t1) & t2);
            s[y+1] = t1 ^ ((~t2) & t3);
            s[y+2] = t2 ^ ((~t3) & t4);
            s[y+3] = t3 ^ ((~t4) & t0);
            s[y+4] = t4 ^ ((~t0) & t1);
        }
        s[0] ^= kKeccakRC[round];
    }
}

void keccak256(const uint8_t* data, uint64_t len, uint8_t* out) {
    uint64_t s[25] = {};
    constexpr uint32_t rate = 136;
    uint64_t off = 0;
    while (len - off >= rate) {
        for (uint32_t i = 0; i < rate; ++i) {
            uint32_t lane = i / 8u, sh = (i % 8u) * 8u;
            s[lane] ^= uint64_t(data[off + i]) << sh;
        }
        keccak_f1600(s);
        off += rate;
    }
    uint8_t block[rate] = {};
    uint64_t rem = len - off;
    for (uint64_t i = 0; i < rem; ++i) block[i] = data[off + i];
    block[rem]      ^= 0x01;
    block[rate - 1] ^= 0x80;
    for (uint32_t i = 0; i < rate; ++i) {
        uint32_t lane = i / 8u, sh = (i % 8u) * 8u;
        s[lane] ^= uint64_t(block[i]) << sh;
    }
    keccak_f1600(s);
    for (uint32_t i = 0; i < 32u; ++i) {
        uint32_t lane = i / 8u, sh = (i % 8u) * 8u;
        out[i] = uint8_t((s[lane] >> sh) & 0xFFu);
    }
}

void absorb_u32(uint8_t* dst, uint32_t off, uint32_t v) {
    for (uint32_t k = 0; k < 4u; ++k) dst[off + k] = uint8_t((v >> (k*8)) & 0xFFu);
}
void absorb_u64(uint8_t* dst, uint32_t off, uint64_t v) {
    for (uint32_t k = 0; k < 8u; ++k) dst[off + k] = uint8_t((v >> (k*8)) & 0xFFu);
}

inline uint64_t saturating_add(uint64_t a, uint64_t b) {
    uint64_t r = a + b;
    return (r < a) ? UINT64_MAX : r;
}

// =============================================================================
// First-8-byte-LE → uint64 (used as hash table key)
// =============================================================================

inline uint64_t key_from_digest(const uint8_t d[32]) {
    uint64_t k = 0;
    for (uint32_t i = 0; i < 8u; ++i) k |= uint64_t(d[i]) << (i*8u);
    return k;
}

inline bool digest_equal(const uint8_t a[32], const uint8_t b[32]) {
    for (uint32_t i = 0; i < 32u; ++i) if (a[i] != b[i]) return false;
    return true;
}

inline bool digest_zero(const uint8_t a[32]) {
    for (uint32_t i = 0; i < 32u; ++i) if (a[i] != 0) return false;
    return true;
}

inline bool key48_zero(const uint8_t a[48]) {
    for (uint32_t i = 0; i < 48u; ++i) if (a[i] != 0) return false;
    return true;
}

// Open-addressing helpers — match the GPU kernels byte-for-byte.

inline uint32_t hash_index(uint64_t k, uint32_t mask) {
    uint64_t h = 0xcbf29ce484222325ULL;
    h = (h ^ k) * 0x100000001b3ULL;
    return uint32_t(h) & mask;
}

uint32_t attestation_locate(std::vector<Attestation>& tab,
                            const uint8_t digest[32],
                            bool insert_if_missing)
{
    uint32_t mask = uint32_t(tab.size()) - 1u;
    uint64_t key  = key_from_digest(digest);
    uint32_t idx  = hash_index(key, mask);
    for (uint32_t probe = 0; probe < tab.size(); ++probe) {
        auto& s = tab[idx];
        if (s.occupied == 0u) {
            if (insert_if_missing) {
                std::memset(&s, 0, sizeof(s));
                std::memcpy(s.tee_quote_digest, digest, 32);
                s.occupied = 1u;
                return idx;
            }
            return 0xFFFFFFFFu;
        }
        if (digest_equal(s.tee_quote_digest, digest)) return idx;
        idx = (idx + 1u) & mask;
    }
    return 0xFFFFFFFFu;
}

uint32_t model_locate(std::vector<ModelRegistryEntry>& tab,
                      const uint8_t model_root[32],
                      bool insert_if_missing)
{
    uint32_t mask = uint32_t(tab.size()) - 1u;
    uint64_t key  = key_from_digest(model_root);
    uint32_t idx  = hash_index(key, mask);
    for (uint32_t probe = 0; probe < tab.size(); ++probe) {
        auto& s = tab[idx];
        if (s.occupied == 0u) {
            if (insert_if_missing) {
                std::memset(&s, 0, sizeof(s));
                std::memcpy(s.model_root, model_root, 32);
                s.occupied = 1u;
                return idx;
            }
            return 0xFFFFFFFFu;
        }
        if (digest_equal(s.model_root, model_root)) return idx;
        idx = (idx + 1u) & mask;
    }
    return 0xFFFFFFFFu;
}

// =============================================================================
// Attestation status bits
// =============================================================================

constexpr uint32_t kAttStatusVerified = 0x2u;
constexpr uint32_t kAttStatusExpired  = 0x4u;

// =============================================================================
// Kernel 1: AttestationApply
// =============================================================================
//
// For each op:
//   * reject if attesting_key is zero (would be a real CPU verifier reject)
//   * reject if tee_quote_digest is zero
//   * insert/update the attestation entry; mark verified
//   * mark expired immediately if expiry_ns != 0 and expiry_ns <= timestamp_ns

uint32_t apply_attestation_ops(AIVMReferenceState& state,
                               const AIVMRoundDescriptor& desc,
                               std::span<const AttestationOp> ops)
{
    uint32_t applied = 0;
    for (const auto& op : ops) {
        if (key48_zero(op.attesting_key)) continue;
        if (digest_zero(op.tee_quote_digest)) continue;

        uint32_t idx = attestation_locate(state.attestations, op.tee_quote_digest, true);
        if (idx == 0xFFFFFFFFu) continue;

        auto& s = state.attestations[idx];
        std::memcpy(s.measurement,   op.measurement,   32);
        std::memcpy(s.attesting_key, op.attesting_key, 48);
        s.expiry_ns       = op.expiry_ns;
        s.kind            = op.kind;
        s.evidence_offset = op.evidence_offset;
        s.evidence_len    = op.evidence_len;
        s.status          = kAttStatusVerified;
        if (op.expiry_ns != 0u && op.expiry_ns <= desc.timestamp_ns) {
            s.status |= kAttStatusExpired;
        }
        ++applied;
    }
    return applied;
}

// =============================================================================
// Kernel 2: ProvenanceApply
// =============================================================================

uint32_t apply_model_ops(AIVMReferenceState& state,
                         std::span<const ModelOp> ops)
{
    uint32_t applied = 0;
    for (const auto& op : ops) {
        if (digest_zero(op.model_root)) continue;
        if (digest_zero(op.weight_hash)) continue;

        switch (static_cast<ModelOpKind>(op.kind)) {
            case ModelOpKind::Register: {
                uint32_t idx = model_locate(state.models, op.model_root, true);
                if (idx == 0xFFFFFFFFu) break;
                auto& s = state.models[idx];
                std::memcpy(s.weight_hash, op.weight_hash, 32);
                std::memcpy(s.license_root, op.license_root, 32);
                std::memcpy(s.owner_addr, op.owner_addr, 20);
                s.parameter_count = op.parameter_count;
                s.modality        = op.modality;
                s.version         = 1u;
                ++applied;
                break;
            }
            case ModelOpKind::UpdateWeights: {
                uint32_t idx = model_locate(state.models, op.model_root, false);
                if (idx == 0xFFFFFFFFu) break;
                auto& s = state.models[idx];
                std::memcpy(s.weight_hash, op.weight_hash, 32);
                s.version = saturating_add(s.version, 1u);
                if (op.parameter_count != 0u) s.parameter_count = op.parameter_count;
                ++applied;
                break;
            }
            case ModelOpKind::UpdateLicense: {
                uint32_t idx = model_locate(state.models, op.model_root, false);
                if (idx == 0xFFFFFFFFu) break;
                auto& s = state.models[idx];
                std::memcpy(s.license_root, op.license_root, 32);
                ++applied;
                break;
            }
            case ModelOpKind::Transfer: {
                uint32_t idx = model_locate(state.models, op.model_root, false);
                if (idx == 0xFFFFFFFFu) break;
                auto& s = state.models[idx];
                std::memcpy(s.owner_addr, op.owner_addr, 20);
                ++applied;
                break;
            }
        }
    }
    return applied;
}

// =============================================================================
// Kernel 3: AnchorApply
// =============================================================================
//
// Append-only ring buffer. A new anchor is accepted iff:
//   * commit_root is non-zero, AND
//   * if there is a previous occupied slot, parent_root must equal the
//     previous slot's commit_root (chain integrity), AND
//   * height > previous height (monotonic)
// The kernel scans for the head of the ring (last occupied slot < cursor)
// in the same canonical order as the GPU kernels.

uint32_t apply_anchor_ops(AIVMReferenceState& state,
                          std::span<const AnchorOp> ops,
                          uint32_t& cursor_out)
{
    uint32_t applied = 0;
    // Find next free slot (cursor) by scanning linearly. Anchors are
    // append-only so the first non-occupied slot is the cursor.
    uint32_t cursor = 0;
    while (cursor < state.anchors.size() && state.anchors[cursor].occupied != 0u) ++cursor;

    for (const auto& op : ops) {
        if (digest_zero(op.commit_root)) continue;
        if (cursor >= state.anchors.size()) break;

        // Chain integrity: if there is a head, parent_root must match it.
        // Heights must be strictly monotonic.
        if (cursor > 0u) {
            const auto& prev = state.anchors[cursor - 1u];
            if (!digest_equal(op.parent_root, prev.commit_root)) continue;
            if (op.height <= prev.height) continue;
        }

        auto& dst = state.anchors[cursor];
        std::memcpy(dst.commit_root, op.commit_root, 32);
        std::memcpy(dst.parent_root, op.parent_root, 32);
        std::memcpy(dst.validator_set_root_at_commit,
                    op.validator_set_root_at_commit, 32);
        dst.height       = op.height;
        dst.timestamp_ns = op.timestamp_ns;
        dst.occupied     = 1u;
        ++cursor;
        ++applied;
    }
    cursor_out = cursor;
    return applied;
}

// =============================================================================
// Kernel 4: EpochTransition (root computation + expiry pruning)
// =============================================================================

void compute_attestation_root(const std::vector<Attestation>& atts,
                              uint64_t timestamp_ns,
                              uint8_t out[32],
                              uint32_t& active_out,
                              uint32_t& expired_out)
{
    std::array<uint8_t, 32> acc{};
    active_out = expired_out = 0u;
    for (uint32_t i = 0; i < atts.size(); ++i) {
        const auto& a = atts[i];
        if (a.occupied == 0u) continue;

        bool expired = (a.expiry_ns != 0u && a.expiry_ns <= timestamp_ns)
                    || (a.status & kAttStatusExpired) != 0u;
        bool verified = (a.status & kAttStatusVerified) != 0u;
        if (expired) ++expired_out;
        else if (verified) ++active_out;

        // leaf = keccak( tee_quote_digest || measurement || attesting_key
        //               || expiry_ns || kind || evidence_offset || evidence_len
        //               || status || expired_flag || index )
        uint8_t leaf[32 + 32 + 48 + 8 + 4 + 4 + 4 + 4 + 4 + 4] = {};
        uint32_t o = 0;
        std::memcpy(leaf + o, a.tee_quote_digest, 32); o += 32;
        std::memcpy(leaf + o, a.measurement,      32); o += 32;
        std::memcpy(leaf + o, a.attesting_key,    48); o += 48;
        absorb_u64(leaf, o, a.expiry_ns);       o += 8;
        absorb_u32(leaf, o, a.kind);            o += 4;
        absorb_u32(leaf, o, a.evidence_offset); o += 4;
        absorb_u32(leaf, o, a.evidence_len);    o += 4;
        absorb_u32(leaf, o, a.status);          o += 4;
        absorb_u32(leaf, o, expired ? 1u : 0u); o += 4;
        absorb_u32(leaf, o, i);                 o += 4;

        uint8_t leaf_hash[32];
        keccak256(leaf, o, leaf_hash);

        uint8_t buf[64];
        std::memcpy(buf, acc.data(), 32);
        std::memcpy(buf + 32, leaf_hash, 32);
        keccak256(buf, 64, acc.data());
    }
    std::memcpy(out, acc.data(), 32);
}

void compute_model_registry_root(const std::vector<ModelRegistryEntry>& models,
                                 uint8_t out[32],
                                 uint32_t& count_out)
{
    std::array<uint8_t, 32> acc{};
    count_out = 0u;
    for (uint32_t i = 0; i < models.size(); ++i) {
        const auto& m = models[i];
        if (m.occupied == 0u) continue;
        ++count_out;

        uint8_t leaf[32 + 32 + 32 + 20 + 8 + 8 + 4 + 4] = {};
        uint32_t o = 0;
        std::memcpy(leaf + o, m.model_root,   32); o += 32;
        std::memcpy(leaf + o, m.weight_hash,  32); o += 32;
        std::memcpy(leaf + o, m.license_root, 32); o += 32;
        std::memcpy(leaf + o, m.owner_addr,   20); o += 20;
        absorb_u64(leaf, o, m.version);         o += 8;
        absorb_u64(leaf, o, m.parameter_count); o += 8;
        absorb_u32(leaf, o, m.modality);        o += 4;
        absorb_u32(leaf, o, i);                 o += 4;

        uint8_t leaf_hash[32];
        keccak256(leaf, o, leaf_hash);
        uint8_t buf[64];
        std::memcpy(buf, acc.data(), 32);
        std::memcpy(buf + 32, leaf_hash, 32);
        keccak256(buf, 64, acc.data());
    }
    std::memcpy(out, acc.data(), 32);
}

void compute_audit_root(const std::vector<AuditAnchor>& anchors,
                        uint8_t out[32],
                        uint32_t& count_out)
{
    std::array<uint8_t, 32> acc{};
    count_out = 0u;
    for (uint32_t i = 0; i < anchors.size(); ++i) {
        const auto& a = anchors[i];
        if (a.occupied == 0u) continue;
        ++count_out;

        uint8_t leaf[32 + 32 + 32 + 8 + 8 + 4] = {};
        uint32_t o = 0;
        std::memcpy(leaf + o, a.commit_root, 32); o += 32;
        std::memcpy(leaf + o, a.parent_root, 32); o += 32;
        std::memcpy(leaf + o, a.validator_set_root_at_commit, 32); o += 32;
        absorb_u64(leaf, o, a.height);       o += 8;
        absorb_u64(leaf, o, a.timestamp_ns); o += 8;
        absorb_u32(leaf, o, i);              o += 4;

        uint8_t leaf_hash[32];
        keccak256(leaf, o, leaf_hash);
        uint8_t buf[64];
        std::memcpy(buf, acc.data(), 32);
        std::memcpy(buf + 32, leaf_hash, 32);
        keccak256(buf, 64, acc.data());
    }
    std::memcpy(out, acc.data(), 32);
}

void close_epoch(AIVMReferenceState& state,
                 const AIVMRoundDescriptor& desc,
                 AIVMTransitionResult& r)
{
    // Mark expired attestations against the round timestamp first so the
    // root reflects the post-epoch state.
    for (auto& a : state.attestations) {
        if (a.occupied == 0u) continue;
        if (a.expiry_ns != 0u && a.expiry_ns <= desc.timestamp_ns) {
            a.status |= kAttStatusExpired;
        }
    }

    uint32_t active = 0, expired = 0, model_count = 0, anchor_count = 0;
    compute_attestation_root(state.attestations, desc.timestamp_ns,
                             state.epoch.attestation_root, active, expired);
    compute_model_registry_root(state.models, state.epoch.model_registry_root,
                                model_count);
    compute_audit_root(state.anchors, state.epoch.audit_root, anchor_count);

    state.epoch.active_model_count        = model_count;
    state.epoch.expired_attestation_count = expired;
    state.epoch.total_active_attestations = active;

    uint64_t target_epoch = desc.closing_flag != 0u ? desc.epoch + 1u : desc.epoch;
    if (desc.closing_flag != 0u) state.epoch.current_epoch = target_epoch;

    // aivm_state_root = keccak(parent || attestation_root || model_root
    //                          || audit_root || epoch || active || model_count
    //                          || anchor_count)
    uint8_t composed[32 + 32 + 32 + 32 + 8 + 4 + 4 + 4] = {};
    uint32_t o = 0;
    std::memcpy(composed + o, desc.parent_aivm_root, 32);            o += 32;
    std::memcpy(composed + o, state.epoch.attestation_root, 32);     o += 32;
    std::memcpy(composed + o, state.epoch.model_registry_root, 32);  o += 32;
    std::memcpy(composed + o, state.epoch.audit_root, 32);           o += 32;
    absorb_u64(composed, o, state.epoch.current_epoch); o += 8;
    absorb_u32(composed, o, active);                    o += 4;
    absorb_u32(composed, o, model_count);               o += 4;
    absorb_u32(composed, o, anchor_count);              o += 4;
    keccak256(composed, o, state.epoch.aivm_state_root);

    std::memcpy(r.attestation_root,    state.epoch.attestation_root,    32);
    std::memcpy(r.model_registry_root, state.epoch.model_registry_root, 32);
    std::memcpy(r.audit_root,          state.epoch.audit_root,          32);
    std::memcpy(r.aivm_state_root,     state.epoch.aivm_state_root,     32);
    r.active_attestations  = active;
    r.expired_attestations = expired;
    r.model_count          = model_count;
    r.anchor_count         = anchor_count;
    r.epoch                = state.epoch.current_epoch;
    r.total_models         = uint64_t(model_count);
    r.total_anchors        = uint64_t(anchor_count);
}

}  // anonymous namespace

AIVMReferenceState AIVMReferenceState::empty() {
    AIVMReferenceState s;
    s.attestations.assign(kDefaultAttestationSlots, Attestation{});
    s.models.assign(kDefaultModelSlots, ModelRegistryEntry{});
    s.anchors.assign(kDefaultAnchorSlots, AuditAnchor{});
    s.epoch = AIVMEpochState{};
    return s;
}

AIVMTransitionResult run_reference(AIVMReferenceState& state,
                                   const AIVMRoundDescriptor& desc,
                                   std::span<const AttestationOp> attestation_ops,
                                   std::span<const ModelOp>       model_ops,
                                   std::span<const AnchorOp>      anchor_ops)
{
    if (state.attestations.empty())
        state.attestations.assign(kDefaultAttestationSlots, Attestation{});
    if (state.models.empty())
        state.models.assign(kDefaultModelSlots, ModelRegistryEntry{});
    if (state.anchors.empty())
        state.anchors.assign(kDefaultAnchorSlots, AuditAnchor{});

    AIVMTransitionResult r{};
    auto mode = static_cast<AIVMTransitionMode>(desc.mode);

    if (mode == AIVMTransitionMode::AttestationApply ||
        mode == AIVMTransitionMode::FullRound) {
        r.attestation_apply_count = apply_attestation_ops(state, desc, attestation_ops);
    }
    if (mode == AIVMTransitionMode::ProvenanceApply ||
        mode == AIVMTransitionMode::FullRound) {
        r.model_apply_count = apply_model_ops(state, model_ops);
    }
    if (mode == AIVMTransitionMode::AnchorApply ||
        mode == AIVMTransitionMode::FullRound) {
        uint32_t cursor = 0;
        r.anchor_apply_count = apply_anchor_ops(state, anchor_ops, cursor);
    }
    // EpochTransition runs unconditionally so caller always gets fresh roots.
    close_epoch(state, desc, r);
    r.status = 1u;
    return r;
}

}  // namespace aivm::gpu::ref
