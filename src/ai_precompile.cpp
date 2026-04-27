// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// ai_precompile.cpp — CPU reference for the AI precompile service.
//
// This is the deterministic oracle. The Metal driver and any future CUDA /
// WGSL drivers MUST match these bytes for the same call. No tolerance.
//
// GEMM kernel:
//   acc(int32) = bias(int32);
//   for (k = 0..K-1) acc += int32(W[i][k]) * int32(X[k]);  // canonical row-major
//   acc >>= shift;
//   y_i = saturate_int8(acc);                               // clamp [-128, 127]
//
// Reduction order is fixed (0..K-1, no parallel-tree, no FMA, no fused-mul-add
// reordering). Saturation is the only nonlinearity. Same bytes everywhere.

#include "lux/aivm/ai_precompile.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace aivm::ai {

// =============================================================================
// keccak256 (matches aivm_cpu_reference.cpp byte-for-byte)
// =============================================================================

namespace {

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

inline uint64_t rotl64(uint64_t x, uint32_t n) {
    return (x << (n & 63u)) | (x >> ((64u - n) & 63u));
}

#if defined(__clang__)
__attribute__((optnone))
#elif defined(__GNUC__)
__attribute__((optimize("O0")))
#endif
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

inline void absorb_u32(uint8_t* dst, uint32_t off, uint32_t v) {
    for (uint32_t k = 0; k < 4u; ++k) dst[off + k] = uint8_t((v >> (k*8)) & 0xFFu);
}
inline void absorb_u64(uint8_t* dst, uint32_t off, uint64_t v) {
    for (uint32_t k = 0; k < 8u; ++k) dst[off + k] = uint8_t((v >> (k*8)) & 0xFFu);
}

inline Hash hash_bytes(const uint8_t* data, std::size_t len) {
    Hash h{};
    keccak256(data, len, h.data());
    return h;
}

inline bool hash_zero(const Hash& h) {
    for (auto b : h) if (b != 0) return false;
    return true;
}

inline int32_t saturate_int8(int32_t v) {
    if (v >  127) return  127;
    if (v < -128) return -128;
    return v;
}

}  // namespace

// =============================================================================
// AIPrecompileArtifact
// =============================================================================

void AIPrecompileArtifact::encode_preimage(uint8_t out[kPreimageBytes]) const noexcept
{
    uint32_t o = 0;
    std::memcpy(out + o, model_hash.data(),        32); o += 32;
    std::memcpy(out + o, model_config_hash.data(), 32); o += 32;
    std::memcpy(out + o, input_commitment.data(),  32); o += 32;
    std::memcpy(out + o, output_commitment.data(), 32); o += 32;
    std::memcpy(out + o, attestation_root.data(),  32); o += 32;
    std::memcpy(out + o, policy_root.data(),       32); o += 32;
    absorb_u32(out, o, batch_count); o += 4;
    absorb_u32(out, o, flags);       o += 4;
}

Hash AIPrecompileArtifact::root() const noexcept
{
    uint8_t buf[kPreimageBytes];
    encode_preimage(buf);
    return hash_bytes(buf, kPreimageBytes);
}

// =============================================================================
// Tiny classifier fixture (32 -> 16 -> 1)
// =============================================================================

namespace {

// Linear congruential generator — deterministic across all platforms. We
// avoid std::mt19937 specifically because libstdc++ and libc++ produce
// different sequences for the same seed.
struct Lcg {
    uint64_t state;
    explicit Lcg(uint64_t seed) : state(seed | 1ULL) {}
    uint32_t next() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return uint32_t(state >> 32);
    }
    int8_t next_i8(int8_t lo, int8_t hi) {
        uint32_t r = next();
        int32_t span = int32_t(hi) - int32_t(lo) + 1;
        return int8_t(int32_t(lo) + int32_t(r % uint32_t(span)));
    }
};

constexpr uint32_t kFixtureMagic   = 0x4D564941u; // "AIVM" little-endian
constexpr uint16_t kFixtureVersion = 1u;
constexpr std::size_t kHeaderBytes = 32;
constexpr std::size_t kBodyBytes   =
    /*W1*/ 32 * 16 +
    /*B1*/ 16 * 4 +
    /*W2*/ 16 * 1 +
    /*B2*/  1 * 4 +
    /*shift1+shift2*/ 2;
constexpr std::size_t kFixtureBytes = kHeaderBytes + kBodyBytes;

}  // namespace

TinyClassifier make_tiny_classifier_fixture()
{
    TinyClassifier m;
    Lcg rng(0xa1b2c3d4e5f60718ULL);
    for (auto& w : m.w1) w = rng.next_i8(-32, 32);
    for (auto& b : m.b1) b = int32_t(rng.next() % 256u) - 128;
    for (auto& w : m.w2) w = rng.next_i8(-32, 32);
    for (auto& b : m.b2) b = int32_t(rng.next() % 256u) - 128;
    m.shift1 = 5;
    m.shift2 = 5;

    auto bytes = tiny_classifier_to_bytes(m);
    m.model_hash         = hash_bytes(bytes.data(), bytes.size());
    // config hash binds dims + shifts only (small, stable across weight rotation).
    uint8_t cfg[12] = {};
    absorb_u32(cfg, 0, TinyClassifier::kInDim);
    absorb_u32(cfg, 4, TinyClassifier::kHidden);
    cfg[8]  = uint8_t(m.shift1);
    cfg[9]  = uint8_t(m.shift2);
    cfg[10] = uint8_t(TinyClassifier::kOutDim);
    cfg[11] = 0;
    m.model_config_hash = hash_bytes(cfg, sizeof(cfg));
    return m;
}

std::vector<uint8_t> tiny_classifier_to_bytes(const TinyClassifier& m)
{
    std::vector<uint8_t> out(kFixtureBytes, 0);
    uint32_t o = 0;
    absorb_u32(out.data(), o, kFixtureMagic);   o += 4;
    absorb_u32(out.data(), o, kFixtureVersion); o += 4;
    out[o++] = 2; out[o++] = 0;                              // layers = 2
    out[o++] = uint8_t(TinyClassifier::kInDim & 0xFFu);  out[o++] = 0;
    out[o++] = uint8_t(TinyClassifier::kHidden & 0xFFu); out[o++] = 0;
    out[o++] = uint8_t(TinyClassifier::kOutDim & 0xFFu); out[o++] = 0;
    o += 16;                                                  // pad
    // o should now be 32.
    std::memcpy(out.data() + o, m.w1.data(), m.w1.size()); o += uint32_t(m.w1.size());
    for (auto v : m.b1) { absorb_u32(out.data(), o, uint32_t(v)); o += 4; }
    std::memcpy(out.data() + o, m.w2.data(), m.w2.size()); o += uint32_t(m.w2.size());
    for (auto v : m.b2) { absorb_u32(out.data(), o, uint32_t(v)); o += 4; }
    out[o++] = uint8_t(m.shift1);
    out[o++] = uint8_t(m.shift2);
    return out;
}

TinyClassifier tiny_classifier_from_bytes(const std::vector<uint8_t>& bytes)
{
    TinyClassifier m;
    if (bytes.size() < kFixtureBytes) return m;
    auto r32 = [&](uint32_t off) {
        return uint32_t(bytes[off])
             | (uint32_t(bytes[off+1]) << 8)
             | (uint32_t(bytes[off+2]) << 16)
             | (uint32_t(bytes[off+3]) << 24);
    };
    if (r32(0) != kFixtureMagic) return m;
    uint32_t o = kHeaderBytes;
    std::memcpy(m.w1.data(), bytes.data() + o, m.w1.size()); o += uint32_t(m.w1.size());
    for (auto& v : m.b1) { v = int32_t(r32(o)); o += 4; }
    std::memcpy(m.w2.data(), bytes.data() + o, m.w2.size()); o += uint32_t(m.w2.size());
    for (auto& v : m.b2) { v = int32_t(r32(o)); o += 4; }
    m.shift1 = int8_t(bytes[o++]);
    m.shift2 = int8_t(bytes[o++]);

    m.model_hash = hash_bytes(bytes.data(), bytes.size());
    uint8_t cfg[12] = {};
    absorb_u32(cfg, 0, TinyClassifier::kInDim);
    absorb_u32(cfg, 4, TinyClassifier::kHidden);
    cfg[8]  = uint8_t(m.shift1);
    cfg[9]  = uint8_t(m.shift2);
    cfg[10] = uint8_t(TinyClassifier::kOutDim);
    m.model_config_hash = hash_bytes(cfg, sizeof(cfg));
    return m;
}

// =============================================================================
// Forward inference — the CPU reference. Output buffer must hold kOutDim bytes.
// =============================================================================

namespace {

void mlp_forward_int8(const TinyClassifier& m,
                      const int8_t* x,
                      int8_t* y_out)
{
    // Layer 1: 32 -> 16
    int8_t h[TinyClassifier::kHidden];
    for (uint32_t i = 0; i < TinyClassifier::kHidden; ++i) {
        int32_t acc = m.b1[i];
        for (uint32_t k = 0; k < TinyClassifier::kInDim; ++k) {
            // Canonical row-major: w1[i*K + k]
            acc += int32_t(m.w1[i * TinyClassifier::kInDim + k]) * int32_t(x[k]);
        }
        // Right-shift by shift1, then saturate. Sign-aware shift on int32.
        acc >>= m.shift1;
        h[i] = int8_t(saturate_int8(acc));
    }

    // Layer 2: 16 -> 1
    for (uint32_t i = 0; i < TinyClassifier::kOutDim; ++i) {
        int32_t acc = m.b2[i];
        for (uint32_t k = 0; k < TinyClassifier::kHidden; ++k) {
            acc += int32_t(m.w2[i * TinyClassifier::kHidden + k]) * int32_t(h[k]);
        }
        acc >>= m.shift2;
        y_out[i] = int8_t(saturate_int8(acc));
    }
}

}  // namespace

// =============================================================================
// AIPrecompile — Impl
// =============================================================================

struct AIPrecompile::Impl {
    std::mutex mu;
    // Cache of admitted models keyed by model_hash.
    struct ModelEntry {
        TinyClassifier classifier;
        std::vector<uint8_t> bytes;
    };
    std::unordered_map<std::string, ModelEntry> models;

    static std::string key_of(const Hash& h) {
        return std::string(reinterpret_cast<const char*>(h.data()), h.size());
    }

    bool find(const Hash& h, ModelEntry& out) {
        std::lock_guard<std::mutex> g(mu);
        auto it = models.find(key_of(h));
        if (it == models.end()) return false;
        out = it->second;
        return true;
    }

    void insert(const Hash& h, ModelEntry e) {
        std::lock_guard<std::mutex> g(mu);
        models[key_of(h)] = std::move(e);
    }
};

AIPrecompile::AIPrecompile() : impl_(std::make_unique<Impl>()) {}
AIPrecompile::~AIPrecompile() = default;

bool AIPrecompile::has_metal_engine() const noexcept { return false; }
bool AIPrecompile::enable_metal_engine() { return false; }

AIStatus AIPrecompile::model_verify(const std::vector<uint8_t>& model_bytes,
                                    const Hash& model_hash,
                                    const std::array<uint8_t, 64>& signature)
{
    // Hash the bytes and require a match against the claimed model_hash.
    Hash actual = hash_bytes(model_bytes.data(), model_bytes.size());
    if (actual != model_hash) return AIStatus::SignatureRejected;

    // Accept any non-zero signature (v0.1 stand-in for Ed25519 / BLS).
    bool any_nonzero = false;
    for (auto b : signature) if (b != 0) { any_nonzero = true; break; }
    if (!any_nonzero) return AIStatus::SignatureRejected;

    auto m = tiny_classifier_from_bytes(model_bytes);
    if (hash_zero(m.model_hash)) return AIStatus::UnknownModel;

    Impl::ModelEntry e{m, model_bytes};
    impl_->insert(model_hash, std::move(e));
    return AIStatus::Ok;
}

Hash AIPrecompile::inference_commit(const uint8_t salt[32],
                                    const uint8_t* input, std::size_t len)
{
    std::vector<uint8_t> buf(32 + len);
    std::memcpy(buf.data(),     salt,  32);
    std::memcpy(buf.data() + 32, input, len);
    return hash_bytes(buf.data(), buf.size());
}

Hash AIPrecompile::result_attest(const Hash& model_hash,
                                 const Hash& input_commitment,
                                 const Hash& output_commitment,
                                 const Hash& policy_hash,
                                 uint64_t timestamp_ns)
{
    uint8_t buf[32 + 32 + 32 + 32 + 8] = {};
    uint32_t o = 0;
    std::memcpy(buf + o, model_hash.data(),        32); o += 32;
    std::memcpy(buf + o, input_commitment.data(),  32); o += 32;
    std::memcpy(buf + o, output_commitment.data(), 32); o += 32;
    std::memcpy(buf + o, policy_hash.data(),       32); o += 32;
    absorb_u64(buf, o, timestamp_ns);                   o += 8;
    return hash_bytes(buf, sizeof(buf));
}

Hash AIPrecompile::audit_commit(const AIPrecompileArtifact& artifact)
{
    return artifact.root();
}

namespace {

// Saturating sign-aware right-shift used by the GEMM wrapper. Match the
// kernel exactly.
inline int32_t srshift(int32_t v, int8_t shift) { return v >> shift; }

// Mode 3 — sample-replay. K=10, seeded by round_id. We pick `count / K`
// indices via a deterministic LCG and run them again on the CPU reference.
// We deliberately use the CPU reference for the replay (oracle), so even
// if the primary path is the GPU driver, the verification compares its
// output against the oracle byte-for-byte.
bool sampled_replay_ok(const TinyClassifier& m,
                       const std::vector<int8_t>& inputs,
                       const std::vector<int8_t>& outputs,
                       uint64_t round_id,
                       uint32_t K = 10u)
{
    const uint32_t n = uint32_t(inputs.size() / TinyClassifier::kInDim);
    if (n == 0) return true;
    const uint32_t pick = std::max<uint32_t>(1u, n / K);
    Lcg rng(round_id ^ 0xfeedfacecafebabeULL);
    for (uint32_t i = 0; i < pick; ++i) {
        uint32_t idx = rng.next() % n;
        int8_t out_local[TinyClassifier::kOutDim];
        mlp_forward_int8(m, inputs.data() + idx * TinyClassifier::kInDim, out_local);
        for (uint32_t k = 0; k < TinyClassifier::kOutDim; ++k) {
            if (outputs[idx * TinyClassifier::kOutDim + k] != out_local[k]) return false;
        }
    }
    return true;
}

}  // namespace

AIInferenceResult AIPrecompile::inference_execute(const AIInferenceCall& call,
                                                  const uint8_t* input,
                                                  uint8_t* output)
{
    AIInferenceResult r{};
    r.status = AIStatus::Ok;

    Impl::ModelEntry e;
    if (!impl_->find(call.model_hash, e)) {
        r.status = AIStatus::UnknownModel;
        return r;
    }

    // Validate input — must be a multiple of kInDim.
    if (call.input_len == 0 || (call.input_len % TinyClassifier::kInDim) != 0) {
        r.status = AIStatus::InputTooLarge;
        return r;
    }
    const uint32_t n = call.input_len / TinyClassifier::kInDim;
    const uint32_t out_bytes = n * TinyClassifier::kOutDim;
    if (out_bytes > call.output_capacity) {
        r.status = AIStatus::OutputCapacity;
        return r;
    }

    // Snapshot inputs and run forward.
    std::vector<int8_t> in_snap(input, input + call.input_len);
    std::vector<int8_t> out_snap(out_bytes, 0);
    for (uint32_t i = 0; i < n; ++i) {
        mlp_forward_int8(e.classifier,
                         in_snap.data() + i * TinyClassifier::kInDim,
                         out_snap.data() + i * TinyClassifier::kOutDim);
    }

    // Commitments (over salt || raw bytes).
    r.input_commitment  = inference_commit(call.salt,
        reinterpret_cast<const uint8_t*>(in_snap.data()), in_snap.size());
    r.output_commitment = inference_commit(call.salt,
        reinterpret_cast<const uint8_t*>(out_snap.data()), out_snap.size());

    switch (call.mode) {
    case AIExecutionMode::DeterministicInteger: {
        // Host sees raw output.
        std::memcpy(output, out_snap.data(), out_bytes);
        break;
    }
    case AIExecutionMode::ConfidentialInference: {
        // Host MUST NOT see input or output. Zero the output buffer (the
        // input buffer is the caller's; we cannot zero it without lying
        // about ownership). The input_commitment is the only proof of
        // input the host gets.
        if (output && call.output_capacity > 0) {
            std::memset(output, 0, call.output_capacity);
        }
        r.attestation_root = result_attest(call.model_hash,
                                           r.input_commitment,
                                           r.output_commitment,
                                           call.policy_hash,
                                           call.timestamp_ns);
        break;
    }
    case AIExecutionMode::VerifiableInference: {
        if (!sampled_replay_ok(e.classifier, in_snap, out_snap, call.round_id)) {
            r.status = AIStatus::DivergenceDetected;
            return r;
        }
        std::memcpy(output, out_snap.data(), out_bytes);
        r.attestation_root = result_attest(call.model_hash,
                                           r.input_commitment,
                                           r.output_commitment,
                                           call.policy_hash,
                                           call.timestamp_ns);
        break;
    }
    }

    r.output_len = out_bytes;
    return r;
}

AIInferenceResult AIPrecompile::execute(const AIInferenceCall& call,
                                        const uint8_t* input,
                                        uint8_t* output,
                                        AIPrecompileArtifact& artifact_out)
{
    auto r = inference_execute(call, input, output);
    artifact_out.model_hash         = call.model_hash;
    artifact_out.input_commitment   = r.input_commitment;
    artifact_out.output_commitment  = r.output_commitment;
    artifact_out.attestation_root   = r.attestation_root;
    artifact_out.policy_root        = call.policy_hash;
    artifact_out.batch_count        = (call.input_len ? call.input_len / TinyClassifier::kInDim : 0);
    artifact_out.flags = 0;
    if (call.mode == AIExecutionMode::DeterministicInteger)  artifact_out.flags |= 0x1u;
    if (call.mode == AIExecutionMode::ConfidentialInference) artifact_out.flags |= 0x2u;
    if (call.mode == AIExecutionMode::VerifiableInference)
        artifact_out.flags |= 0x1u | 0x4u;  // Mode 3 implies deterministic + verified
    // Bind the model_config_hash if a model is admitted.
    Impl::ModelEntry e;
    if (impl_->find(call.model_hash, e)) {
        artifact_out.model_config_hash = e.classifier.model_config_hash;
    } else {
        artifact_out.model_config_hash = Hash{};
    }
    return r;
}

}  // namespace aivm::ai
