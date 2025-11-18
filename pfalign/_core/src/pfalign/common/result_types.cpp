#include "pfalign/common/result_types.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "pfalign/common/arena_allocator.h"
#include "pfalign/adapters/alignment_types.h"
#include "pfalign/dispatch/backend_traits.h"
#include "pfalign/modules/pairwise/pairwise_align.h"
#include "pfalign/primitives/kabsch/kabsch_impl.h"
#include "pfalign/primitives/structural_metrics/structural_metrics_impl.h"
#include "pfalign/tools/weights/save_npy.h"

namespace pfalign {
namespace {

using pfalign::AlignmentPair;
using pfalign::pairwise::AlignmentResult;

std::vector<AlignmentPair> build_alignment_path(const std::vector<std::pair<int, int>>& alignment,
                                                int L1, int L2, const float* posteriors) {
    std::vector<AlignmentPair> path;
    path.reserve(alignment.size());

    for (const auto& pair : alignment) {
        AlignmentPair entry;
        entry.i = pair.first;
        entry.j = pair.second;
        if (entry.i >= 0 && entry.i < L1 && entry.j >= 0 && entry.j < L2 && posteriors) {
            entry.posterior = posteriors[entry.i * L2 + entry.j];
        } else {
            entry.posterior = 0.0f;
        }
        path.push_back(entry);
    }

    return path;
}

AlignmentResult build_alignment_result(const PairwiseResult& result, AlignmentPair* path,
                                       int path_length, const float* coords1, int L1_coords,
                                       const float* coords2, int L2_coords, const std::string& id1,
                                       const std::string& id2) {
    AlignmentResult legacy;
    legacy.L1 = result.L1();
    legacy.L2 = result.L2();
    legacy.partition = result.partition();
    legacy.score = result.score();
    legacy.posteriors = const_cast<float*>(result.posteriors());
    legacy.alignment_path = path;
    legacy.path_length = path_length;
    legacy.max_path_length = path_length;
    legacy.coords1 = coords1;
    legacy.coords2 = coords2;
    legacy.id1 = id1;
    legacy.id2 = id2;
    legacy.protein1 = nullptr;
    legacy.protein2 = nullptr;
    (void)L1_coords;
    (void)L2_coords;
    return legacy;
}

int count_matches(const std::vector<std::pair<int, int>>& alignment) {
    return static_cast<int>(
        std::count_if(alignment.begin(), alignment.end(),
                      [](const std::pair<int, int>& p) { return p.first >= 0 && p.second >= 0; }));
}

void gather_ca_coords(const float* coords, int length, const std::vector<int>& residues,
                      std::vector<float>* out) {
    out->clear();
    out->reserve(residues.size() * 3);
    for (int idx : residues) {
        if (idx < 0 || idx >= length) {
            out->insert(out->end(), {0.0f, 0.0f, 0.0f});
            continue;
        }
        const float* atom = coords + idx * 3;
        out->push_back(atom[0]);
        out->push_back(atom[1]);
        out->push_back(atom[2]);
    }
}

float compute_rmsd(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) {
        return 0.0f;
    }
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        sum += diff * diff;
    }
    return static_cast<float>(std::sqrt(sum / (a.size() / 3)));
}

[[maybe_unused]] std::vector<std::string> amino_alphabet() {
    return {"-", "A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M",
            "F", "P", "S", "T", "W", "Y", "V", "B", "Z", "X", "O", "U", "J"};
}

int residue_to_index(char aa) {
    static const std::string alphabet = "-ARNDCEQGHILKMFPSTWYVBZXOUJ";
    size_t pos = alphabet.find(static_cast<char>(std::toupper(static_cast<unsigned char>(aa))));
    if (pos == std::string::npos) {
        return static_cast<int>(alphabet.find('X'));
    }
    return static_cast<int>(pos);
}

}  // namespace

PairwiseResult::PairwiseResult(float* posteriors, int L1, int L2, float score, float partition,
                               std::vector<std::pair<int, int>> alignment)
    : posteriors_(posteriors),
      L1_(L1),
      L2_(L2),
      score_(score),
      partition_(partition),
      alignment_(std::move(alignment)) {
    if (L1_ <= 0 || L2_ <= 0) {
        throw std::invalid_argument("PairwiseResult dimensions must be positive");
    }
    if (!posteriors_) {
        throw std::invalid_argument("PairwiseResult requires a valid posterior buffer");
    }
}

void PairwiseResult::write_fasta(const std::string& path, const std::string& seq1,
                                 const std::string& seq2, const std::string& id1,
                                 const std::string& id2) const {
    if (alignment_.empty()) {
        throw std::runtime_error("Alignment path is empty");
    }

    auto legacy_path = build_alignment_path(alignment_, L1_, L2_, posteriors_.get());
    auto legacy =
        build_alignment_result(*this, legacy_path.data(), static_cast<int>(legacy_path.size()),
                               nullptr, 0, nullptr, 0, id1, id2);

    if (!legacy.write_fasta(path, seq1, seq2)) {
        throw std::runtime_error("Failed to write FASTA to " + path);
    }
}

void PairwiseResult::write_npy(const std::string& path) const {
    save_npy_2d(path, posteriors_.get(), L1_, L2_);
}

void PairwiseResult::write_pdb(const std::string& path, const float* coords1, int coords1_length,
                               const float* coords2, int coords2_length, int reference) const {
    if (!coords1 || !coords2) {
        throw std::invalid_argument("write_pdb requires coordinate arrays for both structures");
    }
    (void)coords1_length;
    (void)coords2_length;

    auto legacy_path = build_alignment_path(alignment_, L1_, L2_, posteriors_.get());
    auto legacy =
        build_alignment_result(*this, legacy_path.data(), static_cast<int>(legacy_path.size()),
                               coords1, coords1_length, coords2, coords2_length, "", "");
    pfalign::memory::GrowableArena arena(32);  // 32 MB
    if (!legacy.write_superposed_pdb(path, reference, &arena)) {
        throw std::runtime_error("Failed to write superposed PDB to " + path);
    }
}

std::vector<std::pair<int, int>> PairwiseResult::get_aligned_residues() const {
    return alignment_;
}

float PairwiseResult::compute_coverage() const {
    if (alignment_.empty()) {
        return 0.0f;
    }
    const int matches = count_matches(alignment_);
    const int denom = std::max(L1_, L2_);
    if (denom == 0) {
        return 0.0f;
    }
    return static_cast<float>(matches) / static_cast<float>(denom);
}

PairwiseResult PairwiseResult::threshold(float cutoff) const {
    const size_t size = static_cast<size_t>(L1_) * static_cast<size_t>(L2_);
    auto filtered = std::make_unique<float[]>(size);
    std::vector<std::pair<int, int>> filtered_alignment;
    filtered_alignment.reserve(alignment_.size());

    for (int i = 0; i < L1_; ++i) {
        for (int j = 0; j < L2_; ++j) {
            const float value = posteriors_.get()[i * L2_ + j];
            filtered[i * L2_ + j] = (value >= cutoff) ? value : 0.0f;
        }
    }

    for (const auto& pair : alignment_) {
        if (pair.first >= 0 && pair.second >= 0) {
            const float value = posteriors_.get()[pair.first * L2_ + pair.second];
            if (value >= cutoff) {
                filtered_alignment.push_back(pair);
            }
        } else {
            filtered_alignment.push_back(pair);
        }
    }

    return PairwiseResult(filtered.release(), L1_, L2_, score_, partition_,
                          std::move(filtered_alignment));
}

void PairwiseResult::get_aligned_coords(const float* coords1, const float* coords2,
                                        float* out_coords1, float* out_coords2,
                                        int reference) const {
    if (!coords1 || !coords2 || !out_coords1 || !out_coords2) {
        throw std::invalid_argument("get_aligned_coords requires valid coordinate buffers");
    }

    std::vector<int> ref_indices;
    std::vector<int> mob_indices;
    for (const auto& pair : alignment_) {
        if (pair.first < 0 || pair.second < 0) {
            continue;
        }
        if (reference == 0) {
            ref_indices.push_back(pair.first);
            mob_indices.push_back(pair.second);
        } else {
            ref_indices.push_back(pair.second);
            mob_indices.push_back(pair.first);
        }
    }

    if (ref_indices.size() < 3) {
        throw std::runtime_error("At least three aligned residues are required for superposition");
    }

    std::vector<float> ref_coords;
    std::vector<float> mob_coords;
    gather_ca_coords(reference == 0 ? coords1 : coords2, reference == 0 ? L1_ : L2_, ref_indices,
                     &ref_coords);
    gather_ca_coords(reference == 0 ? coords2 : coords1, reference == 0 ? L2_ : L1_, mob_indices,
                     &mob_coords);

    float R[9];
    float t[3];
    float rmsd = 0.0f;
    kabsch::kabsch_align<ScalarBackend>(mob_coords.data(), ref_coords.data(),
                                        static_cast<int>(ref_indices.size()), R, t, &rmsd);

    std::vector<float> mob_aligned(mob_coords.size());
    for (size_t i = 0; i < mob_indices.size(); ++i) {
        const float* src = mob_coords.data() + i * 3;
        float* dst = mob_aligned.data() + i * 3;
        dst[0] = R[0] * src[0] + R[1] * src[1] + R[2] * src[2] + t[0];
        dst[1] = R[3] * src[0] + R[4] * src[1] + R[5] * src[2] + t[1];
        dst[2] = R[6] * src[0] + R[7] * src[1] + R[8] * src[2] + t[2];
    }

    const size_t bytes = ref_coords.size() * sizeof(float);
    if (reference == 0) {
        std::memcpy(out_coords1, ref_coords.data(), bytes);
        std::memcpy(out_coords2, mob_aligned.data(), bytes);
    } else {
        std::memcpy(out_coords1, mob_aligned.data(), bytes);
        std::memcpy(out_coords2, ref_coords.data(), bytes);
    }
}

float PairwiseResult::compute_rmsd(const float* coords1, const float* coords2) const {
    const int matches = count_matches(alignment_);
    if (matches < 3) {
        return 0.0f;
    }

    std::vector<float> aligned_ref(matches * 3);
    std::vector<float> aligned_mob(matches * 3);
    get_aligned_coords(coords1, coords2, aligned_ref.data(), aligned_mob.data(), 0);
    return pfalign::compute_rmsd(aligned_ref, aligned_mob);
}

float PairwiseResult::compute_tm_score(const float* coords1, const float* coords2) const {
    const int matches = count_matches(alignment_);
    if (matches < 3) {
        return 0.0f;
    }

    std::vector<float> aligned_ref(matches * 3);
    std::vector<float> aligned_mob(matches * 3);
    get_aligned_coords(coords1, coords2, aligned_ref.data(), aligned_mob.data(), 0);

    const float tm_ref = structural_metrics::compute_tm_score<ScalarBackend>(
        aligned_mob.data(), aligned_ref.data(), matches, L1_);
    const float tm_mob = structural_metrics::compute_tm_score<ScalarBackend>(
        aligned_mob.data(), aligned_ref.data(), matches, L2_);
    return 0.5f * (tm_ref + tm_mob);
}

MSAResult::MSAResult(int num_sequences, int alignment_length, float ecs_score,
                     std::vector<std::string> sequences, std::vector<std::string> identifiers)
    : num_sequences_(num_sequences),
      alignment_length_(alignment_length),
      ecs_score_(ecs_score),
      sequences_(std::move(sequences)),
      identifiers_(std::move(identifiers)) {
    if (num_sequences_ <= 0 || alignment_length_ <= 0) {
        throw std::invalid_argument("MSAResult requires positive dimensions");
    }
    if (static_cast<int>(sequences_.size()) != num_sequences_) {
        throw std::invalid_argument("Sequence list size does not match num_sequences");
    }
    for (const auto& seq : sequences_) {
        if (static_cast<int>(seq.size()) != alignment_length_) {
            throw std::invalid_argument("All sequences must match alignment_length");
        }
    }
    if (identifiers_.empty()) {
        identifiers_.resize(num_sequences_);
    } else if (static_cast<int>(identifiers_.size()) != num_sequences_) {
        throw std::invalid_argument("Identifier list size mismatch");
    }
}

void MSAResult::write_fasta(const std::string& path) const {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open FASTA file: " + path);
    }
    for (int i = 0; i < num_sequences_; ++i) {
        const std::string& id =
            identifiers_[i].empty() ? ("sequence_" + std::to_string(i)) : identifiers_[i];
        out << ">" << id;
        if (ecs_score_ > 0.0f && i == 0) {
            out << " | ecs=" << ecs_score_;
        }
        out << "\n";
        out << sequences_[i] << "\n";
    }
}

void MSAResult::write_pdb(const std::string& path, const float* const* coords_list,
                          int num_coords) const {
    if (!coords_list) {
        throw std::invalid_argument("coords_list is null");
    }

    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open PDB output: " + path);
    }

    const int models = std::min(num_sequences_, num_coords);
    if (models == 0) {
        throw std::runtime_error("No coordinate sets provided for PDB export");
    }

    for (int model = 0; model < models; ++model) {
        const float* coords = coords_list[model];
        if (!coords) {
            throw std::runtime_error("Null coordinate pointer for sequence " +
                                     std::to_string(model));
        }

        out << "MODEL     " << (model + 1) << "\n";
        int residue_index = 0;
        size_t coord_offset = 0;

        for (int col = 0; col < alignment_length_; ++col) {
            const char aa = sequences_[model][col];
            if (aa == '-') {
                continue;
            }
            const float x = coords[coord_offset + 0];
            const float y = coords[coord_offset + 1];
            const float z = coords[coord_offset + 2];
            coord_offset += 3;

            out << "ATOM  " << std::setw(5) << residue_index + 1 << "  CA  "
                << "ALA " << static_cast<char>('A' + (model % 26)) << std::setw(4)
                << residue_index + 1 << "    " << std::fixed << std::setprecision(3) << std::setw(8)
                << x << std::setw(8) << y << std::setw(8) << z << "  1.00 20.00          "
                << std::setw(2) << aa << "\n";

            ++residue_index;
        }

        out << "ENDMDL\n";
    }

    out << "END\n";
}

void MSAResult::to_array(int* out_array) const {
    if (!out_array) {
        throw std::invalid_argument("to_array requires a valid destination buffer");
    }

    for (int col = 0; col < alignment_length_; ++col) {
        for (int row = 0; row < num_sequences_; ++row) {
            out_array[col * num_sequences_ + row] = residue_to_index(sequences_[row][col]);
        }
    }
}

std::string MSAResult::get_sequence(int index) const {
    if (index < 0 || index >= num_sequences_) {
        throw std::out_of_range("Sequence index out of range");
    }
    return sequences_[index];
}

std::string MSAResult::get_column(int index) const {
    if (index < 0 || index >= alignment_length_) {
        throw std::out_of_range("Column index out of range");
    }
    std::string column;
    column.reserve(num_sequences_);
    for (const auto& seq : sequences_) {
        column.push_back(seq[index]);
    }
    return column;
}

std::string MSAResult::get_consensus(float threshold) const {
    if (threshold < 0.0f || threshold > 1.0f) {
        throw std::invalid_argument("Consensus threshold must be in [0, 1]");
    }

    std::string consensus(alignment_length_, '-');
    for (int col = 0; col < alignment_length_; ++col) {
        std::array<int, 128> counts{};
        int best_char = '-';
        int best_count = 0;

        for (int row = 0; row < num_sequences_; ++row) {
            const char aa = sequences_[row][col];
            if (aa == '-') {
                continue;
            }
            const unsigned char idx = static_cast<unsigned char>(aa);
            ++counts[idx];
            if (counts[idx] > best_count) {
                best_count = counts[idx];
                best_char = aa;
            }
        }

        const float frac = static_cast<float>(best_count) / static_cast<float>(num_sequences_);
        if (frac >= threshold && best_char != '-') {
            consensus[col] = static_cast<char>(best_char);
        }
    }

    return consensus;
}

void MSAResult::compute_conservation(float* out_conservation) const {
    if (!out_conservation) {
        throw std::invalid_argument("Conservation buffer is null");
    }

    for (int col = 0; col < alignment_length_; ++col) {
        std::array<int, 128> counts{};
        for (int row = 0; row < num_sequences_; ++row) {
            const char aa = sequences_[row][col];
            if (aa != '-') {
                ++counts[static_cast<unsigned char>(aa)];
            }
        }
        int max_count = 0;
        for (int value : counts) {
            max_count = std::max(max_count, value);
        }
        out_conservation[col] = static_cast<float>(max_count) / static_cast<float>(num_sequences_);
    }
}

MSAResult MSAResult::filter_gaps(float threshold) const {
    if (threshold < 0.0f || threshold > 1.0f) {
        throw std::invalid_argument("Gap threshold must be in [0, 1]");
    }

    std::vector<std::string> filtered(num_sequences_);
    for (auto& seq : filtered) {
        seq.reserve(alignment_length_);
    }

    for (int col = 0; col < alignment_length_; ++col) {
        int gaps = 0;
        for (int row = 0; row < num_sequences_; ++row) {
            if (sequences_[row][col] == '-') {
                ++gaps;
            }
        }

        const float gap_fraction = static_cast<float>(gaps) / static_cast<float>(num_sequences_);
        if (gap_fraction > threshold) {
            continue;
        }

        for (int row = 0; row < num_sequences_; ++row) {
            filtered[row].push_back(sequences_[row][col]);
        }
    }

    const int new_length = filtered.empty() ? 0 : static_cast<int>(filtered.front().size());
    return MSAResult(num_sequences_, new_length, ecs_score_, filtered, identifiers_);
}

void MSAResult::compute_pairwise_identity(float* out_identity) const {
    if (!out_identity) {
        throw std::invalid_argument("Identity buffer is null");
    }

    for (int i = 0; i < num_sequences_; ++i) {
        out_identity[i * num_sequences_ + i] = 1.0f;
        for (int j = i + 1; j < num_sequences_; ++j) {
            int matches = 0;
            int valid = 0;
            for (int col = 0; col < alignment_length_; ++col) {
                const char a = sequences_[i][col];
                const char b = sequences_[j][col];
                if (a == '-' || b == '-') {
                    continue;
                }
                ++valid;
                if (a == b) {
                    ++matches;
                }
            }
            const float identity =
                (valid == 0) ? 0.0f : static_cast<float>(matches) / static_cast<float>(valid);
            out_identity[i * num_sequences_ + j] = identity;
            out_identity[j * num_sequences_ + i] = identity;
        }
    }
}

EmbeddingResult::EmbeddingResult(float* embeddings, int L, int hidden_dim)
    : embeddings_(embeddings), L_(L), hidden_dim_(hidden_dim) {
    if (!embeddings_) {
        throw std::invalid_argument("EmbeddingResult requires a valid buffer");
    }
    if (L_ <= 0 || hidden_dim_ <= 0) {
        throw std::invalid_argument("EmbeddingResult dimensions must be positive");
    }
}

void EmbeddingResult::save(const std::string& path) const {
    save_npy_2d(path, embeddings_.get(), L_, hidden_dim_);
}

void EmbeddingResult::compute_pairwise_distances(float* out_distances) const {
    if (!out_distances) {
        throw std::invalid_argument("Distance buffer is null");
    }

    for (int i = 0; i < L_; ++i) {
        out_distances[i * L_ + i] = 0.0f;
        const float* row_i = embeddings_.get() + static_cast<size_t>(i) * hidden_dim_;

        for (int j = i + 1; j < L_; ++j) {
            const float* row_j = embeddings_.get() + static_cast<size_t>(j) * hidden_dim_;
            double sum = 0.0;
            for (int d = 0; d < hidden_dim_; ++d) {
                const double diff = static_cast<double>(row_i[d]) - static_cast<double>(row_j[d]);
                sum += diff * diff;
            }
            const float distance = static_cast<float>(std::sqrt(sum));
            out_distances[i * L_ + j] = distance;
            out_distances[j * L_ + i] = distance;
        }
    }
}

EmbeddingResult EmbeddingResult::get_subset(const std::vector<int>& indices) const {
    if (indices.empty()) {
        throw std::invalid_argument("Subset indices must not be empty");
    }
    auto subset = std::make_unique<float[]>(indices.size() * hidden_dim_);
    for (size_t row = 0; row < indices.size(); ++row) {
        const int idx = indices[row];
        if (idx < 0 || idx >= L_) {
            throw std::out_of_range("Subset index out of range");
        }
        const float* src = embeddings_.get() + static_cast<size_t>(idx) * hidden_dim_;
        float* dst = subset.get() + row * hidden_dim_;
        std::copy(src, src + hidden_dim_, dst);
    }
    return EmbeddingResult(subset.release(), static_cast<int>(indices.size()), hidden_dim_);
}

EmbeddingResult EmbeddingResult::normalize() const {
    auto normalized = std::make_unique<float[]>(static_cast<size_t>(L_) * hidden_dim_);
    for (int i = 0; i < L_; ++i) {
        const float* src = embeddings_.get() + static_cast<size_t>(i) * hidden_dim_;
        float* dst = normalized.get() + static_cast<size_t>(i) * hidden_dim_;
        double norm_sq = 0.0;
        for (int d = 0; d < hidden_dim_; ++d) {
            norm_sq += static_cast<double>(src[d]) * static_cast<double>(src[d]);
        }
        const double inv_norm = (norm_sq > 0.0) ? 1.0 / std::sqrt(norm_sq) : 0.0;
        for (int d = 0; d < hidden_dim_; ++d) {
            dst[d] = static_cast<float>(src[d] * inv_norm);
        }
    }
    return EmbeddingResult(normalized.release(), L_, hidden_dim_);
}

SimilarityResult::SimilarityResult(float* similarity, int L1, int L2)
    : similarity_(similarity), L1_(L1), L2_(L2) {
    if (!similarity_) {
        throw std::invalid_argument("SimilarityResult requires a valid buffer");
    }
    if (L1_ <= 0 || L2_ <= 0) {
        throw std::invalid_argument("SimilarityResult dimensions must be positive");
    }
}

void SimilarityResult::save(const std::string& path) const {
    save_npy_2d(path, similarity_.get(), L1_, L2_);
}

std::vector<std::tuple<int, int, float>> SimilarityResult::get_top_k(int k) const {
    if (k <= 0) {
        throw std::invalid_argument("k must be positive");
    }
    const int total = L1_ * L2_;
    k = std::min(k, total);

    std::vector<std::tuple<int, int, float>> entries;
    entries.reserve(total);
    for (int i = 0; i < L1_; ++i) {
        for (int j = 0; j < L2_; ++j) {
            entries.emplace_back(i, j, similarity_.get()[i * L2_ + j]);
        }
    }

    std::partial_sort(entries.begin(), entries.begin() + k, entries.end(),
                      [](const auto& a, const auto& b) { return std::get<2>(a) > std::get<2>(b); });
    entries.resize(k);
    return entries;
}

SimilarityResult SimilarityResult::threshold(float cutoff) const {
    auto filtered = std::make_unique<float[]>(static_cast<size_t>(L1_) * L2_);
    for (int i = 0; i < L1_; ++i) {
        for (int j = 0; j < L2_; ++j) {
            const float value = similarity_.get()[i * L2_ + j];
            filtered[i * L2_ + j] = (value >= cutoff) ? value : 0.0f;
        }
    }
    return SimilarityResult(filtered.release(), L1_, L2_);
}

SimilarityResult SimilarityResult::normalize() const {
    const size_t size = static_cast<size_t>(L1_) * L2_;
    float min_value = std::numeric_limits<float>::max();
    float max_value = std::numeric_limits<float>::lowest();
    for (size_t i = 0; i < size; ++i) {
        min_value = std::min(min_value, similarity_.get()[i]);
        max_value = std::max(max_value, similarity_.get()[i]);
    }

    auto normalized = std::make_unique<float[]>(size);
    if (max_value <= min_value + 1e-12f) {
        std::fill(normalized.get(), normalized.get() + size, 0.0f);
        return SimilarityResult(normalized.release(), L1_, L2_);
    }

    const float scale = 1.0f / (max_value - min_value);
    for (size_t i = 0; i < size; ++i) {
        normalized[i] = (similarity_.get()[i] - min_value) * scale;
    }
    return SimilarityResult(normalized.release(), L1_, L2_);
}

}  // namespace pfalign
