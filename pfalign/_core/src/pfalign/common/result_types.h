#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace pfalign {

class PairwiseResult {
public:
    PairwiseResult(float* posteriors, int L1, int L2, float score, float partition,
                   std::vector<std::pair<int, int>> alignment);
    ~PairwiseResult() = default;

    PairwiseResult(PairwiseResult&& other) noexcept = default;
    PairwiseResult& operator=(PairwiseResult&& other) noexcept = default;

    PairwiseResult(const PairwiseResult&) = delete;
    PairwiseResult& operator=(const PairwiseResult&) = delete;

    const float* posteriors() const {
        return posteriors_.get();
    }
    int L1() const {
        return L1_;
    }
    int L2() const {
        return L2_;
    }
    float score() const {
        return score_;
    }
    float partition() const {
        return partition_;
    }
    const std::vector<std::pair<int, int>>& alignment() const {
        return alignment_;
    }

    void write_fasta(const std::string& path, const std::string& seq1, const std::string& seq2,
                     const std::string& id1 = "seq1", const std::string& id2 = "seq2") const;
    void write_npy(const std::string& path) const;
    void write_pdb(const std::string& path, const float* coords1, int coords1_length,
                   const float* coords2, int coords2_length, int reference = 0) const;

    std::vector<std::pair<int, int>> get_aligned_residues() const;
    float compute_coverage() const;
    PairwiseResult threshold(float cutoff) const;

    void get_aligned_coords(const float* coords1, const float* coords2, float* out_coords1,
                            float* out_coords2, int reference = 0) const;
    float compute_rmsd(const float* coords1, const float* coords2) const;
    float compute_tm_score(const float* coords1, const float* coords2) const;

private:
    std::unique_ptr<float[]> posteriors_;
    int L1_;
    int L2_;
    float score_;
    float partition_;
    std::vector<std::pair<int, int>> alignment_;
};

class MSAResult {
public:
    MSAResult(int num_sequences, int alignment_length, float ecs_score,
              std::vector<std::string> sequences, std::vector<std::string> identifiers);

    int num_sequences() const {
        return num_sequences_;
    }
    int alignment_length() const {
        return alignment_length_;
    }
    float ecs_score() const {
        return ecs_score_;
    }
    const std::vector<std::string>& sequences() const {
        return sequences_;
    }
    const std::vector<std::string>& identifiers() const {
        return identifiers_;
    }

    void write_fasta(const std::string& path) const;
    void write_pdb(const std::string& path, const float* const* coords_list, int num_coords) const;
    void to_array(int* out_array) const;

    std::string get_sequence(int index) const;
    std::string get_column(int index) const;
    std::string get_consensus(float threshold = 0.5f) const;

    void compute_conservation(float* out_conservation) const;
    MSAResult filter_gaps(float threshold = 0.5f) const;
    void compute_pairwise_identity(float* out_identity) const;

private:
    int num_sequences_;
    int alignment_length_;
    float ecs_score_;
    std::vector<std::string> sequences_;
    std::vector<std::string> identifiers_;
};

class EmbeddingResult {
public:
    EmbeddingResult(float* embeddings, int L, int hidden_dim);
    ~EmbeddingResult() = default;

    EmbeddingResult(EmbeddingResult&& other) noexcept = default;
    EmbeddingResult& operator=(EmbeddingResult&& other) noexcept = default;

    EmbeddingResult(const EmbeddingResult&) = delete;
    EmbeddingResult& operator=(const EmbeddingResult&) = delete;

    const float* embeddings() const {
        return embeddings_.get();
    }
    int sequence_length() const {
        return L_;
    }
    int hidden_dim() const {
        return hidden_dim_;
    }

    void save(const std::string& path) const;

    void compute_pairwise_distances(float* out_distances) const;
    EmbeddingResult get_subset(const std::vector<int>& indices) const;
    EmbeddingResult normalize() const;

private:
    std::unique_ptr<float[]> embeddings_;
    int L_;
    int hidden_dim_;
};

class SimilarityResult {
public:
    SimilarityResult(float* similarity, int L1, int L2);
    ~SimilarityResult() = default;

    SimilarityResult(SimilarityResult&& other) noexcept = default;
    SimilarityResult& operator=(SimilarityResult&& other) noexcept = default;

    SimilarityResult(const SimilarityResult&) = delete;
    SimilarityResult& operator=(const SimilarityResult&) = delete;

    const float* similarity() const {
        return similarity_.get();
    }
    int L1() const {
        return L1_;
    }
    int L2() const {
        return L2_;
    }
    std::pair<int, int> shape() const {
        return {L1_, L2_};
    }

    void save(const std::string& path) const;

    std::vector<std::tuple<int, int, float>> get_top_k(int k) const;
    SimilarityResult threshold(float cutoff) const;
    SimilarityResult normalize() const;

private:
    std::unique_ptr<float[]> similarity_;
    int L1_;
    int L2_;
};

}  // namespace pfalign
