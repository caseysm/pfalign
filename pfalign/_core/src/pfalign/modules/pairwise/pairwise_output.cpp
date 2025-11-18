/**
 * Pairwise alignment output utilities (FASTA + PDB).
 */

#include "pfalign/modules/pairwise/pairwise_align.h"
#include "pfalign/io/fasta_writer.h"
#include "pfalign/io/pdb_writer.h"
#include "pfalign/io/protein_structure.h"
#include "pfalign/primitives/kabsch/kabsch_impl.h"
#include "pfalign/primitives/structural_metrics/structural_metrics_impl.h"
#include "pfalign/dispatch/backend_traits.h"
#include <algorithm>
#include <string>
#include <vector>

namespace pfalign {
namespace pairwise {

namespace {

using pfalign::io::Chain;
using pfalign::io::Protein;
using pfalign::io::Residue;

[[maybe_unused]] constexpr size_t kDefaultArenaBytes = size_t(16) * 1024 * 1024;

const float* ca_ptr(const float* coords, int residue_idx) {
    // Layout: [L * 4 * 3], CA atom is index 1
    return coords + residue_idx * 12 + 3;
}

Protein make_placeholder_protein(int length, char chain_id) {
    Protein protein;
    protein.chains.emplace_back(chain_id);
    Chain& chain = protein.chains.back();
    chain.residues.reserve(length);

    for (int i = 0; i < length; ++i) {
        chain.residues.emplace_back(i + 1, ' ', "ALA");
    }

    return protein;
}

std::string fallback_id(const std::string& id, const std::string& fallback) {
    return id.empty() ? fallback : id;
}

}  // namespace

bool AlignmentResult::write_fasta(const std::string& output_path, const std::string& seq1,
                                  const std::string& seq2) const {
    if (!alignment_path || path_length <= 0) {
        return false;
    }

    const std::string id_a = fallback_id(id1, "sequence_1");
    const std::string id_b = fallback_id(id2, "sequence_2");

    return io::write_alignment_fasta(output_path, id_a, id_b, seq1, seq2, alignment_path,
                                     path_length, score, partition);
}

bool AlignmentResult::write_superposed_pdb(const std::string& output_path, int reference,
                                           pfalign::memory::GrowableArena* arena) const {
    if (!alignment_path || path_length <= 0) {
        return false;
    }

    if (!coords1 || !coords2 || L1 <= 0 || L2 <= 0) {
        return false;
    }

    const int ref_index = (reference == 0) ? 0 : 1;

    const float* ref_coords = (ref_index == 0) ? coords1 : coords2;
    const float* mob_coords = (ref_index == 0) ? coords2 : coords1;
    const int ref_len = (ref_index == 0) ? L1 : L2;
    const int mob_len = (ref_index == 0) ? L2 : L1;

    if (!ref_coords || !mob_coords) {
        return false;
    }

    std::vector<float> ref_ca;
    std::vector<float> mob_ca;
    std::vector<std::pair<int, int>> aligned_pairs;
    ref_ca.reserve(static_cast<size_t>(path_length) * 3);
    mob_ca.reserve(static_cast<size_t>(path_length) * 3);
    aligned_pairs.reserve(path_length);

    for (int k = 0; k < path_length; ++k) {
        const AlignmentPair& pair = alignment_path[k];
        if (pair.i < 0 || pair.j < 0) {
            continue;
        }

        const int ref_pos = (ref_index == 0) ? pair.i : pair.j;
        const int mob_pos = (ref_index == 0) ? pair.j : pair.i;
        if (ref_pos < 0 || mob_pos < 0 || ref_pos >= ref_len || mob_pos >= mob_len) {
            continue;
        }

        const float* ref_atom = ca_ptr(ref_coords, ref_pos);
        const float* mob_atom = ca_ptr(mob_coords, mob_pos);

        for (int dim = 0; dim < 3; ++dim) {
            ref_ca.push_back(ref_atom[dim]);
            mob_ca.push_back(mob_atom[dim]);
        }

        const int mobile_pair = (ref_index == 0) ? pair.j : pair.i;
        const int reference_pair = (ref_index == 0) ? pair.i : pair.j;
        aligned_pairs.emplace_back(mobile_pair, reference_pair);
    }

    const int num_aligned = static_cast<int>(ref_ca.size() / 3);
    if (num_aligned < 3) {
        return false;  // Need at least 3 points for a stable transform
    }

    float R[9];
    float t[3];
    float rmsd = 0.0f;
    kabsch::kabsch_align<ScalarBackend>(mob_ca.data(), ref_ca.data(), num_aligned, R, t, &rmsd);

    std::vector<float> mob_ca_aligned(static_cast<size_t>(num_aligned) * 3);
    for (int idx = 0; idx < num_aligned; ++idx) {
        const float* src = mob_ca.data() + idx * 3;
        float* dst = mob_ca_aligned.data() + idx * 3;
        dst[0] = R[0] * src[0] + R[1] * src[1] + R[2] * src[2] + t[0];
        dst[1] = R[3] * src[0] + R[4] * src[1] + R[5] * src[2] + t[1];
        dst[2] = R[6] * src[0] + R[7] * src[1] + R[8] * src[2] + t[2];
    }

    const float tm_ref = structural_metrics::compute_tm_score<ScalarBackend>(
        mob_ca_aligned.data(), ref_ca.data(), num_aligned, ref_len);
    const float tm_mob = structural_metrics::compute_tm_score<ScalarBackend>(
        mob_ca_aligned.data(), ref_ca.data(), num_aligned, mob_len);

    float gdt_ts = 0.0f;
    float gdt_ha = 0.0f;
    structural_metrics::compute_gdt<ScalarBackend>(mob_ca_aligned.data(), ref_ca.data(),
                                                   num_aligned, &gdt_ts, &gdt_ha);

    pfalign::memory::GrowableArena local_arena(16);  // 16 MB (kDefaultArenaBytes / (1024*1024))
    pfalign::memory::GrowableArena* work_arena = arena ? arena : &local_arena;

    float* coords_transformed = work_arena->allocate<float>(static_cast<size_t>(mob_len) * 4 * 3);
    if (coords_transformed == nullptr) {
        return false;
    }

    kabsch::apply_transformation<ScalarBackend>(R, t, mob_coords, coords_transformed, mob_len);

    const io::Protein* reference_meta = (ref_index == 0) ? protein1 : protein2;
    const io::Protein* mobile_meta = (ref_index == 0) ? protein2 : protein1;

    Protein placeholder_reference;
    Protein placeholder_mobile;

    if (reference_meta == nullptr) {
        placeholder_reference = make_placeholder_protein(ref_len, 'A');
        reference_meta = &placeholder_reference;
    }
    if (mobile_meta == nullptr) {
        placeholder_mobile = make_placeholder_protein(mob_len, 'B');
        mobile_meta = &placeholder_mobile;
    }

    return io::write_superposed_pair(output_path, *mobile_meta, coords_transformed, *reference_meta,
                                     ref_coords, R, t, rmsd, tm_mob, tm_ref, gdt_ts, gdt_ha,
                                     aligned_pairs);
}

}  // namespace pairwise
}  // namespace pfalign
