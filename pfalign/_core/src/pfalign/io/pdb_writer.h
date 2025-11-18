/**
 * PDB Writer for Aligned Protein Structures
 *
 * Writes superposed protein pairs to multi-model PDB format.
 * Includes REMARK records with alignment metrics.
 */

#pragma once

#include "protein_structure.h"
#include <string>
#include <vector>
#include <utility>

namespace pfalign {
namespace io {

/**
 * Write superposed protein pair to PDB format.
 *
 * Outputs multi-model PDB file:
 * - MODEL 1: Aligned protein 1 (after Kabsch transformation)
 * - MODEL 2: Reference protein 2 (original coordinates)
 *
 * Features:
 * - REMARK records: rotation matrix, translation, RMSD, TM-score, GDT
 * - B-factor encoding: 1.00 for aligned residues, 0.00 for gaps
 * - Preserves original chain IDs, residue numbers, residue names
 *
 * @param output_path Output PDB file path
 * @param protein1 Original protein 1 structure metadata
 * @param coords1_aligned Transformed coordinates [L1 * 4 * 3]
 * @param protein2 Original protein 2 structure metadata
 * @param coords2 Original coords [L2 * 4 * 3]
 * @param rotation Rotation matrix [9] from Kabsch
 * @param translation Translation vector [3] from Kabsch
 * @param rmsd RMSD value
 * @param tm_score1 TM-score normalized by L1
 * @param tm_score2 TM-score normalized by L2
 * @param gdt_ts GDT-TS score
 * @param gdt_ha GDT-HA score
 * @param aligned_pairs Alignment path for B-factor coloring [(i,j), ...]
 * @return true on success, false on I/O error
 */
bool write_superposed_pair(const std::string& output_path, const Protein& protein1,
                           const float* coords1_aligned, const Protein& protein2,
                           const float* coords2, const float* rotation, const float* translation,
                           float rmsd, float tm_score1, float tm_score2, float gdt_ts, float gdt_ha,
                           const std::vector<std::pair<int, int>>& aligned_pairs);

}  // namespace io
}  // namespace pfalign
