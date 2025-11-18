#pragma once

#include <string>
#include <vector>

#include "pfalign/io/protein_structure.h"

namespace pfalign {
namespace commands {

enum class InputType { kStructure, kEmbeddings };

bool IsStructureExtension(const std::string& path);
bool IsEmbeddingExtension(const std::string& path);
InputType DetectInputType(const std::string& path);

io::Protein LoadStructureFile(const std::string& path);

struct EmbeddingArray {
    std::vector<float> values;
    int rows = 0;
    int cols = 0;
};

EmbeddingArray LoadEmbeddingFile(const std::string& path);

/**
 * Chain specification - can be either an index or a chain ID.
 *
 * Supports unified chain handling matching Python API:
 * - "0", "1", "2" -> chain by index
 * - "A", "B", "C" -> chain by ID
 */
struct ChainSpec {
    enum class Type { Index, ID };

    Type type;
    int index;
    std::string id;

    /**
     * Parse chain specification from string.
     *
     * @param spec Chain specification ("0", "1", "A", "B", etc.)
     * @return ChainSpec object
     */
    static ChainSpec parse(const std::string& spec);

    /**
     * Resolve chain specification to chain index.
     *
     * @param protein Protein structure containing chains
     * @param path Path to structure file (for error messages)
     * @return Chain index (0-based)
     * @throws ChainNotFoundError if chain ID not found
     * @throws ValidationError if index out of range
     */
    int resolve(const io::Protein& protein, const std::string& path) const;
};

}  // namespace commands
}  // namespace pfalign
