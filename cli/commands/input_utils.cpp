#include "commands/input_utils.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "pfalign/io/mmcif_parser.h"
#include "pfalign/io/pdb_parser.h"
#include "pfalign/errors/messages.h"
#include "pfalign/errors/validators.h"

namespace pfalign {
namespace commands {

namespace {

std::string ToLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool IEndsWith(const std::string& value, const std::string& suffix) {
    if (value.length() < suffix.length()) {
        return false;
    }
    return std::equal(suffix.rbegin(), suffix.rend(), value.rbegin(),
                      [](char a, char b) { return std::tolower(a) == std::tolower(b); });
}

std::string ReadFirstLine(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw pfalign::errors::FileNotFoundError(path, "File");
    }
    std::string line;
    std::getline(file, line);
    return line;
}

std::vector<size_t> ParseShape(const std::string& header) {
    // Try multiple quote styles for robustness
    size_t shape_pos = std::string::npos;
    const std::vector<std::string> shape_patterns = {"'shape':", "\"shape\":", "shape:"};

    for (const auto& pattern : shape_patterns) {
        shape_pos = header.find(pattern);
        if (shape_pos != std::string::npos) {
            break;
        }
    }

    if (shape_pos == std::string::npos) {
        throw std::runtime_error("Failed to parse shape from .npy header - 'shape' key not found");
    }

    size_t paren_open = header.find('(', shape_pos);
    size_t paren_close = header.find(')', paren_open);
    if (paren_open == std::string::npos || paren_close == std::string::npos ||
        paren_close <= paren_open) {
        throw std::runtime_error("Invalid shape tuple in .npy header - malformed tuple");
    }

    std::string shape_str = header.substr(paren_open + 1, paren_close - paren_open - 1);
    std::vector<size_t> shape;

    std::stringstream ss(shape_str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        // Trim whitespace
        token.erase(token.begin(), std::find_if(token.begin(), token.end(), [](unsigned char ch) {
                        return !std::isspace(ch);
                    }));
        token.erase(std::find_if(token.rbegin(), token.rend(),
                                 [](unsigned char ch) { return !std::isspace(ch); })
                        .base(),
                    token.end());
        if (!token.empty()) {
            try {
                shape.push_back(static_cast<size_t>(std::stoll(token)));
            } catch (const std::exception& e) {
                throw std::runtime_error("Invalid shape value in .npy header: '" + token + "'");
            }
        }
    }

    if (shape.empty()) {
        throw std::runtime_error("Empty shape in .npy header");
    }

    return shape;
}

}  // namespace

bool IsStructureExtension(const std::string& path) {
    std::string lower = ToLower(path);
    return IEndsWith(lower, ".pdb") || IEndsWith(lower, ".cif") || IEndsWith(lower, ".mmcif");
}

bool IsEmbeddingExtension(const std::string& path) {
    std::string lower = ToLower(path);
    return IEndsWith(lower, ".npy");
}

InputType DetectInputType(const std::string& path) {
    pfalign::validation::validate_file_exists(path, "Input file");

    if (IsEmbeddingExtension(path)) {
        return InputType::kEmbeddings;
    }
    if (IsStructureExtension(path)) {
        return InputType::kStructure;
    }

    std::string first_line = ReadFirstLine(path);
    std::string upper_line = first_line;
    std::transform(upper_line.begin(), upper_line.end(), upper_line.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });

    if (upper_line.rfind("HEADER", 0) == 0 || upper_line.rfind("ATOM", 0) == 0 ||
        upper_line.rfind("HETATM", 0) == 0 || upper_line.rfind("MODEL", 0) == 0 ||
        upper_line.rfind("REMARK", 0) == 0 || upper_line.rfind("COMPND", 0) == 0) {
        return InputType::kStructure;
    }

    if (upper_line.rfind("DATA_", 0) == 0) {
        return InputType::kStructure;
    }

    throw pfalign::errors::messages::unsupported_format(
        path,
        std::filesystem::path(path).extension().string(),
        {".pdb", ".cif", ".mmcif", ".npy"}
    );
}

io::Protein LoadStructureFile(const std::string& path) {
    bool prefer_cif = IEndsWith(path, ".cif") || IEndsWith(path, ".mmcif");
    io::Protein protein;

    auto try_parse = [&](bool use_cif) -> bool {
        try {
            if (use_cif) {
                io::mmCIFParser parser;
                protein = parser.parse_file(path);
            } else {
                io::PDBParser parser;
                protein = parser.parse_file(path);
            }
            return true;
        } catch (...) {
            return false;
        }
    };

    if (prefer_cif) {
        if (try_parse(true) || try_parse(false)) {
            return protein;
        }
    } else {
        if (try_parse(false) || try_parse(true)) {
            return protein;
        }
    }

    std::string format = prefer_cif ? "mmCIF" : "PDB";
    throw pfalign::errors::messages::file_parse_error(
        path,
        format,
        "Could not parse as PDB or mmCIF format"
    );
}

EmbeddingArray LoadEmbeddingFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw pfalign::errors::FileNotFoundError(path, "Embedding file");
    }

    char magic[7] = {0};
    file.read(magic, 6);
    if (std::string(magic, 6) != "\x93NUMPY") {
        throw pfalign::errors::messages::invalid_npy_format(path, "Invalid magic header");
    }

    uint8_t major = 0;
    uint8_t minor = 0;
    file.read(reinterpret_cast<char*>(&major), 1);
    file.read(reinterpret_cast<char*>(&minor), 1);

    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t len16 = 0;
        file.read(reinterpret_cast<char*>(&len16), 2);
        header_len = len16;
    } else if (major == 2 || major == 3) {
        file.read(reinterpret_cast<char*>(&header_len), 4);
    } else {
        throw pfalign::errors::messages::invalid_npy_format(
            path,
            "Unsupported version " + std::to_string(major) + "." + std::to_string(minor)
        );
    }

    std::vector<char> header_buf(header_len);
    file.read(header_buf.data(), header_len);
    std::string header(header_buf.begin(), header_buf.end());

    if (header.find("fortran_order") != std::string::npos) {
        bool is_fortran = header.find("True", header.find("fortran_order")) != std::string::npos;
        if (is_fortran) {
            throw pfalign::errors::messages::invalid_npy_format(
                path,
                "Fortran-order arrays not supported (use C-order)"
            );
        }
    }

    bool is_float32 = header.find("<f4") != std::string::npos ||
                      header.find("'f4'") != std::string::npos ||
                      header.find("\"f4\"") != std::string::npos;
    if (!is_float32) {
        throw pfalign::errors::messages::invalid_npy_format(
            path,
            "Only float32 dtype supported (file uses different dtype)"
        );
    }

    std::vector<size_t> shape = ParseShape(header);
    if (shape.size() < 2) {
        throw pfalign::errors::messages::invalid_npy_format(
            path,
            "Expected 2D array, got " + std::to_string(shape.size()) + "D"
        );
    }

    size_t rows = shape[0];
    size_t cols = shape[1];
    size_t total_elems = 1;
    for (size_t dim : shape) {
        total_elems *= dim;
    }

    EmbeddingArray result;
    result.rows = static_cast<int>(rows);
    result.cols = static_cast<int>(cols);
    result.values.resize(total_elems);

    file.read(reinterpret_cast<char*>(result.values.data()), total_elems * sizeof(float));
    if (!file) {
        throw pfalign::errors::messages::invalid_npy_format(
            path,
            "Incomplete or corrupted data"
        );
    }

    return result;
}

// Chain specification parsing and resolution

ChainSpec ChainSpec::parse(const std::string& spec) {
    ChainSpec result;

    // Try to parse as integer
    try {
        size_t pos = 0;
        int value = std::stoi(spec, &pos);
        if (pos == spec.length()) {
            // Successfully parsed as integer
            result.type = Type::Index;
            result.index = value;
            return result;
        }
    } catch (...) {
        // Not an integer, treat as chain ID
    }

    // Treat as chain ID
    result.type = Type::ID;
    result.id = spec;
    return result;
}

int ChainSpec::resolve(const io::Protein& protein, const std::string& path) const {
    if (type == Type::Index) {
        // Validate index
        pfalign::validation::validate_chain_index(
            static_cast<int>(protein.num_chains()),
            index,
            path
        );
        return index;
    }

    // Type::ID - search for chain
    for (size_t i = 0; i < protein.num_chains(); ++i) {
        if (std::string(1, protein.chains[i].chain_id) == id) {
            return static_cast<int>(i);
        }
    }

    // Chain ID not found - build list of available chains
    std::vector<std::string> available;
    for (size_t i = 0; i < protein.num_chains(); ++i) {
        available.push_back(std::string(1, protein.chains[i].chain_id));
    }

    throw pfalign::errors::messages::chain_not_found(id, path, available);
}

}  // namespace commands
}  // namespace pfalign
