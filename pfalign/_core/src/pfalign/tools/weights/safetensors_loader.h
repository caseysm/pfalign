/**
 * Minimal safetensors loader for C++17
 *
 * Zero dependencies, header-only implementation.
 *
 * Safetensors format:
 *   - 8 bytes: N (uint64_t little-endian) - header size
 *   - N bytes: JSON UTF-8 string with tensor metadata
 *   - Rest: raw tensor data (row-major, little-endian float32)
 *
 * Header JSON format:
 * {
 *   "tensor_name": {
 *     "dtype": "F32",
 *     "shape": [dim0, dim1, ...],
 *     "data_offsets": [begin, end]
 *   },
 *   ...
 * }
 *
 * Based on: https://github.com/huggingface/safetensors
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <stdexcept>
#include <sstream>

namespace pfalign {
namespace weights {

/**
 * Tensor metadata from safetensors header
 */
struct TensorInfo {
    std::string dtype;          // "F32", "F16", etc.
    std::vector<size_t> shape;  // Tensor dimensions
    size_t data_begin;          // Offset in file (relative to data section)
    size_t data_end;            // End offset

    size_t num_elements() const {
        size_t total = 1;
        for (size_t dim : shape)
            total *= dim;
        return total;
    }

    size_t num_bytes() const {
        return data_end - data_begin;
    }
};

/**
 * Minimal JSON parser for safetensors header
 *
 * Only parses the subset needed for safetensors:
 * - String keys
 * - Nested objects
 * - Integer arrays
 * - String values
 */
class MinimalJSONParser {
public:
    static std::map<std::string, TensorInfo> parse_header(const std::string& json) {
        std::map<std::string, TensorInfo> tensors;

        size_t pos = 0;
        skip_whitespace(json, pos);

        if (json[pos] != '{') {
            throw std::runtime_error("Expected '{' at start of JSON");
        }
        pos++;  // Skip '{'

        while (pos < json.size()) {
            skip_whitespace(json, pos);

            if (json[pos] == '}')
                break;  // End of object
            if (json[pos] == ',') {
                pos++;
                continue;
            }

            // Parse key
            std::string key = parse_string(json, pos);
            skip_whitespace(json, pos);

            if (json[pos] != ':') {
                throw std::runtime_error("Expected ':' after key");
            }
            pos++;
            skip_whitespace(json, pos);

            // Skip __metadata__ key
            if (key == "__metadata__") {
                skip_object(json, pos);
                continue;
            }

            // Parse tensor info object
            TensorInfo info = parse_tensor_info(json, pos);
            tensors[key] = info;
        }

        return tensors;
    }

private:
    static void skip_whitespace(const std::string& s, size_t& pos) {
        while (pos < s.size() &&
               (s[pos] == ' ' || s[pos] == '\n' || s[pos] == '\r' || s[pos] == '\t')) {
            pos++;
        }
    }

    static std::string parse_string(const std::string& s, size_t& pos) {
        if (s[pos] != '"') {
            throw std::runtime_error("Expected '\"' at start of string");
        }
        pos++;  // Skip opening quote

        std::string result;
        while (pos < s.size() && s[pos] != '"') {
            if (s[pos] == '\\') {
                pos++;  // Skip escape char
                if (pos >= s.size())
                    break;
            }
            result += s[pos++];
        }

        if (s[pos] != '"') {
            throw std::runtime_error("Unterminated string");
        }
        pos++;  // Skip closing quote

        return result;
    }

    static int64_t parse_number(const std::string& s, size_t& pos) {
        size_t start = pos;
        if (s[pos] == '-')
            pos++;

        while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') {
            pos++;
        }

        std::string num_str = s.substr(start, pos - start);
        return std::stoll(num_str);
    }

    static std::vector<size_t> parse_array(const std::string& s, size_t& pos) {
        std::vector<size_t> result;

        if (s[pos] != '[') {
            throw std::runtime_error("Expected '[' at start of array");
        }
        pos++;

        while (pos < s.size()) {
            skip_whitespace(s, pos);

            if (s[pos] == ']') {
                pos++;
                break;
            }

            if (s[pos] == ',') {
                pos++;
                continue;
            }

            int64_t num = parse_number(s, pos);
            result.push_back(static_cast<size_t>(num));
        }

        return result;
    }

    static TensorInfo parse_tensor_info(const std::string& s, size_t& pos) {
        TensorInfo info;

        if (s[pos] != '{') {
            throw std::runtime_error("Expected '{' at start of tensor info");
        }
        pos++;

        while (pos < s.size()) {
            skip_whitespace(s, pos);

            if (s[pos] == '}') {
                pos++;
                break;
            }

            if (s[pos] == ',') {
                pos++;
                continue;
            }

            std::string key = parse_string(s, pos);
            skip_whitespace(s, pos);

            if (s[pos] != ':') {
                throw std::runtime_error("Expected ':' after key");
            }
            pos++;
            skip_whitespace(s, pos);

            if (key == "dtype") {
                info.dtype = parse_string(s, pos);
            } else if (key == "shape") {
                info.shape = parse_array(s, pos);
            } else if (key == "data_offsets") {
                auto offsets = parse_array(s, pos);
                if (offsets.size() != 2) {
                    throw std::runtime_error("data_offsets must have 2 elements");
                }
                info.data_begin = offsets[0];
                info.data_end = offsets[1];
            } else {
                // Skip unknown key
                skip_value(s, pos);
            }
        }

        return info;
    }

    static void skip_value(const std::string& s, size_t& pos) {
        skip_whitespace(s, pos);

        if (s[pos] == '"') {
            parse_string(s, pos);
        } else if (s[pos] == '[') {
            parse_array(s, pos);
        } else if (s[pos] == '{') {
            skip_object(s, pos);
        } else {
            parse_number(s, pos);
        }
    }

    static void skip_object(const std::string& s, size_t& pos) {
        if (s[pos] != '{')
            return;
        pos++;

        int depth = 1;
        while (pos < s.size() && depth > 0) {
            if (s[pos] == '{')
                depth++;
            else if (s[pos] == '}')
                depth--;
            else if (s[pos] == '"') {
                // Skip string
                parse_string(s, pos);
                continue;
            }
            pos++;
        }
    }
};

/**
 * Safetensors file loader
 */
class SafetensorsLoader {
public:
    /**
     * Load SafeTensors from file.
     */
    SafetensorsLoader(const std::string& filepath)
        : source_type_(SourceType::FILE), filepath_(filepath), buffer_(nullptr), buffer_size_(0) {
        load_header();
    }

    /**
     * Load SafeTensors from memory buffer.
     *
     * @param buffer Pointer to SafeTensors data (must remain valid!)
     * @param size Buffer size in bytes
     */
    SafetensorsLoader(const uint8_t* buffer, size_t size)
        : source_type_(SourceType::BUFFER), filepath_(), buffer_(buffer), buffer_size_(size) {
        load_header();
    }

    /**
     * Get list of tensor names in file
     */
    std::vector<std::string> keys() const {
        std::vector<std::string> result;
        for (const auto& kv : tensors_) {
            result.push_back(kv.first);
        }
        return result;
    }

    /**
     * Get tensor info
     */
    const TensorInfo& get_info(const std::string& name) const {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) {
            throw std::runtime_error("Tensor not found: " + name);
        }
        return it->second;
    }

    /**
     * Load tensor as float32 array
     *
     * Returns pointer to newly allocated float array.
     * Caller is responsible for deletion.
     */
    float* load_tensor(const std::string& name) const {
        if (source_type_ == SourceType::FILE) {
            return load_tensor_from_file(name);
        } else {
            return load_tensor_from_buffer(name);
        }
    }

    /**
     * Check if tensor exists
     */
    bool has_tensor(const std::string& name) const {
        return tensors_.find(name) != tensors_.end();
    }

private:
    enum class SourceType { FILE, BUFFER };

    void load_header() {
        if (source_type_ == SourceType::FILE) {
            load_header_from_file();
        } else {
            load_header_from_buffer();
        }
    }

    void load_header_from_file() {
        std::ifstream file(filepath_, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open file: " + filepath_);
        }

        // Read header size (8 bytes, little-endian uint64)
        uint64_t header_size;
        file.read(reinterpret_cast<char*>(&header_size), 8);

        if (!file) {
            throw std::runtime_error("Failed to read header size");
        }

        header_size_ = header_size;

        // Sanity check (100MB limit from safetensors spec)
        if (header_size > 100 * 1024 * 1024) {
            throw std::runtime_error("Header size too large: " + std::to_string(header_size));
        }

        // Read header JSON
        std::string header_json(header_size, '\0');
        file.read(&header_json[0], header_size);

        if (!file) {
            throw std::runtime_error("Failed to read header JSON");
        }

        // Parse header
        tensors_ = MinimalJSONParser::parse_header(header_json);
    }

    void load_header_from_buffer() {
        if (buffer_size_ < 8) {
            throw std::runtime_error("Buffer too small for SafeTensors header");
        }

        // Read header size (first 8 bytes, little-endian uint64)
        uint64_t header_size;
        std::memcpy(&header_size, buffer_, 8);

        header_size_ = header_size;

        // Sanity check
        if (header_size > 100 * 1024 * 1024) {
            throw std::runtime_error("Header size too large: " + std::to_string(header_size));
        }

        if (8 + header_size > buffer_size_) {
            throw std::runtime_error("Buffer too small for SafeTensors header JSON");
        }

        // Read header JSON
        std::string header_json(reinterpret_cast<const char*>(buffer_ + 8), header_size);

        // Parse header
        tensors_ = MinimalJSONParser::parse_header(header_json);
    }

    float* load_tensor_from_file(const std::string& name) const {
        const auto& info = get_info(name);

        if (info.dtype != "F32") {
            throw std::runtime_error("Only F32 dtype supported, got: " + info.dtype);
        }

        size_t num_elements = info.num_elements();
        float* data = new float[num_elements];

        // Open file and seek to data
        std::ifstream file(filepath_, std::ios::binary);
        if (!file) {
            delete[] data;
            throw std::runtime_error("Failed to open file: " + filepath_);
        }

        // Seek to tensor data (header_size + 8 + data_begin)
        size_t offset = 8 + header_size_ + info.data_begin;
        file.seekg(offset);

        // Read data
        file.read(reinterpret_cast<char*>(data), num_elements * sizeof(float));

        if (!file) {
            delete[] data;
            throw std::runtime_error("Failed to read tensor data");
        }

        return data;
    }

    float* load_tensor_from_buffer(const std::string& name) const {
        const auto& info = get_info(name);

        if (info.dtype != "F32") {
            throw std::runtime_error("Only F32 dtype supported, got: " + info.dtype);
        }

        size_t num_elements = info.num_elements();
        float* data = new float[num_elements];

        // Calculate absolute offset in buffer
        size_t data_section_offset = 8 + header_size_;
        size_t offset = data_section_offset + info.data_begin;

        // Bounds check
        if (offset + info.num_bytes() > buffer_size_) {
            delete[] data;
            throw std::runtime_error("Tensor data exceeds buffer size");
        }

        // Copy from buffer (already in memory!)
        std::memcpy(data, buffer_ + offset, info.num_bytes());

        return data;
    }

    SourceType source_type_;
    std::string filepath_;
    const uint8_t* buffer_;
    size_t buffer_size_;
    size_t header_size_;
    std::map<std::string, TensorInfo> tensors_;
};

}  // namespace weights
}  // namespace pfalign
