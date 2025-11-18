#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <cctype>

namespace pfalign {
namespace io {

inline const std::unordered_map<std::string, char>& amino_map() {
    static const std::unordered_map<std::string, char> kMap = {
        {"Ala", 'A'}, {"ALA", 'A'}, {"Arg", 'R'}, {"ARG", 'R'}, {"Asn", 'N'}, {"ASN", 'N'},
        {"Asp", 'D'}, {"ASP", 'D'}, {"Cys", 'C'}, {"CYS", 'C'}, {"Gln", 'Q'}, {"GLN", 'Q'},
        {"Glu", 'E'}, {"GLU", 'E'}, {"Gly", 'G'}, {"GLY", 'G'}, {"His", 'H'}, {"HIS", 'H'},
        {"Ile", 'I'}, {"ILE", 'I'}, {"Leu", 'L'}, {"LEU", 'L'}, {"Lys", 'K'}, {"LYS", 'K'},
        {"Met", 'M'}, {"MET", 'M'}, {"Phe", 'F'}, {"PHE", 'F'}, {"Pro", 'P'}, {"PRO", 'P'},
        {"Ser", 'S'}, {"SER", 'S'}, {"Thr", 'T'}, {"THR", 'T'}, {"Trp", 'W'}, {"TRP", 'W'},
        {"Tyr", 'Y'}, {"TYR", 'Y'}, {"Val", 'V'}, {"VAL", 'V'}, {"Asx", 'B'}, {"ASX", 'B'},
        {"Glx", 'Z'}, {"GLX", 'Z'}, {"Xaa", 'X'}, {"XAA", 'X'}, {"Pyl", 'O'}, {"PYL", 'O'},
        {"Sec", 'U'}, {"SEC", 'U'}, {"Xle", 'J'}, {"XLE", 'J'}};
    return kMap;
}

inline char three_to_one(std::string_view three_letter) {
    const auto& map = amino_map();
    auto it = map.find(std::string(three_letter));
    if (it != map.end()) {
        return it->second;
    }

    if (three_letter.size() == 3) {
        std::string key;
        key.reserve(3);
        key.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(three_letter[0]))));
        key.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(three_letter[1]))));
        key.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(three_letter[2]))));
        it = map.find(key);
        if (it != map.end()) {
            return it->second;
        }
    }

    return 'X';
}

}  // namespace io
}  // namespace pfalign
