#pragma once

// Minimal CLI library for PFalign
// Inspired by CLI11 but 78% smaller (~2,800 LOC vs ~12,825 LOC)
//
// Features:
// - Subcommands (1-level: pfalign <encode|similarity|align|full>)
// - Positional arguments with validation
// - Optional flags (--gap-open, --mode, etc.)
// - Type parsing (string, int, float)
// - Validators (ExistingFile, Range)
// - Automatic help generation
// - Clear error messages
//
// Missing (vs CLI11):
// - Config file support
// - Environment variables
// - Nested subcommands
// - Complex validators
// - Windows-style options
// - Unicode/encoding

#include "errors.h"
#include "types.h"
#include "validators.h"
#include "option.h"
#include "app.h"
#include "formatter.h"

namespace pfalign {
namespace cli {

// Convenience macro for parse + error handling
#define PFALIGN_PARSE(app, argc, argv)       \
    try {                                    \
        app.parse(argc, argv);               \
    } catch (const pfalign::cli::Error& e) { \
        return app.exit(e);                  \
    }

}  // namespace cli
}  // namespace pfalign
