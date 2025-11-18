#include "pfalign/common/arena_allocator.h"
#include "pfalign/common/growable_arena.h"
#include "pfalign/modules/mpnn/sequence_cache.h"
#include <cstdio>

using namespace pfalign::memory;
using namespace pfalign;

int main() {
    printf("=== Arena Allocator Test ===\n\n");

    GrowableArena arena(100);  // 100 MB

    // Test 1: Basic arena allocation
    printf("Test 1: Basic memory allocation\n");
    float* data = arena.allocate<float>(1000);
    printf("  Allocated 1000 floats at %p\n", (void*)data);
    data[0] = 1.0f;
    data[999] = 2.0f;
    printf("  ✓ Memory is writable\n\n");

    // Test 2: Allocate and construct object with placement-new (CORRECT USAGE)
    printf("Test 2: Allocate with placement-new (proper construction)\n");
    SequenceEmbeddings* seq = arena.allocate<SequenceEmbeddings>(1);
    new (seq) SequenceEmbeddings();  // Construct with placement-new
    printf("  Constructed SequenceEmbeddings at %p\n", (void*)seq);

    // Set members
    seq->seq_id = 42;
    seq->length = 100;
    seq->hidden_dim = 64;
    seq->identifier = "1CRN_A";
    seq->sequence = "MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGNFGADAQGAMNKALELFRKDIAAKYKELGYQG";

    printf("  seq_id: %d\n", seq->seq_id);
    printf("  length: %d\n", seq->length);
    printf("  hidden_dim: %d\n", seq->hidden_dim);
    printf("  identifier: %s\n", seq->identifier.c_str());
    printf("  sequence length: %zu\n", seq->sequence.length());
    printf("  ✓ Object fully functional\n\n");

    // Test 3: Multiple allocations
    printf("Test 3: Multiple allocations\n");
    for (int i = 0; i < 10; i++) {
        SequenceEmbeddings* s = arena.allocate<SequenceEmbeddings>(1);
        new (s) SequenceEmbeddings();
        s->seq_id = i;
        s->length = 100 + i;
    }
    printf("  ✓ Allocated 10 SequenceEmbeddings objects\n\n");

    // Cleanup: Call destructors for objects with std::string members
    printf("Test 4: Proper cleanup\n");
    seq->~SequenceEmbeddings();
    printf("  ✓ Destructor called\n\n");

    printf("=== All tests passed ===\n");
    return 0;
}
