# IO parsers (PDB and mmCIF) and writers (FASTA, PDB)

# FASTA writer library
fasta_writer_lib = static_library('fasta_writer',
  'fasta_writer.cpp',
  include_directories: [inc_pfalign],
  install: false,
)

fasta_writer_dep = declare_dependency(
  link_with: fasta_writer_lib,
  include_directories: [inc_pfalign],
)

# PDB writer library
pdb_writer_lib = static_library('pdb_writer',
  'pdb_writer.cpp',
  include_directories: [inc_pfalign],
  install: false,
)

pdb_writer_dep = declare_dependency(
  link_with: pdb_writer_lib,
  include_directories: [inc_pfalign],
)

# Alignment format conversion library
alignment_formats_lib = static_library('alignment_formats',
  'alignment_formats.cpp',
  'alignment_transform.cpp',
  include_directories: [inc_pfalign],
  install: false,
)

alignment_formats_dep = declare_dependency(
  link_with: alignment_formats_lib,
  include_directories: [inc_pfalign],
)
