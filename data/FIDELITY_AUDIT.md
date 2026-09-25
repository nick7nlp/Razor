# RazorCal provenance status

The calibration corpus is `RazorCal.json`. Its `sub_source` fields record
attribution information for the included samples.

## Available information

The released record array has 2,048 samples, seven domains and 15 distinct
source labels. The source-label inventory is listed in [README.md](README.md).
These are properties of the released artifact, not proof of upstream provenance.

## Source matching

The released metadata does not establish an exact upstream record for every
sample. Labels name the upstream dataset, and for slices taken whole -- such as
the 160 `nemotron-cascade2-math` records -- they do not identify a position
within it. No complete source-fidelity claim is made.

Structural consistency, source matching, privacy processing and license review
are separate questions. None can substitute for the others. A source label
does not establish a license or certify the sample's contents.
See [DATASHEET.md](DATASHEET.md) and [LICENSE-DATA](LICENSE-DATA).
