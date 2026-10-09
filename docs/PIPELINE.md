# Pipeline notes and known caveats

## Alignment granularity differs between cohorts

D1NAMO: the anchor is the START timestamp of each ECG recording. The reference
reading nearest to t_start + tau within the centred window is selected, and every
qualifying window from that recording inherits it. One glucose event per aligned
recording. A single reading therefore labels a mean of 12.1 hours of ECG
(maximum 16.8 h), and the nominal tau = +5 min offset holds exactly only near
the start of each recording.

Consequence: the type 1 diabetes subcohort wore a 5-minute CGM and has 909-1,438
readings per participant, of which 3,892 co-occur with ECG; 32 become events.
D1NAMO's label sparsity is therefore a property of this implementation as much
as of its daytime-only recording coverage, and the D1NAMO arm carries label
noise independent of leakage.

PhysioCGM: the anchor is each CGM reading. Up to four non-overlapping windows
are drawn from within its own +-5 min interval, so the bound holds for every
window. Every conclusion in the paper is reproduced independently on this cohort.

## Normalisation scope
Per recording on D1NAMO, per window on PhysioCGM. Neither crosses a subject
boundary, so neither leaks into the subject-level evaluation.

## Window overlap
50% on D1NAMO (15 s step), none on PhysioCGM (30 s step). Window overlap is a
leakage pathway present on one cohort and absent on the other; this is what makes
the two-channel result in Section "Evaluation Protocol" possible.

## Randomness
Seed 42 throughout. Random number use is confined to: the label-permutation
control, window subsampling (which real windows to keep), the PhysioCGM
per-reading window draw, and bootstrap resampling. No data are generated.
