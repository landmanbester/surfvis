# surfvis

Per-baseline diagnostics of a Measurement Set: chi-squared summaries, the
browser over them, and flagging by hand from a baseline's waterfall.

## Manual flagging

**Selection**:
A polygon drawn on one scan's waterfall in (time slot, channel), not yet written to FLAG. A rectangle is a four-vertex selection.
_Avoid_: region, box, shape

**Pending selection**:
A closed selection shown with the cells it covers, awaiting confirmation.
_Avoid_: preview (that is how it is shown, not what it is)

**Edit**:
A selection applied to FLAG over its scope, recording exactly the cells it turned on; undo reverts one edit, whatever its scope.
_Avoid_: flagging operation, change

**Scope**:
Where a selection is applied, always within its scan: which baselines (the one it was drawn on, every baseline of one antenna including its auto-correlation, or every baseline in the scan) and which correlations (all of them, by default, or only those its product is formed from).
_Avoid_: target, extent

**Backup**:
The flag version a flagging session saves before its first edit.
_Avoid_: snapshot, autosave

**Audit log**:
The durable record next to the MS of every edit applied and undone: when, by whom, where, its scope, its selection and the cells it changed.
_Avoid_: history, journal (the journal is what makes one write crash-safe)

**Stale**:
A heatmap cell whose χ² was computed before an edit that touches it, or before the undo of an edit it included: it no longer describes the MS's FLAG.
_Avoid_: outdated, dirty

**Recompute**:
Forming stale cells' χ² again from the Measurement Set, by the same recipe chi2 used, and writing them into the chi2 output in place of the stale values.
_Avoid_: refresh, patch, rerun (rerunning chi2 is something else)
