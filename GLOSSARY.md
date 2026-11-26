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
A selection applied to FLAG on one baseline, recording exactly the cells it turned on; undo reverts one edit.
_Avoid_: flagging operation, change

**Scope**:
Where a selection is applied: one baseline's correlations (all of them, by default, or only those its product is formed from); later, an antenna's baselines or all baselines.
_Avoid_: target, extent

**Backup**:
The flag version a flagging session saves before its first edit.
_Avoid_: snapshot, autosave
