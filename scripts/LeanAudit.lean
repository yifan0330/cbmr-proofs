import CbmrProofs
import Lean.Util.CollectAxioms

open Lean Elab Command

/-! Reject admissions and extra axioms transitively, not just by text search. -/
run_cmd do
  let env ← getEnv
  let mut checked : Nat := 0
  let allowed := #[``propext, ``Classical.choice, ``Quot.sound]
  for (name, info) in env.constants.toList do
    let projectModule := match env.getModuleIdxFor? name with
      | some index => (`CbmrProofs).isPrefixOf (env.header.moduleNames[index.toNat]!)
      | none => false
    if projectModule || (`CbmrProofs).isPrefixOf name then
      match info with
      | .axiomInfo _ =>
        throwError m!"Project axiom is not permitted: {name}"
      | .thmInfo _ =>
        let axioms ← Lean.collectAxioms name
        let unexpected := axioms.filter fun ax => !allowed.contains ax
        unless unexpected.isEmpty do
          throwError m!"Untrusted theorem {name}: {unexpected}"
        checked := checked + 1
        logInfo m!"CHECKED {name}"
      | _ => pure ()
  if checked == 0 then
    throwError "No CBMR theorems were audited"
  logInfo m!"Lean axiom audit passed for {checked} theorem declarations."
