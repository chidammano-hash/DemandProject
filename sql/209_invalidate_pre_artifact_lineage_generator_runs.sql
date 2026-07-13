-- Retire ready candidates produced before the immutable canonical-five
-- artifact/roster/snapshot lineage contract. Rows remain audit evidence, but
-- neither release candidates nor snapshot contenders may be reused.

BEGIN;

UPDATE forecast_generation_run
SET run_status = 'invalid',
    promotion_eligible = FALSE,
    invalid_reason = 'generation predates canonical five artifact-lineage contract',
    completed_at = COALESCE(completed_at, NOW())
WHERE generation_purpose IN ('release_candidate', 'snapshot_contender')
  AND run_status = 'ready'
  AND metadata ->> 'generator_contract_version'
      IS DISTINCT FROM 'canonical-five-artifact-lineage-v2';

COMMIT;
