-- Retire ready release candidates produced before the canonical five-model
-- production adapters were enforced. Their staging rows remain immutable for
-- audit, but they must not be promoted under a canonical model label.

BEGIN;

UPDATE forecast_generation_run
SET run_status = 'invalid',
    promotion_eligible = FALSE,
    invalid_reason = 'release candidate predates canonical five-model generator contract',
    completed_at = COALESCE(completed_at, NOW())
WHERE generation_purpose = 'release_candidate'
  AND run_status = 'ready'
  AND metadata ->> 'generator_contract_version'
      IS DISTINCT FROM 'canonical-five-real-adapters-v1';

COMMIT;
