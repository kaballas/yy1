DROP VIEW IF EXISTS tabfm_trainable_validation_runners;

CREATE VIEW tabfm_trainable_validation_runners AS
SELECT
    rr.*
FROM race_runners AS rr
WHERE rr.top3_mask IN (0, 1)
   AND EXISTS (
      SELECT 1
      FROM v_market_top3_complete_misses AS vm
      WHERE vm.race_id = rr.race_id
  );
