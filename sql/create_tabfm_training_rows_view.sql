UPDATE race_runners
SET competition_id = 999
WHERE race_id IN (
    SELECT race_id
    FROM v_market_top3_complete_misses
);
DROP VIEW tabfm_trainable_validation_runners;
CREATE VIEW tabfm_trainable_validation_runners AS
SELECT
    rr.*
FROM race_runners AS rr
WHERE rr.top3_mask IN (0, 1) and competition_id = 999;

DROP VIEW tabfm_validation_runners;
CREATE VIEW tabfm_validation_runners AS
SELECT
    rr.*
FROM race_runners AS rr
WHERE rr.top3_mask IN (0, 1) and competition_id = 580;


    