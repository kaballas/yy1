UPDATE race_runners
SET competition_id = 999
WHERE race_id IN (
    SELECT race_id
    FROM v_market_top3_training_cohorts_10pct
);
DROP VIEW tabfm_trainable_validation_runners;
CREATE VIEW tabfm_trainable_validation_runners AS
SELECT
    rr.*
FROM race_runners AS rr
WHERE rr.top3_mask IN (0, 1) and competition_id in (279);

DROP VIEW tabfm_validation_runners;
CREATE VIEW tabfm_validation_runners AS
SELECT
    rr.*
FROM race_runners AS rr
WHERE rr.top3_mask IN (0, 1) and competition_id in (
999
);


    select competition_id,competition_name,SUM(prize_money) AS total_prize_money from race_runners
GROUP BY competition_id
order by total_prize_money desc