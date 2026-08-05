DROP VIEW IF EXISTS tabfm_backtest_runners;

CREATE VIEW tabfm_backtest_runners AS
WITH complete_races AS (
    SELECT race_id
    FROM race_runners
    GROUP BY race_id
    HAVING COUNT(*) >= 4
       AND MIN(status) = 'finished'
       AND MAX(status) = 'finished'
       AND COUNT(DISTINCT start_time_iso) = 1
       AND MIN(start_time_iso) IS NOT NULL
       AND SUM(CASE WHEN top3_mask = 1 THEN 1 ELSE 0 END) = 3
       AND SUM(CASE WHEN top3_mask = 0 THEN 1 ELSE 0 END) >= 1
       AND SUM(CASE WHEN top3_mask IN (0, 1) THEN 0 ELSE 1 END) = 0
)
SELECT rr.*
FROM race_runners AS rr
JOIN complete_races AS complete USING (race_id);
