DROP VIEW IF EXISTS v_market_top3_complete_misses;

CREATE VIEW v_market_top3_complete_misses AS
WITH eligible AS (
    SELECT
        race_id,
        race_number,
        race_name,
        competition_id,
        competition_name,
        start_time_iso,
        runner_number,
        runner_name,
        fluc2,
        finish_place,
        top3_mask,
        COUNT(*) OVER (PARTITION BY race_id) AS runner_count,
        SUM(CASE WHEN fluc2 > 0 THEN 1 ELSE 0 END)
            OVER (PARTITION BY race_id) AS valid_fluc2_count
    FROM race_runners
    WHERE is_trainable = 1
      AND runner_mask = 1
),
ranked AS (
    SELECT
        *,
        CASE
            WHEN fluc2 > 0 THEN
                ROW_NUMBER() OVER (
                    PARTITION BY race_id
                    ORDER BY
                        CASE WHEN fluc2 > 0 THEN 0 ELSE 1 END,
                        fluc2,
                        runner_number
                )
        END AS market_rank
    FROM eligible
),
race_summary AS (
    SELECT
        race_id,
        MAX(race_number) AS race_number,
        MAX(race_name) AS race_name,
        MAX(competition_id) AS competition_id,
        MAX(competition_name) AS competition_name,
        MAX(start_time_iso) AS start_time_iso,
        MAX(runner_count) AS runner_count,
        MAX(valid_fluc2_count) AS valid_fluc2_count,
        MAX(CASE WHEN market_rank = 1 THEN runner_number END) AS market_top1_number,
        MAX(CASE WHEN market_rank = 1 THEN runner_name END) AS market_top1_name,
        MAX(CASE WHEN market_rank = 1 THEN fluc2 END) AS market_top1_fluc2,
        MAX(CASE WHEN market_rank = 2 THEN runner_number END) AS market_top2_number,
        MAX(CASE WHEN market_rank = 2 THEN runner_name END) AS market_top2_name,
        MAX(CASE WHEN market_rank = 2 THEN fluc2 END) AS market_top2_fluc2,
        MAX(CASE WHEN market_rank = 3 THEN runner_number END) AS market_top3_number,
        MAX(CASE WHEN market_rank = 3 THEN runner_name END) AS market_top3_name,
        MAX(CASE WHEN market_rank = 3 THEN fluc2 END) AS market_top3_fluc2,
        MAX(CASE WHEN finish_place = 1 THEN runner_number END) AS actual_first_number,
        MAX(CASE WHEN finish_place = 1 THEN runner_name END) AS actual_first_name,
        MAX(CASE WHEN finish_place = 2 THEN runner_number END) AS actual_second_number,
        MAX(CASE WHEN finish_place = 2 THEN runner_name END) AS actual_second_name,
        MAX(CASE WHEN finish_place = 3 THEN runner_number END) AS actual_third_number,
        MAX(CASE WHEN finish_place = 3 THEN runner_name END) AS actual_third_name,
        SUM(CASE WHEN top3_mask = 1 THEN 1 ELSE 0 END) AS actual_top3_count,
        SUM(
            CASE
                WHEN market_rank <= 3 AND top3_mask = 1 THEN 1
                ELSE 0
            END
        ) AS market_top3_actual_overlap
    FROM ranked
    GROUP BY race_id
)
SELECT
    race_id,
    race_number,
    race_name,
    competition_id,
    competition_name,
    start_time_iso,
    runner_count,
    valid_fluc2_count,
    market_top1_number,
    market_top1_name,
    market_top1_fluc2,
    market_top2_number,
    market_top2_name,
    market_top2_fluc2,
    market_top3_number,
    market_top3_name,
    market_top3_fluc2,
    actual_first_number,
    actual_first_name,
    actual_second_number,
    actual_second_name,
    actual_third_number,
    actual_third_name,
    market_top3_actual_overlap
FROM race_summary
WHERE runner_count > 7
  AND valid_fluc2_count >= 3
  AND actual_top3_count = 3
  AND market_top3_actual_overlap = 0;
