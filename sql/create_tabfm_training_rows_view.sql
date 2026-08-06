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
WHERE rr.top3_mask IN (0, 1) and competition_id in (231,
271,
275,
293,
317,
335,
353,
358,
371,
486,
488,
498,
526,
554,
556,
570,
573,
577,
585,
588,
590,
600,
602,
622,
661,
696,
24168,
24176,
27205,
29418,
30188
);

DROP VIEW tabfm_validation_runners;
CREATE VIEW tabfm_validation_runners AS
SELECT
    rr.*
FROM race_runners AS rr
WHERE rr.top3_mask IN (0, 1) and competition_id not in (231,
271,
275,
293,
317,
335,
353,
358,
371,
486,
488,
498,
526,
554,
556,
570,
573,
577,
585,
588,
590,
600,
602,
622,
661,
696,
24168,
24176,
27205,
29418,
30188
);;


    