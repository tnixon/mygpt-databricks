-- SJ Global: the original (keyword-hinted) questions vs the reworded Q6-10 variants.
SELECT dataset_label, model_alias, agent_version,
       CASE metric WHEN 'yes_no_correct/sme/mean' THEN 'Original wording (SME-labelled)'
                   ELSE 'Reworded, no keyword hints (Q6-10)' END AS wording,
       round(value, 3) AS yes_no_correct
FROM ours
WHERE rk = 1 AND use_case = 'sj_global'
  AND metric IN ('yes_no_correct/sme/mean', 'yes_no_correct/inherited/mean')
