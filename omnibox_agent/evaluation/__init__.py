"""v4.1 Evaluation framework: eval set schema, runner, and PR curve calibration.

Design doc §5.4:
  - Build 50-100 annotated query eval set
  - Each query annotated with relevant content_ids
  - Run gate on eval set → PR curve → calibrate θ
  - Eval set doubles as regression test

Usage:
  from omnibox_agent.evaluation import EvalSet, EvalRunner

  # Load or create eval set
  eval_set = EvalSet.load("eval_sets/default.json")
  # or create manually
  eval_set = EvalSet(queries=[
      EvalQuery(query="上海美食推荐", expected_content_ids=[123, 456], label="food"),
      ...
  ])

  # Run evaluation
  runner = EvalRunner()
  results = runner.run(eval_set)

  # Calibrate threshold
  pr = runner.pr_curve(results)
  theta = runner.calibrate_theta(pr, target_precision=0.8)
"""
