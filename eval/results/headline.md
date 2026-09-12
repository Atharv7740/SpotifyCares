# Headline results (n=100)

| system   | intent acc | intent macro-F1 | reply (judge avg) | escalation F1 |
|----------|------------|------------------|-------------------|---------------|
| trivial  | 0.30 | 0.06 | 3.49 | 0.69 |
| strong   | 0.48 | 0.49 | 4.73 | 0.17 |
| ours     | 0.79 | 0.78 | 4.04 | 0.49 |

## Cost & latency

- total cost per 1k decisions (ours): $0.0
- cost is $0 on free tier — break-even is any auto-handle rate > 0

## Per-dimension judge scores (ours)

- helpfulness: 3.71
- groundedness: 4.1
- tone_match: 4.01
- safety: 4.32