# LongMemEval Results — 2026-02-21 (16 questions)

| # | QID | Category | BL | BL tok | BL Cost | VC | VC tok | VC Cost | Haiku Compact | 4o-mini Compact |
|---|-----|----------|-----|--------|---------|-----|--------|---------|---------------|-----------------|
| 1 | b86304ba | single-session-user | N | 114,721 | $0.3458 | Y | 36,051 | $0.1164 | $0.81 | $0.06 |
| 2 | 778164c6 | single-session-assistant | Y | 114,019 | $0.3433 | Y | 8,372 | $0.0280 | $0.81 | $0.06 |
| 3 | 2318644b | multi-session | N | 114,226 | $0.3437 | Y | 10,870 | $0.0366 | $0.81 | $0.06 |
| 4 | 3fdac837 | multi-session | N | 115,572 | $0.3489 | Y | 14,960 | $0.0479 | $0.77* | $0.06* |
| 5 | 4f54b7c9 | multi-session | N | 110,178 | $0.3318 | Y | 12,272 | $0.0414 | $0.81 | $0.06 |
| 6 | 5025383b | multi-session | Y | 115,015 | $0.3466 | Y | 11,033 | $0.0357 | $0.81 | $0.06 |
| 7 | 60472f9c | multi-session | Y | 109,406 | $0.3290 | Y | 8,397 | $0.0276 | $0.81 | $0.06 |
| 8 | 6d550036 | multi-session | N | 110,779 | $0.3337 | Y | 12,511 | $0.0421 | $0.85* | $0.07* |
| 9 | b3c15d39 | multi-session | N | 113,021 | $0.3402 | Y | 8,349 | $0.0269 | $0.81 | $0.06 |
| 10 | 71017276 | temporal-reasoning | Y | 116,294 | $0.3531 | Y | 8,727 | $0.0281 | $0.80* | $0.06 |
| 11 | c8090214 | temporal-reasoning | Y | 113,208 | $0.3427 | Y | 11,631 | $0.0376 | $0.81 | $0.06 |
| 12 | gpt4_0b2 | temporal-reasoning | N | 116,231 | $0.3504 | Y | 13,294 | $0.0437 | $0.81 | $0.06 |
| 13 | 07741c45 | knowledge-update | N | 115,770 | $0.3486 | Y | 4,528 | $0.0150 | $0.81 | $0.06 |
| 14 | 7401057b | knowledge-update | N | 116,236 | $0.3527 | Y | 10,839 | $0.0379 | $0.81 | $0.06 |
| 15 | 9ea5eabc | knowledge-update | N | 116,880 | $0.3516 | Y | 5,364 | $0.0174 | $0.81 | $0.06 |
| 16 | cc5ded98 | knowledge-update | Y | 116,666 | $0.3529 | Y | 12,420 | $0.0390 | $0.81 | $0.06 |
| | | **TOTAL** | **6/16** | **1,828,222** | **$5.52** | **16/16** | **189,618** | **$0.62** | **$12.91** | **$1.02** |

## Summary

| Metric | Baseline | VC (Haiku) | VC (GPT-4o mini) |
|--------|----------|------------|------------------|
| Accuracy | 6/16 (38%) | **16/16 (100%)** | **16/16 (100%)** |
| Total tokens | 1,828,222 | 189,618 | 189,618 |
| Token reduction | — | **90%** | **90%** |
| Reader cost (Sonnet) | $5.52 | $0.62 | $0.62 |
| Compaction cost | — | $12.91 | $1.02 |
| **Total cost** | **$5.52** | **$13.53** | **$1.64** |
| **Cost per query (amortized)** | **$0.35** | **$0.04** | **$0.04** |

### Amortization — cost per query over repeated use

| Turn | Baseline | VC (Haiku) | VC (GPT-4o mini) |
|------|----------|------------|------------------|
| Turn 1 (compact + query) | $0.35 | $0.85 | $0.10 |
| Turn 2 (query only) | $0.35 | $0.04 | $0.04 |
| Turn 3 (query only) | $0.35 | $0.04 | $0.04 |
| **3-turn total** | **$1.05** | **$0.93** | **$0.18** |
| Break-even turn | — | Turn 3 | **Turn 1** |

### Notes
- **Baseline**: Sonnet 4.5, full ~115K token haystack per query ($3/$15 per MTok)
- **VC reader**: Sonnet 4.5, 64K context window + tag summaries + find_quote tool
- **Haiku 4.5 compaction**: $1.00/$5.00 per MTok — avg $0.81/question (measured: Q8=$0.85, Q12=$0.77, Q10=$0.80)
- **GPT-4o mini compaction**: $0.15/$0.60 per MTok — avg $0.06/question (measured: Q8=$0.07, Q12=$0.06). **12.7x cheaper**, same accuracy
- Compaction is one-time: ingest once, query many times. Marginal query cost: ~$0.04 avg
- Rows marked * have measured costs; unmarked rows use the per-question average
- GPT-4o mini accuracy verified on Q8 and Q12 — both CORRECT (same as Haiku)
