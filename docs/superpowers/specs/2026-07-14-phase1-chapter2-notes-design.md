# Phase 1 Chapter 2 Notes Design

## Goal

Expand `quant/docs/phase1-outline.md` section "四、第二章：收益率与数据分析" into a self-contained beginner tutorial based on `02_你的第一个量化实验.ipynb`.

## Audience

The reader has basic Python and Jupyter experience but no prior knowledge of financial data analysis. New terms must be introduced with everyday examples before formulas or APIs.

## Scope

The chapter will cover:

1. Learning goals and the chapter's analysis workflow.
2. OHLCV fields and why closing prices are commonly used.
3. Simple returns using a 100-to-110 yuan example and the daily-return formula.
4. Daily returns with pandas `pct_change()`, including why the first result is `NaN`.
5. Return time-series plots and histograms, including how to read each chart.
6. Cumulative returns with `(1 + r).cumprod() - 1` and a small numeric example.
7. Volatility comparison using the standard deviation of AAPL, TSLA, and NVDA daily returns.
8. Result interpretation, common beginner mistakes, exercises, summary, and self-check questions.

## Content Rules

- Preserve the user's existing uncommitted edits, including the OHLCV table.
- Do not change Chapter 1 or later chapters.
- Reuse the notebook's concepts and code, but explain code in short steps rather than dumping unexplained cells.
- Treat downloaded market values as examples that change with the execution date.
- Do not claim that historical returns predict future returns.
- Explain that this chapter uses unadjusted `Close` data from the notebook and briefly distinguish it from adjusted prices without adding unsupported code.
- Use `> **待补充：**` markers for screenshots, current run results, and the learner's own observations.
- Keep headings and terminology consistent with the existing Chinese course notes.

## Planned Structure

1. 本章目标
2. 2.1 认识股票行情数据：OHLCV
3. 2.2 什么是收益率
4. 2.3 使用 pandas 计算日收益率
5. 2.4 用图看懂收益率
6. 2.5 累计收益率
7. 2.6 小实验：谁的波动更大
8. 2.7 如何解读实验结果
9. 常见误区
10. 实操任务
11. 本章总结
12. 自查题

## Validation

- Confirm Markdown heading structure remains valid.
- Confirm all referenced image placeholders are clearly marked for manual insertion.
- Confirm formulas match the notebook implementation.
- Confirm every listed notebook concept appears in the expanded chapter.
- Review the final diff to ensure unrelated user changes remain untouched.
