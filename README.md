# AI Credibility and Human Verification Behaviour

### How do AI accuracy, confidence cues, and decision importance relate to fact-checking?

**When an AI system appears more credible, are users less likely to verify its answers?**

---

## Overview

AI systems are used increasingly often to answer questions, solve problems, and support decisions. As these systems improve, a practical question follows: when do users decide to verify an AI-generated answer rather than accept it as given?

This project examines the relationship between AI response characteristics, credibility cues, decision context, and user fact-checking behaviour.

Using a practice dataset of 1,000 AI-user interactions, I test whether users were more or less likely to fact-check an AI response depending on:

- AI answer accuracy
- AI-expressed confidence
- presence of cited sources
- hedging language
- disclaimers
- response length
- decision importance

The outcome of interest is whether the user performed a fact-check.

---

## Research question

How do AI response characteristics and decision context relate to users' fact-checking behaviour?

### Hypotheses

**H1:** Higher AI accuracy is associated with a lower probability of fact-checking.

**H2:** Higher AI-expressed confidence is associated with a lower probability of fact-checking.

**H3:** Credibility cues such as citations are associated with reduced verification.

**H4:** Users are more likely to verify information when the decision is more important.

---

## Conceptual framework

```text
                         AI RESPONSE
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
          Accuracy       Confidence       Citations
              |               |               |
              +---------------+---------------+
                              |
                              v
                    PERCEIVED CREDIBILITY
                              |
                              v
                    VERIFICATION DECISION
                       /              \
                      /                \
                FACT-CHECK          NO CHECK

                              ^
                              |
                     DECISION CONTEXT
```

This framework is conceptual. The analysis tests statistical associations, not a causal pathway.

---

## Dataset

N = 1,000 AI-user interactions.

**AI response characteristics**
- Answer accuracy
- AI confidence
- Response length
- Answer detail level

**Credibility cues**
- Cited sources
- Hedging language
- Disclaimer

**User and decision context**
- Decision importance
- AI familiarity
- Digital literacy
- Subject-matter expertise
- Belief alignment
- Urgency

**Behavioural outcomes**
- Whether the user fact-checked
- Verification duration
- Trust score
- User skepticism category

The primary outcome used in the statistical models was `performed_fact_check`.

---

## Analytical approach

Analysis was conducted in Python using pandas and NumPy for data handling, Matplotlib for visualisation, SciPy for statistical testing, statsmodels for logistic regression, and scikit-learn for predictive evaluation.

**Workflow**

```text
Data inspection
      ↓
Exploratory data analysis
      ↓
Fact-checking comparisons
      ↓
Logistic regression
      ↓
Model comparison
      ↓
ROC-AUC evaluation
```

Because the outcome is binary (fact-check vs. no fact-check), logistic regression was used to model the probability of fact-checking as a function of the predictors above.

---

## Key findings

### 1. Higher AI accuracy was associated with less fact-checking

Fact-checking rates decreased across accuracy bands:

| AI accuracy | Fact-check rate |
|---|---:|
| Low (0–50%) | 71.9% |
| Moderate (50–70%) | 68.6% |
| High (70–85%) | 58.5% |
| Very high (85–100%) | 54.2% |

This is a descriptive pattern consistent with H1. It is not causal evidence.

### 2. AI confidence and accuracy were strongly related

Accuracy and expressed confidence were highly correlated (r ≈ 0.86).

Modelled separately, both accuracy and confidence were significant predictors of fact-checking. Modelled together, AI confidence remained significant while accuracy did not retain an independent association. This indicates that accuracy and expressed confidence carry substantial overlapping information in this dataset, not that users were "fooled" by confidence — the data are observational and cannot distinguish between these explanations.

### 3. Decision importance was strongly associated with verification

| Decision importance | Fact-check rate |
|---|---:|
| Low | 47.5% |
| Medium | 51.6% |
| High | 70.1% |
| Critical | 84.1% |

Verification rates rose sharply with the stakes of the decision, consistent with H4.

### 4. Credibility cues showed mixed patterns

| AI response feature | Association with fact-checking |
|---|---|
| Higher AI confidence | Lower verification |
| Cited sources | Lower verification |
| Hedging language | No clear association |
| Disclaimer | No clear association |
| Response length | No clear association |

In the full regression model, AI confidence and cited sources remained statistically significant predictors, while accuracy no longer had an independent effect once the other response characteristics and decision importance were included.

---

## Regression model

```text
Fact-checking ~
    Accuracy
    + AI confidence
    + Cited sources
    + Hedging
    + Disclaimer
    + Response length
    + Decision importance
```

**Model performance:** ROC-AUC = 0.712, indicating moderate ability to distinguish fact-checking from non-fact-checking interactions. This should not be read as 71.2% prediction accuracy.

---
## Canva Presentation Link
https://canva.link/a9i5co23pljy43p

**Author:** Nitya Joshi — Psychology graduate, MSc Social Cognition. Interested in human-AI interaction, cognition, behavioural science, and human-centred AI design.
