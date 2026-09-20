#ai accuracy, ai credibility and fact-checking behaviour analysis 
#exploratory data analysis

#data was cleaned and preprocessed in a separate notebook before this analysis

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv('/Users/nityajoshi/Downloads/ai_skepticism_dataset.csv')

print(df.head())
print(df.info())
print(df.describe())

sns.histplot(data=df, x="ai_model_name")
plt.show()

from scipy import stats

df["trust_score_out_of_10"].hist(bins=20)

plt.xlabel("Trust Score")
plt.ylabel("Frequency")
plt.title("Distribution of Trust Scores")

plt.show()

plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Numeric Variables")
plt.show()

#fact-checking behaviour analysis

fact_checkers = df[df["performed_fact_check"] == True]
non_fact_checkers = df[df["performed_fact_check"] == False]
print("Fact-checkers:", len(fact_checkers))
print("Non-fact-checkers:", len(non_fact_checkers))

#fact-checking behaviour across AI accuracy levels
df["accuracy_group"] = pd.cut(
    df["answer_accuracy_percentage"],
    bins=[0, 50, 70, 85, 100],
    labels=[
        "Low (0-50%)",
        "Moderate (50-70%)",
        "High (70-85%)",
        "Very High (85-100%)"
    ]
)

accuracy_factcheck = df.groupby(
    "accuracy_group",
    observed=False
)["performed_fact_check"].mean() * 100

print(accuracy_factcheck)

accuracy_factcheck.plot(
    kind="bar"
)

plt.title("Fact-Checking Behaviour Across AI Accuracy Levels")
plt.xlabel("AI Response Accuracy")
plt.ylabel("Percentage Who Fact-Checked")

plt.show()

sns.regplot(
    data=df,
    x="answer_accuracy_percentage",
    y="performed_fact_check",
    logistic=True,
    scatter_kws={"alpha": 0.3}
)

plt.title("Probability of Fact-Checking Across AI Accuracy")
plt.xlabel("AI Answer Accuracy (%)")
plt.ylabel("Probability of Fact-Checking")

plt.show()

#fact-checking behaviour across AI confidence levels
sns.boxplot(
    data=df,
    x="performed_fact_check",
    y="ai_confidence_percentage"
)

plt.title("AI Confidence and User Fact-Checking")
plt.xlabel("Performed Fact Check")
plt.ylabel("AI Expressed Confidence (%)")

plt.show()

df.groupby("performed_fact_check")[
    "ai_confidence_percentage"
].mean()

#fact-checking behaviour across hedging language, disclaimers and response length
hedging_factcheck = df.groupby(
    "contains_hedging_words"
)["performed_fact_check"].mean() * 100

hedging_factcheck.plot(kind="bar")

plt.title("Hedging Language and Fact-Checking")
plt.xlabel("Contains Hedging Language")
plt.ylabel("Percentage Who Fact-Checked")

plt.show()

disclaimer_factcheck = df.groupby(
    "includes_disclaimer"
)["performed_fact_check"].mean() * 100

disclaimer_factcheck.plot(kind="bar")

plt.title("Disclaimers and Fact-Checking Behaviour")
plt.xlabel("Includes Disclaimer")
plt.ylabel("Percentage Who Fact-Checked")

plt.show()

sns.boxplot(
    data=df,
    x="performed_fact_check",
    y="response_character_count"
)

plt.title("Response Length and Fact-Checking")
plt.xlabel("Performed Fact Check")
plt.ylabel("Response Character Count")

plt.show()

sns.regplot(
    data=df,
    x="response_character_count",
    y="performed_fact_check",
    logistic=True,
    scatter_kws={"alpha": 0.2}
)

plt.title("Response Length and Probability of Fact-Checking")

plt.show()

#Regression analysis 

predictors = [
    "answer_accuracy_percentage",
    "ai_confidence_percentage",
    "has_cited_sources",
    "contains_hedging_words",
    "includes_disclaimer",
    "response_character_count"
]

analysis_df = pd.get_dummies(
    df[
        predictors + [
            "answer_detail_level",
            "performed_fact_check"
        ]
    ],
    drop_first=True
)

import statsmodels.api as sm

X = analysis_df.drop(
    "performed_fact_check",
    axis=1
)

y = analysis_df["performed_fact_check"]
X = sm.add_constant(X)
model = sm.Logit(
    y,
    X
).fit()

print(model.summary())


df["confidence_accuracy_gap"] = (
    df["ai_confidence_percentage"]
    -
    df["answer_accuracy_percentage"]
)
sns.scatterplot(
    data=df,
    x="answer_accuracy_percentage",
    y="ai_confidence_percentage",
    hue="performed_fact_check",
    alpha=0.6
)

plt.plot(
    [0, 100],
    [0, 100],
    linestyle="--"
)

plt.xlabel("Actual Answer Accuracy (%)")
plt.ylabel("AI Expressed Confidence (%)")

plt.title(
    "AI Confidence vs Actual Accuracy by Fact-Checking Behaviour"
)

plt.show()

# second round of analysis after feedback


numeric_cols = [
    "ai_confidence_percentage",
    "response_character_count",
    "trust_score_out_of_10",
    "verification_duration_mins",
    "answer_accuracy_percentage"
]

df[numeric_cols].describe().T

for col in numeric_cols:
    plt.figure(figsize=(7,4))
    plt.hist(df[col], bins=25)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

    df["accuracy_bin"] = pd.cut(
    df["answer_accuracy_percentage"],
    bins=10
)

#fact-checking behaviour across AI accuracy levels

factcheck_by_accuracy = (
    df.groupby("accuracy_bin", observed=False)
      ["performed_fact_check"]
      .mean()
      * 100
)

print(factcheck_by_accuracy)

plt.figure(figsize=(9,5))

plt.plot(
    range(len(factcheck_by_accuracy)),
    factcheck_by_accuracy,
    marker="o"
)

plt.xticks(
    range(len(factcheck_by_accuracy)),
    factcheck_by_accuracy.index.astype(str),
    rotation=45,
    ha="right"
)

plt.xlabel("AI Accuracy")
plt.ylabel("% Who Fact-Checked")
plt.title("Fact-Checking Across AI Accuracy Levels")

plt.tight_layout()
plt.show()

#fact-checking behaviour across AI confidence levels

df["confidence_bin"] = pd.cut(
    df["ai_confidence_percentage"],
    bins=10
)

factcheck_by_confidence = (
    df.groupby("confidence_bin", observed=False)
      ["performed_fact_check"]
      .mean()
      * 100
)

print(factcheck_by_confidence)
plt.figure(figsize=(9,5))
plt.plot(
    range(len(factcheck_by_confidence)),
    factcheck_by_confidence,
    marker="o"
)

plt.xticks(
    range(len(factcheck_by_confidence)),
    factcheck_by_confidence.index.astype(str),
    rotation=45,
    ha="right"
)

plt.xlabel("AI Confidence")
plt.ylabel("% Who Fact-Checked")
plt.title("Fact-Checking Across AI Confidence Levels")

plt.tight_layout()
plt.show()
#verification of sample sizes across accuracy bins

accuracy_summary = df.groupby(
    "accuracy_bin",
    observed=False
).agg(
    n=("performed_fact_check", "size"),
    fact_check_rate=("performed_fact_check", "mean")
)

accuracy_summary["fact_check_rate"] *= 100
print(accuracy_summary)

#comparison of fact-checkers and non-fact-checkers

comparison = df.groupby("performed_fact_check")[
    [
        "answer_accuracy_percentage",
        "ai_confidence_percentage",
        "trust_score_out_of_10",
        "response_character_count",
        "verification_duration_mins"
    ]
].agg(["mean", "median", "std"])

print(comparison)

#how ai familiarity level affects fact-checking behaviour

familiarity_table = pd.crosstab(
    df["ai_familiarity_level"],
    df["performed_fact_check"],
    normalize="index"
) * 100

print(familiarity_table)

pd.set_option("display.max_columns", None)

print(comparison)

#distribution of trust scores, ai confidence and response length among fact-checkers and non-fact-checkers
df.groupby("performed_fact_check")[
    "trust_score_out_of_10"
].agg(["mean", "median", "std"])

plt.figure(figsize=(7,5))

sns.boxplot(
    data=df,
    x="performed_fact_check",
    y="trust_score_out_of_10"
)

plt.xlabel("Performed Fact Check")
plt.ylabel("Trust Score (0–10)")
plt.title("Trust Among Fact-Checkers and Non-Fact-Checkers")

plt.show()

df.groupby("performed_fact_check")[
    "ai_confidence_percentage"
].agg(["mean", "median", "std"])

plt.figure(figsize=(7,5))

sns.boxplot(
    data=df,
    x="performed_fact_check",
    y="ai_confidence_percentage"
)

plt.xlabel("Performed Fact Check")
plt.ylabel("AI Expressed Confidence (%)")
plt.title("AI Confidence Among Fact-Checkers and Non-Fact-Checkers")

plt.show()

df.groupby("performed_fact_check")[
    "response_character_count"
].agg(["mean", "median", "std"])
plt.figure(figsize=(7,5))

sns.boxplot(
    data=df,
    x="performed_fact_check",
    y="response_character_count"
)

plt.xlabel("Performed Fact Check")
plt.ylabel("Response Length (characters)")
plt.title("Response Length Among Fact-Checkers and Non-Fact-Checkers")

plt.show()

#fact-checking behaviour across cited sources, hedging language and disclaimers

for variable in [
    "has_cited_sources",
    "contains_hedging_words",
    "includes_disclaimer"
]:
    
    print("\n", variable)
    
    table = pd.crosstab(
        df[variable],
        df["performed_fact_check"],
        normalize="index"
    ) * 100
    
    print(table)

#fact-checking behaviour across user skepticism, decision importance and subject matter expertise

skepticism_table = pd.crosstab(
    df["user_skepticism_category"],
    df["performed_fact_check"],
    normalize="index"
) * 100

print(skepticism_table)

importance_table = pd.crosstab(
    df["decision_importance"],
    df["performed_fact_check"],
    normalize="index"
) * 100

print(importance_table)

expertise_table = pd.crosstab(
    df["subject_matter_expertise"],
    df["performed_fact_check"],
    normalize="index"
) * 100

print(expertise_table)

print(df["user_skepticism_category"].value_counts())

print(
    pd.crosstab(
        df["user_skepticism_category"],
        df["performed_fact_check"]
    )
)

