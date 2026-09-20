# regression 

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv("/Users/nityajoshi/Downloads/ai_skepticism_dataset.csv")


df["fact_check"] = df["performed_fact_check"].astype(int)

# Model 1: AI accuracy only
model_accuracy = smf.logit(
    "fact_check ~ answer_accuracy_percentage",
    data=df
).fit()

print(model_accuracy.summary())

# Model 2: AI confidence only
model_confidence = smf.logit(
    "fact_check ~ ai_confidence_percentage",
    data=df
).fit()

print(model_confidence.summary())

# Model 3: AI accuracy and AI confidence
model_acc_conf = smf.logit(
    "fact_check ~ answer_accuracy_percentage + ai_confidence_percentage",
    data=df
).fit()

print(model_acc_conf.summary())

#calculating VIF for the predictors in Model 3 (answer_accuracy_percentage and ai_confidence_percentage)

from statsmodels.stats.outliers_influence import variance_inflation_factor

X = df[
    ["answer_accuracy_percentage",
     "ai_confidence_percentage"]
].copy()

X = sm.add_constant(X)

vif = pd.DataFrame()
vif["variable"] = X.columns
vif["VIF"] = [
    variance_inflation_factor(X.values, i)
    for i in range(X.shape[1])
]

print(vif)

# Model 4: Adding credibility cues - cited sources, hedging language, and disclaimers
model_credibility = smf.logit(
    "fact_check ~ has_cited_sources + contains_hedging_words + includes_disclaimer",
    data=df
).fit()

print(model_credibility.summary())

# Model 5: Adding response length
model_full_response = smf.logit(
    "fact_check ~ answer_accuracy_percentage + "
    "ai_confidence_percentage + "
    "has_cited_sources + "
    "contains_hedging_words + "
    "includes_disclaimer + "
    "response_character_count",
    data=df
).fit()

print(model_full_response.summary())

# Model 6: Adding decision importance
model_context = smf.logit(
    "fact_check ~ answer_accuracy_percentage + "
    "ai_confidence_percentage + "
    "has_cited_sources + "
    "contains_hedging_words + "
    "includes_disclaimer + "
    "response_character_count + "
    "C(decision_importance)",
    data=df
).fit()

print(model_context.summary())

# Model 7: Interaction between AI confidence and decision importance
model_interaction = smf.logit(
    "fact_check ~ answer_accuracy_percentage + "
    "ai_confidence_percentage * C(decision_importance) + "
    "has_cited_sources + "
    "contains_hedging_words + "
    "includes_disclaimer + "
    "response_character_count",
    data=df
).fit()

print(model_interaction.summary())

#predicting probabilities of fact-checking across AI confidence levels for different decision importance levels
import pandas as pd

prediction_data = pd.DataFrame({
    "ai_confidence_percentage": [50, 70, 90] * 4,
    "answer_accuracy_percentage": [70] * 12,
    "has_cited_sources": [False] * 12,
    "contains_hedging_words": [False] * 12,
    "includes_disclaimer": [False] * 12,
    "response_character_count": [400] * 12,
    "decision_importance": (
        ["Critical"] * 3 +
        ["High"] * 3 +
        ["Medium"] * 3 +
        ["Low"] * 3
    )
})

prediction_data["predicted_probability"] = (
    model_interaction.predict(prediction_data)
)

print(prediction_data)

import matplotlib.pyplot as plt


plot_data = prediction_data.copy()


for importance in ["Critical", "High", "Medium", "Low"]:
    subset = plot_data[
        plot_data["decision_importance"] == importance
    ]
    
    plt.plot(
        subset["ai_confidence_percentage"],
        subset["predicted_probability"],
        marker="o",
        label=importance
    )

plt.xlabel("AI confidence (%)")
plt.ylabel("Predicted probability of fact-checking")
plt.title("Predicted fact-checking probability across AI confidence\nby decision importance")
plt.legend(title="Decision importance")
plt.ylim(0, 1)
plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


confidence_values = np.linspace(40, 99, 100)


plot_data = pd.DataFrame({
    "ai_confidence_percentage": np.tile(confidence_values, 4),
    "answer_accuracy_percentage": 70,
    "has_cited_sources": False,
    "contains_hedging_words": False,
    "includes_disclaimer": False,
    "response_character_count": 400,
    "decision_importance": np.repeat(
        ["Critical", "High", "Medium", "Low"],
        len(confidence_values)
    )
})


plot_data["predicted_probability"] = model_interaction.predict(plot_data)


plt.figure(figsize=(9, 6))

for importance in ["Critical", "High", "Medium", "Low"]:
    subset = plot_data[
        plot_data["decision_importance"] == importance
    ]

    plt.plot(
        subset["ai_confidence_percentage"],
        subset["predicted_probability"],
        label=importance
    )

plt.xlabel("AI confidence (%)")
plt.ylabel("Predicted probability of fact-checking")
plt.title(
    "Predicted fact-checking probability across AI confidence\n"
    "by decision importance"
)

plt.ylim(0, 1)
plt.legend(title="Decision importance")
plt.tight_layout()
plt.show()

#testing the significance of the interaction term between AI confidence and decision importance using a likelihood ratio test
import scipy.stats as stats

# Likelihood-ratio statistic
lr_stat = 2 * (model_interaction.llf - model_context.llf)

# Difference in number of parameters
df_diff = model_interaction.df_model - model_context.df_model

# p-value
p_value = stats.chi2.sf(lr_stat, df_diff)

print("Likelihood Ratio Statistic:", lr_stat)
print("Degrees of Freedom:", df_diff)
print("p-value:", p_value)

#evaluative tests and correlation analysis
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


y_true = df["fact_check"]

y_pred = model_context.predict(df)

# Calculate AUC
auc = roc_auc_score(y_true, y_pred)

print("ROC-AUC:", auc)

print(
    df[
        [
            "answer_accuracy_percentage",
            "ai_confidence_percentage"
        ]
    ].corr()
)

#how does ai accuracy align with ai confidence 
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.scatter(
    df["answer_accuracy_percentage"],
    df["ai_confidence_percentage"],
    alpha=0.4
)

plt.xlabel("Answer accuracy (%)")
plt.ylabel("AI confidence (%)")
plt.title("Relationship between AI accuracy and expressed confidence")

plt.tight_layout()
plt.show()