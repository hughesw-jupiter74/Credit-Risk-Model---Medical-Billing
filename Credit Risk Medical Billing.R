# Load packages
library(tidyverse)
library(caret)
library(ROCR)
library(randomForest)
library(ggplot2)
library(pROC)
set.seed(42)

n <- 2000  # simulate more patients for realism

df <- tibble(
  PatientID = 1:n,
  Age = sample(18:90, n, replace = TRUE),
  Gender = factor(sample(c("Male", "Female"), n, replace = TRUE)),
  InsuranceType = factor(sample(c("Private", "Medicare", "Medicaid", "Uninsured"), n, replace = TRUE, prob = c(0.5, 0.2, 0.2, 0.1))),
  TotalCharges = round(runif(n, 100, 10000), 2),
  PaidAmount = round(runif(n, 0, 10000), 2),
  DaysToPay = sample(0:120, n, replace = TRUE),
  NumClaimsLastYear = sample(0:10, n, replace = TRUE),
  HasChronicCondition = sample(c(0, 1), n, replace = TRUE, prob = c(0.7, 0.3))
)

# Define Credit Risk:
# Risky if <80% paid or paid after 60 days
df <- df %>%
  mutate(
    PayRatio = PaidAmount / TotalCharges,
    CreditRisk = as.factor(ifelse(PayRatio < 0.8 | DaysToPay > 60, 1, 0))
  )

# Proportion of risky cases
ggplot(df, aes(CreditRisk)) +
  geom_bar(fill = "steelblue") +
  labs(title = "Distribution of Credit Risk", y = "Count")

# Pay ratio vs. risk
ggplot(df, aes(x = PayRatio, fill = CreditRisk)) +
  geom_histogram(bins = 30, alpha = 0.7, position = "identity") +
  labs(title = "Payment Ratio vs Credit Risk")

# Drop PatientID, PaidAmount, PayRatio (used to define target)
df_model <- df %>%
  select(-PatientID, -PaidAmount, -PayRatio)

# Train-test split
split <- createDataPartition(df_model$CreditRisk, p = 0.8, list = FALSE)
train <- df_model[split, ]
test <- df_model[-split, ]

logit_model <- glm(CreditRisk ~ ., data = train, family = "binomial")

summary(logit_model)

# Predict probabilities
pred_probs <- predict(logit_model, newdata = test, type = "response")
pred_class <- ifelse(pred_probs > 0.5, 1, 0)
pred_class <- factor(pred_class, levels = c(0,1))

# Confusion Matrix
confusionMatrix(pred_class, test$CreditRisk)

# ROC & AUC
roc_obj <- roc(as.numeric(test$CreditRisk), pred_probs)
plot(roc_obj, col = "blue", main = "ROC Curve - Logistic Model")
auc(roc_obj)

rf_model <- randomForest(CreditRisk ~ ., data = train, ntree = 200, importance = TRUE)

# Predict & evaluate
rf_pred <- predict(rf_model, test, type = "response")
confusionMatrix(rf_pred, test$CreditRisk)

# AUC
rf_probs <- predict(rf_model, test, type = "prob")[,2]
roc_rf <- roc(as.numeric(test$CreditRisk), rf_probs)
auc(roc_rf)

# Feature Importance
varImpPlot(rf_model)

# Top reasons for high credit risk
df_risk <- df %>% filter(CreditRisk == 1)

summary_stats <- df_risk %>%
  group_by(InsuranceType) %>%
  summarise(
    AvgDaysToPay = mean(DaysToPay),
    AvgTotalCharges = mean(TotalCharges),
    RiskyClaims = n()
  ) %>%
  arrange(desc(RiskyClaims))

summary_stats$InsuranceType <- reorder(summary_stats$InsuranceType, summary_stats$AvgDaysToPay)

ggplot(summary_stats, aes(x = AvgDaysToPay, y = InsuranceType, fill = RiskyClaims)) +
  geom_col() +
  scale_fill_gradient(low = "#56B1F7", high = "#132B43") +
  labs(
    title = "Average Days to Pay by Insurance Type",
    subtitle = "Color intensity indicates volume of risky claims",
    x = "Average Days to Pay",
    y = "Insurance Type",
    fill = "Risky Claims"
  ) +
  theme_minimal(base_size = 14)

# Convert summary_stats to long format
library(tidyr)

long_stats <- summary_stats %>%
  pivot_longer(cols = c(AvgDaysToPay, AvgTotalCharges), names_to = "Metric", values_to = "Value")

ggplot(long_stats, aes(x = InsuranceType, y = Value, fill = Metric)) +
  geom_col(position = "dodge") +
  labs(
    title = "Avg Days to Pay vs Avg Total Charges by Insurance Type",
    x = "Insurance Type", y = "Value", fill = "Metric"
  ) +
  theme_minimal(base_size = 14)


print(summary_stats)

