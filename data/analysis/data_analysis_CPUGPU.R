# Load Required Libraries
library(dplyr)
library(ggplot2)
library(readr)
library(GGally)

# Load Data
cpu <- read_csv("results/run_table_CPU.csv")
gpu <- read_csv("results/run_table_GPU.csv")

# Add platform column
cpu <- cpu %>% mutate(Platform = "CPU")
gpu <- gpu %>% mutate(Platform = "GPU")

# Combine datasets
df <- bind_rows(cpu, gpu)

# Compute Metrics
df <- df %>%
  mutate(
    M2_Total_Energy_J = Total_Energy_J,
    M4_Model_Accuracy = Accuracy_Pct,
    M5_Energy_Efficiency = Accuracy_Pct / Total_Energy_J
  )

# Summary Statistics
summary_table <- df %>%
  group_by(Platform, Library_Name) %>%
  summarise(
    Mean_M2 = mean(M2_Total_Energy_J),
    SD_M2 = sd(M2_Total_Energy_J),
    Mean_M4 = mean(M4_Model_Accuracy),
    SD_M4 = sd(M4_Model_Accuracy),
    Mean_M5 = mean(M5_Energy_Efficiency),
    SD_M5 = sd(M5_Energy_Efficiency),
    .groups = "drop"
  )

print(summary_table)

# Boxplots
# M2: Total Energy Consumption
ggplot(df, aes(x = Library_Name, y = M2_Total_Energy_J, fill = Library_Name)) +
  geom_boxplot(alpha = 0.7) +
  facet_wrap(~Platform) +
  labs(
    title = "M2: Total Energy Consumption by Library and Platform",
    y = "Total Energy (Joules)",
    x = ""
  ) +
  theme_minimal(base_size = 14)

# M4: Model Accuracy
ggplot(df, aes(x = Library_Name, y = M4_Model_Accuracy, fill = Library_Name)) +
  geom_boxplot(alpha = 0.7) +
  facet_wrap(~Platform) +
  labs(
    title = "M4: Model Accuracy by Library and Platform",
    y = "Accuracy (%)",
    x = ""
  ) +
  theme_minimal(base_size = 14)

# M5: Energy Efficiency
ggplot(df, aes(x = Library_Name, y = M5_Energy_Efficiency, fill = Library_Name)) +
  geom_boxplot(alpha = 0.7) +
  facet_wrap(~Platform) +
  labs(
    title = "M5: Energy Efficiency Ratio by Library and Platform",
    y = "Accuracy per Joule",
    x = ""
  ) +
  theme_minimal(base_size = 14)

# Bar Plots
bar_metrics <- function(metric_col, y_label, title_text) {
  df %>%
    group_by(Platform, Library_Name) %>%
    summarise(Mean = mean(.data[[metric_col]]), .groups = "drop") %>%
    ggplot(aes(x = Library_Name, y = Mean, fill = Library_Name)) +
    geom_col(alpha = 0.7) +
    facet_wrap(~Platform) +
    labs(title = title_text, y = y_label, x = "") +
    theme_minimal(base_size = 14)
}

bar_metrics("M2_Total_Energy_J", "Total Energy (Joules)", "M2: Mean Total Energy Consumption (Bar Plot)")
bar_metrics("M4_Model_Accuracy", "Accuracy (%)", "M4: Mean Model Accuracy (Bar Plot)")
bar_metrics("M5_Energy_Efficiency", "Accuracy per Joule", "M5: Mean Energy Efficiency (Bar Plot)")

# Scatter Plots: Energy vs Accuracy
ggplot(df, aes(x = Total_Energy_J, y = Accuracy_Pct, color = Library_Name, shape = Platform, size = M5_Energy_Efficiency)) +
  geom_point(alpha = 0.8) +
  labs(
    title = "Accuracy vs Total Energy by Library and Platform",
    x = "Total Energy (Joules)",
    y = "Accuracy (%)",
    size = "Energy Efficiency"
  ) +
  theme_minimal(base_size = 14)

# Scatter Matrix
ggpairs(
  df,
  columns = c("M2_Total_Energy_J", "M4_Model_Accuracy", "M5_Energy_Efficiency"),
  mapping = aes(color = Library_Name, shape = Platform)
)

# ANOVA
aov_m2 <- aov(M2_Total_Energy_J ~ Library_Name * Platform, data = df)
aov_m4 <- aov(M4_Model_Accuracy ~ Library_Name * Platform, data = df)
aov_m5 <- aov(M5_Energy_Efficiency ~ Library_Name * Platform, data = df)

summary(aov_m2)
summary(aov_m4)
summary(aov_m5)

# Kruskal-Wallis
kruskal.test(M2_Total_Energy_J ~ interaction(Library_Name, Platform), data = df)
kruskal.test(M4_Model_Accuracy ~ interaction(Library_Name, Platform), data = df)
kruskal.test(M5_Energy_Efficiency ~ interaction(Library_Name, Platform), data = df)
