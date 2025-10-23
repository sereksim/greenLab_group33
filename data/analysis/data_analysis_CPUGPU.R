# Comparison: PyTorch and TensorFlow, on CPU/GPU run on Laptop
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
    M3_Execution_Time_s = Time_Taken_s,
    M4_Model_Accuracy = Accuracy_Pct,
    M5_Energy_Efficiency = Accuracy_Pct / Total_Energy_J
  )

# Summary Statistics
summary_table <- df %>%
  group_by(Platform, Library_Name) %>%
  summarise(
    N = n(),
    Mean_M2 = mean(M2_Total_Energy_J, na.rm = TRUE),
    SD_M2 = sd(M2_Total_Energy_J, na.rm = TRUE),
    Mean_M3 = mean(M3_Execution_Time_s, na.rm = TRUE),
    SD_M3 = sd(M3_Execution_Time_s, na.rm = TRUE),
    Mean_M4 = mean(M4_Model_Accuracy, na.rm = TRUE),
    SD_M4 = sd(M4_Model_Accuracy, na.rm = TRUE),
    Mean_M5 = mean(M5_Energy_Efficiency, na.rm = TRUE),
    SD_M5 = sd(M5_Energy_Efficiency, na.rm = TRUE),
    Mean_Accuracy = mean(Accuracy_Pct, na.rm = TRUE),
    .groups = "drop"
  )

print(summary_table)

# Function to create boxplots with values
boxplot_with_median <- function(data, metric_col, y_label, title_text) {
  # Calculate median values for each group
  median_data <- data %>%
    group_by(Platform, Library_Name) %>%
    summarise(Median = median(.data[[metric_col]], na.rm = TRUE), .groups = "drop")
  
  ggplot(data, aes(x = Library_Name, y = .data[[metric_col]], fill = Library_Name)) +
    geom_boxplot(alpha = 0.7) +
    geom_text(data = median_data, 
              aes(x = Library_Name, y = Median, label = sprintf("%.2f", Median)),
              position = position_dodge(width = 0.8),
              vjust = -0.5, size = 3.5, fontface = "bold", color = "black") +
    facet_wrap(~Platform) +
    labs(
      title = title_text,
      y = y_label,
      x = ""
    ) +
    theme_minimal(base_size = 14) +
    theme(legend.position = "none")
}

# Boxplots
# M2
print(boxplot_with_median(df, "M2_Total_Energy_J", "Total Energy (Joules)", 
                          "M2: Total Energy Consumption by Library and Platform"))
# M3
print(boxplot_with_median(df, "M3_Execution_Time_s", "Execution Time (seconds)", 
                          "M3: Execution Time by Library and Platform"))
# M4
print(boxplot_with_median(df, "M4_Model_Accuracy", "Accuracy (%)", 
                          "M4: Model Accuracy by Library and Platform"))
# M5
print(boxplot_with_median(df, "M5_Energy_Efficiency", "Accuracy per Joule", 
                          "M5: Energy Efficiency Ratio by Library and Platform"))
# Bar Plots config
bar_metrics <- function(metric_col, y_label, title_text) {
  plot_data <- df %>%
    group_by(Platform, Library_Name) %>%
    summarise(Mean = mean(.data[[metric_col]], na.rm = TRUE), .groups = "drop")
  
  ggplot(plot_data, aes(x = Library_Name, y = Mean, fill = Library_Name)) +
    geom_col(alpha = 0.7) +
    geom_text(aes(label = sprintf("%.2f", Mean)), 
              vjust = -0.5, size = 4, fontface = "bold") +
    facet_wrap(~Platform) +
    labs(title = title_text, y = y_label, x = "") +
    theme_minimal(base_size = 14) +
    expand_limits(y = max(plot_data$Mean, na.rm = TRUE) * 1.1)
}

# Generate bar plots
bar_metrics("M2_Total_Energy_J", "Total Energy (Joules)", "M2: Mean Total Energy Consumption")
bar_metrics("M3_Execution_Time_s", "Execution Time (seconds)", "M3: Mean Execution Time")
bar_metrics("M4_Model_Accuracy", "Accuracy (%)", "M4: Mean Model Accuracy")
bar_metrics("M5_Energy_Efficiency", "Accuracy per Joule", "M5: Mean Energy Efficiency")

# Mean Accuracy bar plot
accuracy_summary <- df %>%
  group_by(Platform, Library_Name) %>%
  summarise(Mean_Accuracy = mean(Accuracy_Pct, na.rm = TRUE), .groups = "drop")

ggplot(accuracy_summary, aes(x = Library_Name, y = Mean_Accuracy, fill = Library_Name)) +
  geom_col(alpha = 0.7) +
  geom_text(aes(label = sprintf("%.2f%%", Mean_Accuracy)), 
            vjust = -0.5, size = 4, fontface = "bold") +
  facet_wrap(~Platform) +
  labs(
    title = "Mean Accuracy by Library and Platform",
    y = "Accuracy (%)",
    x = ""
  ) +
  theme_minimal(base_size = 14) +
  expand_limits(y = max(accuracy_summary$Mean_Accuracy, na.rm = TRUE) * 1.1)

# Scatter Plot Energy vs Accuracy
ggplot(df, aes(x = Total_Energy_J, y = Accuracy_Pct, color = Library_Name, shape = Platform)) +
  geom_point(alpha = 0.8, size = 3) +
  labs(
    title = "Accuracy vs Total Energy by Library and Platform",
    x = "Total Energy (Joules)",
    y = "Accuracy (%)"
  ) +
  theme_minimal(base_size = 14)

# Scatter Matrix
print(
  ggpairs(
    df,
    columns = c("M2_Total_Energy_J", "M3_Execution_Time_s", "M4_Model_Accuracy", "M5_Energy_Efficiency"),
    mapping = aes(color = Library_Name, shape = Platform, alpha = 0.7),
    upper = list(continuous = wrap("cor", size = 4)),
    lower = list(continuous = wrap("points", size = 1.5)),
    diag = list(continuous = wrap("densityDiag", alpha = 0.7))
  ) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "bottom")
)

# ANOVA
cat("\n--- ANOVA Results ---\n")
aov_m2 <- aov(M2_Total_Energy_J ~ Library_Name * Platform, data = df)
aov_m3 <- aov(M3_Execution_Time_s ~ Library_Name * Platform, data = df)
aov_m4 <- aov(M4_Model_Accuracy ~ Library_Name * Platform, data = df)
aov_m5 <- aov(M5_Energy_Efficiency ~ Library_Name * Platform, data = df)

cat("\nM2: Total Energy Consumption\n")
print(summary(aov_m2))
cat("\nM3: Execution Time\n")
print(summary(aov_m3))
cat("\nM4: Model Accuracy\n")
print(summary(aov_m4))
cat("\nM5: Energy Efficiency\n")
print(summary(aov_m5))

# Kruskal-Wallis
cat("\n--- Kruskal-Wallis Results ---\n")
cat("\nM2: Total Energy Consumption\n")
print(kruskal.test(M2_Total_Energy_J ~ interaction(Library_Name, Platform), data = df))
cat("\nM3: Execution Time\n")
print(kruskal.test(M3_Execution_Time_s ~ interaction(Library_Name, Platform), data = df))
cat("\nM4: Model Accuracy\n")
print(kruskal.test(M4_Model_Accuracy ~ interaction(Library_Name, Platform), data = df))
cat("\nM5: Energy Efficiency\n")
print(kruskal.test(M5_Energy_Efficiency ~ interaction(Library_Name, Platform), data = df))

cat("Analysis complete.\n")