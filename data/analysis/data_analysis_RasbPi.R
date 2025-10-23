# Comparison: PyTorch, TensorFlow, Scikit-learn, on Rasberry Pi
# Load Required Libraries
library(dplyr)
library(ggplot2)
library(readr)
library(GGally)
library(tidyr)

# Load Data
df <- read_csv("run_table_RasbPi.csv", show_col_types = FALSE) %>%
  rename(Library_Name = Library) %>%
  mutate(across(where(is.numeric), ~ ifelse(. == -999999, NA, .)))

# Compute Metrics
df <- df %>%
  mutate(
    Duration_s = Duration_ns / 1e9,
    M1_Avg_Power_W = if_else(Duration_s > 0, CPU_Energy_J / Duration_s, NA_real_),
    M2_Total_Energy_J = CPU_Energy_J,
    M3_Exec_Time_s = Duration_s,
    M5_Energy_Efficiency = if_else(CPU_Energy_J > 0, Accuracy_Pct / CPU_Energy_J, NA_real_)
  )

# Summary Table with all metrics
summary_table <- df %>%
  group_by(Library_Name) %>%
  summarise(
    N = n(),
    Mean_M1 = mean(M1_Avg_Power_W, na.rm = TRUE),
    Mean_M2 = mean(M2_Total_Energy_J, na.rm = TRUE),
    Mean_M3 = mean(M3_Exec_Time_s, na.rm = TRUE),
    Mean_M5 = mean(M5_Energy_Efficiency, na.rm = TRUE),
    Mean_Precision = mean(Precision_Pct, na.rm = TRUE),
    Mean_Accuracy = mean(Accuracy_Pct, na.rm = TRUE),
    Mean_MSE = mean(MSE, na.rm = TRUE),
    Mean_R2 = mean(r2_score, na.rm = TRUE),
    .groups = "drop"
  )

print(summary_table)

# Plot functions
metric_box <- function(metric, y_label, title) {
  ggplot(df, aes(x = Library_Name, y = .data[[metric]], fill = Library_Name)) +
    geom_boxplot(alpha = 0.8, na.rm = TRUE) +
    labs(title = title, y = y_label, x = "") +
    theme_minimal(base_size = 14) +
    theme(legend.position = "none")
}

bar_plot <- function(metric, y_label, title) {
  df_plot <- df %>%
    group_by(Library_Name) %>%
    summarise(
      Mean = mean(.data[[metric]], na.rm = TRUE),
      .groups = "drop"
    )
  
  ggplot(df_plot, aes(x = Library_Name, y = Mean, fill = Library_Name)) +
    geom_col(alpha = 0.8) +
    geom_text(aes(label = sprintf("%.2f", Mean)), 
              vjust = -0.5, size = 4, fontface = "bold") +
    labs(title = title, y = y_label, x = "") +
    theme_minimal(base_size = 14) +
    theme(legend.position = "none") +
    expand_limits(y = max(df_plot$Mean, na.rm = TRUE) * 1.1)
}

# Generate plots for M1, M2, M3, M5
core_metrics <- list(
  "M1_Avg_Power_W" = "Power (W)",
  "M2_Total_Energy_J" = "Energy (J)", 
  "M3_Exec_Time_s" = "Execution Time (s)",
  "M5_Energy_Efficiency" = "Accuracy per Joule"
)

for (m in names(core_metrics)) {
  print(metric_box(m, core_metrics[[m]], m))
  print(bar_plot(m, core_metrics[[m]], paste("Mean", m)))
}

# Bar plots for Precision, MSE, R², Accuracy
# Precision
if (sum(!is.na(df$Precision_Pct)) > 0) {
  print(bar_plot("Precision_Pct", "Precision (%)", "Mean Precision by Library"))
}

# Accuracy bar plot
if (sum(!is.na(df$Accuracy_Pct)) > 0) {
  print(bar_plot("Accuracy_Pct", "Accuracy (%)", "Mean Accuracy by Library"))
}

# Scatter: Accuracy vs Energy
ggplot(df, aes(x = M2_Total_Energy_J, y = Accuracy_Pct, color = Library_Name)) +
  geom_point(size = 2, alpha = 0.9, na.rm = TRUE) +
  labs(title = "Accuracy vs Total Energy", x = "Total Energy (J)", y = "Accuracy (%)") +
  theme_minimal(base_size = 14)

# Scatter Matrix
scatter_metrics <- c("M1_Avg_Power_W", "M2_Total_Energy_J", "M3_Exec_Time_s",
                     "M5_Energy_Efficiency", "Accuracy_Pct", "Precision_Pct")

df_pairs <- df %>% select(Library_Name, all_of(scatter_metrics))

if (ncol(df_pairs) > 2) {
  print(
    ggpairs(
      df_pairs,
      columns = 2:ncol(df_pairs),
      mapping = aes(color = Library_Name),
      upper = list(continuous = wrap("cor", size = 3)),
      diag = list(continuous = "densityDiag"),
      lower = list(continuous = wrap("points", alpha = 0.6, size = 1.5))
    ) + theme_minimal(base_size = 12)
  )
}

# Bar plots for MSE, r2 score
if (all(c("MSE", "r2_score") %in% names(df))) {
  # R² Score - Bar plot
  if (sum(!is.na(df$r2_score)) > 0) {
    r2_summary <- df %>%
      group_by(Library_Name) %>%
      summarise(
        Mean_r2 = mean(r2_score, na.rm = TRUE),
        .groups = "drop"
      )
    
    print(
      ggplot(r2_summary, aes(x = Library_Name, y = Mean_r2, fill = Library_Name)) +
        geom_col(alpha = 0.8) +
        geom_text(aes(label = sprintf("%.3f", Mean_r2)), 
                  vjust = -0.5, size = 4, fontface = "bold") +
        labs(title = "Mean R² Score by Library", y = "R²", x = "") +
        theme_minimal(base_size = 14) +
        theme(legend.position = "none") +
        expand_limits(y = max(r2_summary$Mean_r2, na.rm = TRUE) * 1.1)
    )
  }
  
  # MSE
  if (sum(!is.na(df$MSE)) > 0) {
    mse_summary <- df %>%
      group_by(Library_Name) %>%
      summarise(
        Mean_MSE = mean(MSE, na.rm = TRUE),
        .groups = "drop"
      ) %>%
      mutate(
        # Format label based on value size
        label = ifelse(Mean_MSE > 1000, 
                       sprintf("%.2e", Mean_MSE),  # Scientific notation for large values
                       sprintf("%.2f", Mean_MSE))  # Regular format for smaller values
      )
    
    print(
      ggplot(mse_summary, aes(x = Library_Name, y = Mean_MSE, fill = Library_Name)) +
        geom_col(alpha = 0.8) +
        geom_text(aes(label = label), 
                  vjust = -0.5, size = 4, fontface = "bold") +
        labs(title = "Mean MSE by Library", y = "MSE", x = "") +
        theme_minimal(base_size = 14) +
        theme(legend.position = "none") +
        expand_limits(y = max(mse_summary$Mean_MSE, na.rm = TRUE) * 1.1)
    )
  }
}

# Statistical Tests Anova and Kruskal
energy_metrics <- c("M1_Avg_Power_W", "M2_Total_Energy_J", "M3_Exec_Time_s",
                    "M5_Energy_Efficiency")

for (col in energy_metrics) {
  cat("Metric:", col, "\n")
  
  df_test <- df %>% filter(!is.na(.data[[col]]))
  
  if (n_distinct(df_test$Library_Name) >= 2 && nrow(df_test) >= 6) {
    # ANOVA
    cat("ANOVA:\n")
    print(summary(aov(as.formula(paste(col, "~ Library_Name")), data = df_test)))
    
    # Kruskal-Wallis
    cat("Kruskal-Wallis:\n")
    print(kruskal.test(as.formula(paste(col, "~ Library_Name")), data = df_test))
  } else {
    cat("Insufficient data for statistical tests\n")
  }
}

cat("Analysis complete.\n")