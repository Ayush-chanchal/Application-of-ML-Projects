import numpy as np
import pandas as pd
import scipy.stats as stats

# Set random seed for reproducibility
np.random.seed(123)

# Generate synthetic data
group1 = np.random.normal(loc=15, scale=3, size=100)  # Group 1 data
group2 = np.random.normal(loc=20, scale=3, size=100)  # Group 2 data
group3 = np.random.normal(loc=25, scale=3, size=100)  # Group 3 data

# Combine data into a DataFrame
data = pd.DataFrame({
    'Group 1': group1,
    'Group 2': group2,
    'Group 3': group3
})

# Perform t-test between Group 1 and Group 2
t_statistic, p_value = stats.ttest_ind(data['Group 1'], data['Group 2'])
print("t-test between Group 1 and Group 2:")
print("t-statistic:", t_statistic)
print("p-value:", p_value)

# Calculate z-score and p-value for comparing Group 1 and Group 2 means
z_score, p_value_z = stats.zscore(data['Group 1']), stats.zscore(data['Group 2'])
print("z-test between Group 1 and Group 2:")
print("z-score:", z_score)
print("p-value:", p_value_z)

# Perform ANOVA test
f_statistic, p_value_anova = stats.f_oneway(data['Group 1'], data['Group 2'], data['Group 3'])
print("ANOVA test:")
print("F-statistic:", f_statistic)
print("p-value:", p_value_anova)