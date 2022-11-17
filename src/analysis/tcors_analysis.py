from tcors_functions import *

random_state = 42

# Load longitudinal data
#s2_full = load_data("s2_data.csv")

# Load Week 12 data
w12_df = load_data("w12_recode.csv")


# (Example)

df_sub = w12_df[["screen_sex", "screen_age", "project", "baseline_cpd", "dose", "total_cpd"]]

X_train, X_test, y_train, y_test = make_training_split(df_sub, y = ["total_cpd"])

print(X_train)
print(y_train)