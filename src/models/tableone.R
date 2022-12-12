library(tableone)
library(dplyr)
library(kableExtra)


df <- read.csv("../../data/processed/model_features.csv")


proj_rename <- c(
  "project 1" = "Low SES Women",
  "project 2" = "Smokers w/ OUD",
  "project 3" = "Smokers w/ Affec."
)

site_rename <- c(
  "uvm" = "UVM",
  "brown" = "Brown",
  "jhu" = "Hopkins"
)



df_sub <- df %>% mutate(
  project = recode(project, !!!proj_rename),
  site = recode(site, !!!site_rename)
) %>%
  select(project,
         site,
         dose,
         screen_sex,
         screen_age,
         carmine_nicotine,
         baseline_cpd)
# 
tab <- CreateTableOne(
  strata = "dose",
  data = df_sub,
  test = FALSE
)

kbl(print(tab), booktabs = TRUE, format = "latex")
# 
# knitr::kable(print(tab), format = 'latex')





