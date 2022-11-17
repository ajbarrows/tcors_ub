# TCORS Study 2 Data Cleaning
# Tony Barrows
# 2022-09-06

library(dplyr)
library(tcoRs)

# functions

pjtstedse <- function(df){
  ste <- substr(df$screen_id, 3, 3)
  
  df$site <- NA
  df$site[ste == "A"] <- "uvm"
  df$site[ste == "C"] <- "jhu"
  df$site[ste == "B"] <- "brown"
  
  df$site <- factor(df$site, levels = c("uvm", "brown", "jhu"))
  lbl <- df$cigarette_label
  
  df$project <- NA
  df$project[lbl == "J2" | lbl == "K2" | lbl == "L2"] <- "project 1"
  df$project[lbl == "M2" | lbl == "N2" | lbl == "P2"] <- "project 2"
  df$project[lbl == "V2" | lbl == "W2" | lbl == "X2"] <- "project 3"
  
  df$dose <- NA
  df$dose[lbl == "J2" | lbl == "N2" | lbl == "V2"] <- 15.8
  df$dose[lbl == "L2" | lbl == "P2" | lbl == "W2"] <- 0.4
  df$dose[lbl == "K2" | lbl == "M2" | lbl == "X2"] <- 2.4
  
  df$dose <- factor(df$dose)
  
  return(df)
}

load_csv <- function(path = "../../data/raw/") {

  # IVR data -----
  ivr <- read.csv(file = paste0(path, "S2IVRbyweek.csv"))
  
  # keep only maximum week in study
  ivr_clean <- ivr %>%
    select(subjectid, week, baseline_cpd = cigs_mean, study_mean, nonstudy_mean) %>%
    tidyr::fill(baseline_cpd) %>%
    group_by(subjectid, week) %>%
    mutate(
      total_cpd = sum(study_mean, nonstudy_mean, na.rm = TRUE),
      adherence = study_mean / total_cpd) %>%
    ungroup() %>%
    select(subjectid, week, baseline_cpd, total_cpd, study_mean, nonstudy_mean, adherence)
  
  ub <- produce_ub_dataset() %>% select(-c(is_100, menthol))
  
  # treatment groups
  rand <- read.csv(file = paste0(path, "s2_randomization_codes.csv"))
  
  # limit to randomized participants, join

  clean <- rand %>%
    left_join(ivr_clean, by = c("screen_id" = "subjectid")) %>%
    left_join(ub, by = "screen_id") %>%
    pjtstedse() %>%
    select(screen_id, week, dose, everything())
  
  # write
  # write.csv(clean, "../../data/clean/ub_joined_ivr.csv", row.names = FALSE)
  clean
}



pull_s2_data <- function(uvm_key, brown_key, jhu_key) {
  
  # load data
  uvm_rcon <- build_rcon(uvm_key)
  brown_rcon <- build_rcon(brown_key)
  jhu_rcon <- build_rcon(jhu_key)
  
  fields <- c("screen_id", "screen_age", "screen_sex")
  events <- NULL
  
  uvm <- download_rc_dataframe(uvm_rcon, fields, events)
  brown <- download_rc_dataframe(brown_rcon, fields, events)
  jhu <- download_rc_dataframe(jhu_rcon, fields, events)
  # 
  rbind(uvm, brown, jhu)
}

merge_trial_data <- function(rc_data, csv_joined) {
  df <- rc_data %>%
    filter(redcap_event_name == "screening_arm_1") %>%
    select(-redcap_event_name) %>%
    right_join(csv_joined, by = "screen_id") %>%
    mutate(screen_sex = redcapAPI::redcapFactorFlip(screen_sex)) %>%
    select(
      screen_id, screen_age, screen_sex, project, everything()
    ) %>%
    select(-cigarette_label) %>%
    mutate(across(
      .cols = c(week, menthol_status, strength, site, project), 
      factor)
      )
  
  # hard-code sex and age for J-C165 [female by eligibility criterion], age reported elsewhere
  df$screen_sex[df$screen_id == "J-C165"] <- "Female"
  df$screen_age[df$screen_id == "J-C165"] <- 34
  
  # same with K-C044
  df$screen_age[df$screen_id == "K-C044"] <- 54
  df
}

make_ub_key <- function(path = "../../data/raw/") {
  ub <- read.csv(file = paste0(path, "UB_edited.csv"))

  
  ub %>%
    mutate(
      usual_brand = stringr::str_trim(usual_brand)
    ) %>%
    distinct(usual_brand) %>%
    write.csv("../../data/raw/ub_key.csv", row.names = FALSE)
}


produce_ub_dataset <- function(path = "../../data/raw/") {
  ub_participants <- read.csv(paste0(path, "UB_edited.csv"))
  ub_key <- read.csv(paste0(path, "ub_key_edited.csv"))
  carmine <- read.csv(paste0(path, "carmine_table.csv"))
  

  
  ub_participants %>%
    mutate(usual_brand = stringr::str_trim(usual_brand)) %>%
    left_join(ub_key, by = "usual_brand") %>%
    left_join(carmine, by = c("carmine" = "Product")) %>%
    rename("carmine_nicotine" = Nicotine)
}


# main

# make UB key
# make_ub_key()

csv_joined <- load_csv()

# key <- read.csv("./password.csv")
# rc_data <- pull_s2_data(
#   uvm_key = key$password[key$username == "rc_s2_uvm"],
#   brown_key = key$password[key$username == "rc_s2_brown"],
#   jhu_key = key$password[key$username == "rc_s2_jhu"]
# )
# save(rc_data, file = "../../data/raw/s2_rc_data.RData")
load("../../data/raw/s2_rc_data.RData")

df <- merge_trial_data(rc_data, csv_joined)
save(df, file = "../../data/clean/s2_data.RData")
write.csv(df, file = "../../data/clean/s2_data.csv", row.names = FALSE)

