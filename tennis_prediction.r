library(tidyverse)
library(lubridate)
library(caret)
library(e1071)      
library(pROC)
library(car)
library(ggplot2)
library(tidyr)
library(kernelshap)
library(shapviz)
library(zoo)

#1 DATA LOADING AND CLEANING
matches_combined <- list.files(path = "./data", 
                               pattern = "atp_matches_20.*\\.csv", 
                               full.names = TRUE)

matches_10yr <- map_df(matches_combined, read_csv)

grand_slams <- matches_10yr %>%
  filter(tourney_name %in% c("Australian Open", "Roland Garros", 
                             "Wimbledon", "US Open"))

nrow(grand_slams)

grand_slams_clean <- grand_slams %>%
  filter(!is.na(winner_rank) & !is.na(loser_rank)) %>%
  filter(!is.na(winner_age) & !is.na(loser_age)) %>%
  arrange(ymd(tourney_date))

nrow(grand_slams_clean) #total grand slam matches (cleaned)

#2 FEATURE ENGINEERING
#STEP 1: create a "player-match" dataset (each row = one player in one match)
#1 create player-match dataset
player_matches_winners <- grand_slams_clean %>%
  mutate(
    date = ymd(tourney_date),
    player = winner_name,
    opponent = loser_name,
    won = 1
  ) %>%
  select(date, surface, tourney_name, tourney_id, round,  # add round
         player, opponent, won)

player_matches_losers <- grand_slams_clean %>%
  mutate(
    date = ymd(tourney_date),
    player = loser_name,
    opponent = winner_name,
    won = 0
  ) %>%
  select(date, surface, tourney_name, tourney_id, round,  # add round
         player, opponent, won)

player_matches <- bind_rows(player_matches_winners, player_matches_losers) %>%
  arrange(date, player)


#2 calculate cumulative stats for each player
player_stats_over_time <- player_matches %>%
  group_by(player) %>%
  arrange(date, .by_group = TRUE) %>%
  mutate(
    # rolling 10-match form
    recent_form = zoo::rollmeanr(won, k = 10, fill = NA),
    
    # cumulative surface stats
    clay_matches  = cumsum(surface == "Clay"),
    clay_wins     = cumsum(surface == "Clay"  & won == 1),
    clay_winrate  = ifelse(clay_matches  > 0, clay_wins  / clay_matches,  NA_real_),
    
    grass_matches = cumsum(surface == "Grass"),
    grass_wins    = cumsum(surface == "Grass" & won == 1),
    grass_winrate = ifelse(grass_matches > 0, grass_wins / grass_matches, NA_real_),
    
    hard_matches  = cumsum(surface == "Hard"),
    hard_wins     = cumsum(surface == "Hard"  & won == 1),
    hard_winrate  = ifelse(hard_matches  > 0, hard_wins  / hard_matches,  NA_real_),
    
    # surface winrate for this match
    surface_winrate = case_when(
      surface == "Clay"  ~ clay_winrate,
      surface == "Grass" ~ grass_winrate,
      surface == "Hard"  ~ hard_winrate
    )
  ) %>%
  ungroup() %>%
  group_by(player, tourney_id) %>%
  mutate(
    match_num_in_tourney = row_number()
  ) %>%
  ungroup() %>%
  select(
    date, player, opponent, surface, tourney_id, round,
    recent_form, surface_winrate, match_num_in_tourney
  )
#STEP 2: merge player stats back into match data
grand_slams_clean <- grand_slams_clean %>%
  mutate(
    date = ymd(tourney_date),
    round = as.character(round)
  )

player_stats_over_time <- player_stats_over_time %>%
  mutate(round = as.character(round))
#WINNER
model_data_enhanced <- grand_slams_clean %>%
  mutate(date = ymd(tourney_date)) %>%
  left_join(
    player_stats_over_time,
    by = c(
      "date" = "date",
      "winner_name" = "player",
      "surface" = "surface",
      "tourney_id" = "tourney_id",
      "round" = "round"        
    )
  ) %>%
  rename(
    winner_recent_form = recent_form,
    winner_surface_winrate = surface_winrate,
    winner_matches_in_tourney = match_num_in_tourney
  )

#LOSER
model_data_enhanced <- model_data_enhanced %>%
  left_join(
    player_stats_over_time,
    by = c(
      "date" = "date",
      "loser_name" = "player",
      "surface" = "surface",
      "tourney_id" = "tourney_id",
      "round" = "round"   
    )
  ) %>%
  rename(
    loser_recent_form = recent_form,
    loser_surface_winrate = surface_winrate,
    loser_matches_in_tourney = match_num_in_tourney
  )

#STEP 3: calculate favourite vs underdog differences

model_data_final <- model_data_enhanced %>%
  mutate(
    match_id = row_number(),
    
    #identify favourite/underdog
    favourite_name = ifelse(winner_rank < loser_rank, 
                           winner_name, loser_name),
    underdog_name = ifelse(winner_rank < loser_rank, 
                           loser_name, winner_name),
    
    favourite_rank = pmin(winner_rank, loser_rank),
    underdog_rank = pmax(winner_rank, loser_rank),
    
    favourite_points = ifelse(winner_rank < loser_rank, 
                             winner_rank_points, loser_rank_points),
    underdog_points = ifelse(winner_rank < loser_rank, 
                             loser_rank_points, winner_rank_points),
    
    favourite_age = ifelse(winner_rank < loser_rank, 
                          winner_age, loser_age),
    underdog_age = ifelse(winner_rank < loser_rank, 
                          loser_age, winner_age),
    
    #assign recent form based on who is favourite
    favourite_recent_form = ifelse(winner_rank < loser_rank,
                                  winner_recent_form,
                                  loser_recent_form),
    underdog_recent_form = ifelse(winner_rank < loser_rank,
                                  loser_recent_form,
                                  winner_recent_form),
    
    #assign surface win rate
    favourite_surface_wr = ifelse(winner_rank < loser_rank,
                                 winner_surface_winrate,
                                 loser_surface_winrate),
    underdog_surface_wr = ifelse(winner_rank < loser_rank,
                                 loser_surface_winrate,
                                 winner_surface_winrate),
    
    #assign tournament fatigue
    favourite_fatigue = ifelse(winner_rank < loser_rank,
                              winner_matches_in_tourney,
                              loser_matches_in_tourney),
    underdog_fatigue = ifelse(winner_rank < loser_rank,
                              loser_matches_in_tourney,
                              winner_matches_in_tourney),
    
    #OUTCOME
    favourite_won = ifelse(winner_rank < loser_rank, 1, 0),
    
    #static features
    rank_diff = favourite_rank - underdog_rank,
    rank_points_diff = favourite_points - underdog_points,
    age_diff = favourite_age - underdog_age,
    seed_diff = ifelse(is.na(winner_seed) | is.na(loser_seed), 0,
                       ifelse(winner_rank < loser_rank,
                              winner_seed - loser_seed,
                              loser_seed - winner_seed)),
    hand_matchup = case_when(
      winner_hand == "R" & loser_hand == "R" ~ 0,
      winner_hand == "L" & loser_hand == "L" ~ 2,
      TRUE ~ 1
    ),
    
    #dynamic features
    recent_form_diff = favourite_recent_form - underdog_recent_form,
    surface_winrate_diff = favourite_surface_wr - underdog_surface_wr,
    fatigue_diff = favourite_fatigue - underdog_fatigue,
    
    surface = as.factor(surface),
    round = as.factor(round),
    year = year(ymd(tourney_date))
  ) %>%
  # emove rows with missing new features (early career players)
  filter(!is.na(recent_form_diff) & 
           !is.na(surface_winrate_diff) &
           !is.na(fatigue_diff)) %>%
  select(
    #identifiers
    match_id, date, tourney_id, tourney_name, round, 
    favourite_name, underdog_name,favourite_rank, underdog_rank,
    
    #features
    rank_diff, rank_points_diff, age_diff, seed_diff, hand_matchup,
    recent_form_diff, surface_winrate_diff, fatigue_diff, surface, 
    
    #outcome
    favourite_won, year)

#check how much data we have left
#matches with complete features
nrow(model_data_final)
#check class distribution 
sum(model_data_final$favourite_won) #favourite won
mean(model_data_final$favourite_won) * 100

sum(1 - model_data_final$favourite_won) #underdog won
sprintf("(%.1f%%)\n", mean(1 - model_data_final$favourite_won) * 100)


#4 Train/test split

train_data <- model_data_final %>% filter(year < 2024) %>% select(-year)
test_data <- model_data_final %>% filter(year == 2024) %>% select(-year)

names(train_data)[names(train_data) == "favourite_won"] <- "outcome"
names(test_data)[names(test_data) == "favourite_won"] <- "outcome"


#5 MODEL 1: logistic regression
model_lr_enhanced <- glm(
  outcome ~ rank_diff + rank_points_diff + age_diff + seed_diff + hand_matchup +
    recent_form_diff + surface_winrate_diff + fatigue_diff + surface + round,
  data = train_data,
  family = binomial
)

summary(model_lr_enhanced)

vif_values <- vif(model_lr_enhanced) #checking the feature engineering has multicollinearity or not
print(vif_values)

pred_lr_prob <- predict(model_lr_enhanced, test_data, type = "response")
pred_lr_class <- ifelse(pred_lr_prob > 0.5, 1, 0)

#evaluation metric
cm_lr <- confusionMatrix(as.factor(pred_lr_class), as.factor(test_data$outcome)) #evaluation
acc_lr <- cm_lr$overall['Accuracy']
lr_sensitivity <- cm_lr$byClass['Sensitivity']
lr_specificity <- cm_lr$byClass['Specificity']
lr_balanced_acc <- cm_lr$byClass['Balanced Accuracy']
lr_f1 <- cm_lr$byClass['F1']

print(cm_lr) #LR results

#feature importance by z-value
lr_coefs <- summary(model_lr_enhanced)$coefficients
lr_importance <- data.frame(
  Feature = rownames(lr_coefs)[-1],
  Coefficient = lr_coefs[-1, "Estimate"],
  Z_value = lr_coefs[-1, "z value"],
  P_value = lr_coefs[-1, "Pr(>|z|)"],
  Abs_Z = abs(lr_coefs[-1, "z value"])
) %>%
  arrange(desc(Abs_Z)) %>%
  head(10)

#top feature importance by z-value
print(lr_importance)

top_feature_lr <- lr_importance$Feature[1]
top_importance_lr <- lr_importance$Abs_Z[1]

#6 CLASS IMBALANCE (CALCULATING WEIGHTS FOR SVM) 

n_total <- nrow(train_data)
n_underdog <- sum(train_data$outcome == 0)
n_favourite <- sum(train_data$outcome == 1)

weight_underdog <- n_total / (2 * n_underdog)
weight_favourite <- n_total / (2 * n_favourite)


n_underdog #underdog wins (0)
sprintf("%.1f%%", n_underdog/n_total*100) #samples

n_favourite #favourite wins (1)
sprintf("%.1f%%", n_favourite/n_total*100) #samples

round(weight_underdog, 3) #underdog weight
round(weight_favourite, 3) #favourite weight

#7 MODEL 2: SVM with RBF kernel
train_data_svm <- train_data
test_data_svm <- test_data
train_data_svm$outcome <- as.factor(train_data_svm$outcome)
test_data_svm$outcome <- as.factor(test_data_svm$outcome)

class_weights_svm <- c("0" = weight_underdog, "1" = weight_favourite) #class weights

model_svm_enhanced <- svm(
  outcome ~ rank_diff + rank_points_diff + age_diff + seed_diff + hand_matchup +
    recent_form_diff + surface_winrate_diff + fatigue_diff +  
    surface + round,
  data = train_data_svm,
  type = "C-classification",
  kernel = "radial",
  cost = 1,
  class.weights = class_weights_svm,
  probability = TRUE
)

#SVM predictions
pred_svm_class <- predict(model_svm_enhanced, test_data_svm)
pred_svm_prob_full <- predict(model_svm_enhanced, test_data_svm, probability = TRUE)
pred_svm_prob <- attr(pred_svm_prob_full, "probabilities")[,"1"]

#evaluation metrics
cm_svm <- confusionMatrix(pred_svm_class, test_data_svm$outcome)
acc_svm <- cm_svm$overall['Accuracy']
svm_sensitivity <- cm_svm$byClass['Sensitivity']
svm_specificity <- cm_svm$byClass['Specificity']
svm_balanced_acc <- cm_svm$byClass['Balanced Accuracy']
svm_f1 <- cm_svm$byClass['F1']

print(cm_svm) #results

#feature importance using permutation
permutation_importance <- function(model, test_data, outcome_col, n_repeats = 5) {
  baseline_acc <- mean(predict(model, test_data) == test_data[[outcome_col]])
  feature_names <- setdiff(names(test_data), outcome_col)
  importance_scores <- numeric(length(feature_names))
  names(importance_scores) <- feature_names
  
  for (feat in feature_names) {
    acc_drops <- numeric(n_repeats)
    for (i in 1:n_repeats) {
      test_shuffled <- test_data
      test_shuffled[[feat]] <- sample(test_shuffled[[feat]])
      acc_shuffled <- mean(predict(model, test_shuffled) == test_data[[outcome_col]])
      acc_drops[i] <- baseline_acc - acc_shuffled
    }
    importance_scores[feat] <- mean(acc_drops)
  }
  return(sort(importance_scores, decreasing = TRUE))
}

svm_importance <- permutation_importance(model_svm_enhanced, test_data_svm, "outcome", n_repeats = 5)
svm_importance_df <- data.frame(
  Feature = names(svm_importance),
  Importance = svm_importance
)

#SVM top features 
print(head(svm_importance_df, 5))

top_feature_svm <- svm_importance_df$Feature[1]
top_importance_svm <- svm_importance[1]

#8 MODEL COMPARISONS
auc_lr <- auc(test_data$outcome, pred_lr_prob)
auc_svm <- auc(test_data$outcome, pred_svm_prob)

#summary table
summary_table <- data.frame(
  Model = c("Logistic Regression", "SVM"),
  Accuracy = c(acc_lr, acc_svm),
  Balanced_Accuracy = c(lr_balanced_acc, svm_balanced_acc),
  Sensitivity = c(lr_sensitivity, svm_sensitivity),
  Specificity = c(lr_specificity, svm_specificity),
  F1_Score = c(lr_f1, svm_f1),
  AUC = c(auc_lr, auc_svm)
) %>%
  arrange(desc(Balanced_Accuracy))

print(summary_table) #FINAL RESULTS FAVE VS UNDERDOG PREDICTION

#BEST MODEL
summary_table$Model[1]
#balanced accuracy 
summary_table$Balanced_Accuracy[1] * 100 #percentage

#9 SHAP Intepretation for SVM
test_data_shap <- test_data_svm %>% select(-outcome)

#prediction function for SHAP
pred_fun_svm <- function(object, newdata) {
  pred <- predict(object, newdata, probability = TRUE)
  prob_matrix <- attr(pred, "probabilities")
  as.numeric(prob_matrix[, "1"])
}

#computing SHAP value 
set.seed(123)
shap_svm <- kernelshap(
  object = model_svm_enhanced,
  X = test_data_shap[1:100, ],
  bg_X = train_data_svm[sample(1:nrow(train_data_svm), 50),
                        -which(names(train_data_svm) == "outcome")],
  pred_fun = pred_fun_svm
)

sv_svm <- shapviz(shap_svm)

#local intepretation
actual_outcomes <- test_data[1:100, ]$outcome
predictions_svm <- predict(model_svm_enhanced, test_data_svm[1:100,])

#EXAMPLE 1: correctly predicted upset

#find a correctly predicted upset
upsets_detected <- which(actual_outcomes == 0 & predictions_svm == "0")

if(length(upsets_detected) > 0) {
  upset_idx <- upsets_detected[1]
}

#waterfall plot shows why the model predicted this upset
sv_waterfall(sv_svm, row_id = upset_idx)

#EXAMPLE 2: correctly predicted favourite win

#find a correctly predicted favourite win
favourites_correct <- which(actual_outcomes == 1 & predictions_svm == "1")

if(length(favourites_correct) > 0) {
  fav_idx <- favourites_correct[1]
}
#waterfall plot shows why the model predicted favourite would win
sv_waterfall(sv_svm, row_id = fav_idx)

#EXAMPLE 3: model error - false positive

#find a false positive (predicted upset but favourite won)
false_positives <- which(actual_outcomes == 1 & predictions_svm == "0")

if(length(false_positives) > 0) {
  fp_idx <- false_positives[1]
}
#waterfall plot shows why the model made this mistake
sv_waterfall(sv_svm, row_id = fp_idx)

#2024 PREDICTION 
results_detailed <- test_data %>%
  mutate(
    predicted_outcome = pred_lr_class, 
    predicted_prob = pred_lr_prob, 
    correct = (predicted_outcome == outcome)
  )

results_summary <-  results_detailed %>%
  select(date, tourney_name, round, 
         favourite_name, underdog_name, 
         favourite_rank, underdog_rank, 
         actual_outcome = outcome, 
         predicted_outcome, 
         predicted_prob,
         correct) %>%
  arrange(date)

print(head(results_summary, 10))

#SVM VER
results_detailed_svm <- test_data %>%
  mutate(
    predicted_outcome_svm = pred_svm_class, 
    predicted_prob_svm = pred_svm_prob, 
    correct = (predicted_outcome_svm == outcome)
  )

results_summary_svm <-  results_detailed_svm %>%
  select(date, tourney_name, round, 
         favourite_name, underdog_name, 
         favourite_rank, underdog_rank, 
         actual_outcome = outcome, 
         predicted_outcome_svm, 
         predicted_prob_svm, 
         correct) %>%
  arrange(date)

print(head(results_summary, 10))
