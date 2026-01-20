# Grand Slam Tennis Match Prediction

Predicting tennis match outcomes in Grand Slam tournaments using machine learning, with a focus on identifying upsets through advanced feature engineering and model interpretability.

## Project Overview

This project develops a predictive model for Grand Slam tennis matches by analyzing historical ATP data from 2015-2024. The model predicts whether the higher-ranked player (favourite) will win, achieving **89.4% balanced accuracy** and **90.6% overall accuracy** using Logistic Regression.

### Key Features
- **Temporal feature engineering** incorporating recent form, surface-specific expertise, and tournament fatigue
- **Class-weighted SVM** to handle imbalanced outcomes (favourites win ~77% of matches)
- **SHAP analysis** for model interpretability and understanding prediction drivers
- Evaluation on 2024 Grand Slam matches as holdout test set

## Results

| Model | Accuracy | Balanced Accuracy | Sensitivity | Specificity | F1 Score | AUC |
|-------|----------|-------------------|-------------|-------------|----------|-----|
| **Logistic Regression** | **90.6%** | **89.4%** | 86.7% | 92.1% | 0.839 | 0.953 |
| SVM (RBF kernel) | 88.7% | 88.6% | 88.3% | 88.8% | 0.815 | 0.957 |

*Baseline (always predict favourite): 77.4% accuracy, 50.0% balanced accuracy*

### Performance Highlights
- **Logistic Regression** selected as best model with 89.4% balanced accuracy and highest overall accuracy (90.6%)
- **+39.4 percentage points** improvement in balanced accuracy over baseline
- Successfully identifies underdog victories with 92.1% specificity
- Exceptional AUC (0.953) indicates strong discriminative ability across all thresholds

## Dataset

**Source**: [Jeff Sackmann's tennis_atp repository](https://github.com/JeffSackmann/tennis_atp) (2015-2024)  
**Scope**: Grand Slam tournaments only (Australian Open, Roland Garros, Wimbledon, US Open)  
**Final dataset**: 3,479 matches with complete feature data

### Feature Engineering

The model uses 10 engineered features categorized into static and dynamic factors:

**Static Features:**
- `rank_diff`: Difference in ATP rankings (favourite - underdog)
- `rank_points_diff`: Difference in ranking points
- `age_diff`: Age difference between players
- `seed_diff`: Difference in tournament seeding
- `hand_matchup`: Encoded handedness interaction (0=R vs R, 1=mixed, 2=L vs L)

**Dynamic Features (time-aware):**
- `recent_form_diff`: Difference in 10-match rolling win rate
- `surface_winrate_diff`: Difference in career surface-specific win rates (Clay/Grass/Hard)
- `fatigue_diff`: Difference in matches played in current tournament

**Contextual Features:**
- `surface`: Match surface (Clay/Grass/Hard)
- `round`: Tournament round (R128 through Finals)

## Methodology

### Data Processing
1. **Temporal ordering**: Arranged matches chronologically to prevent data leakage
2. **Feature calculation**: Computed cumulative statistics using only past matches
3. **Train-test split**: 2015-2023 for training, 2024 as holdout test set
4. **Missing data handling**: Removed matches with insufficient historical data for feature calculation

### Models

**Model 1: Logistic Regression (Best Model)**
- Standard logistic regression with all engineered features
- VIF analysis confirmed no multicollinearity (all VIF < 5)
- Feature importance assessed via z-values

**Model 2: Support Vector Machine**
- RBF (Radial Basis Function) kernel for non-linear decision boundaries
- **Class weighting** to address imbalance:
  - Underdog weight: 1.764
  - Favourite weight: 0.698
- Hyperparameters: cost = 1 (default parameter)
- Feature importance via permutation testing
- Slightly higher AUC but lower overall performance

### Model Interpretability

**SHAP (SHapley Additive exPlanations)** values computed using KernelSHAP on SVM model (100 test samples) for additional interpretability analysis:
- **Global interpretation**: Feature importance across all predictions
- **Local interpretation**: Individual match prediction explanations via waterfall plots
- Analyzed three scenarios:
  1. Correctly predicted upset
  2. Correctly predicted favourite win
  3. Model error (false positive)

*Note: While Logistic Regression performed the best, SHAP analysis on SVM provides additional insights into feature interactions and non-linear effects.*

## Key Findings

### Most Important Features

**Logistic Regression (z-value ranking):**
Top features by statistical significance in predicting match outcomes

**SVM (Permutation importance):**
1. **recent_form_diff** (0.058 accuracy drop)
2. **surface_winrate_diff** (0.047 accuracy drop)
3. **rank_diff** (0.043 accuracy drop)
4. **rank_points_diff** (0.039 accuracy drop)
5. **fatigue_diff** (0.028 accuracy drop)

### Insights
- **Recent form** is the strongest predictor, suggesting current momentum matters more than historical rankings
- **Surface expertise** significantly impacts match outcomes, particularly on Clay and Grass
- **Tournament fatigue** (matches already played) affects performance in later rounds
- Traditional ranking features remain important but are enhanced by dynamic features

## Built With

**Programming Language**: R  

**Key Libraries**:
- `tidyverse`, `dplyr`: Data manipulation
- `lubridate`: Date handling
- `caret`: Model training and evaluation
- `e1071`: SVM implementation
- `pROC`: ROC curve analysis
- `kernelshap`, `shapviz`: SHAP analysis
- `zoo`: Rolling statistics
- `car`: VIF analysis

## Potential Applications

- **Sports betting**: Identify value bets when model disagrees with bookmaker odds
- **Tournament analysis**: Pre-tournament favorite identification and upset likelihood
- **Player development**: Quantify importance of surface expertise and recent form
- **Broadcasting**: Real-time match outcome probabilities for enhanced viewer engagement

## Future Improvements

- Incorporate head-to-head historical matchups
- Add physical metrics (height, playing style)
- Extend to ATP 500/1000 tournaments for larger training set
- Implement ensemble methods (XGBoost, Random Forest)
- Real-time prediction API with live tournament data


*This project was completed as part of the Seminar Data Science for Marketing Analytics course at Erasmus University Rotterdam (2026)*
