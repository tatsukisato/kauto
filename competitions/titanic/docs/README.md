# Titanic - Machine Learning from Disaster

## Overview
- **Competition Name**: titanic
- **Category**: Getting Started
- **Metric**: Categorization Accuracy
- **Deadline**: 2030-01-01 00:00:00
- **URL**: https://www.kaggle.com/competitions/titanic
- **Full Overview**: [OVERVIEW.md](OVERVIEW.md)
- **Full Data Page**: [DATA.md](DATA.md)

## Description
Start here! Predict survival on the Titanic and get familiar with ML basics

## Data Structure
### Files (Raw Data)
- `test.csv`: 0.03 MB
- `train.csv`: 0.06 MB
- `gender_submission.csv`: 0.00 MB

### Column Descriptions
#### train.csv (891 rows, 12 columns)
| Column | Type | Missing | Unique | Example |
|---|---|---|---|---|
| PassengerId | int64 | 0 | 891 | 1 |
| Survived | int64 | 0 | 2 | 0 |
| Pclass | int64 | 0 | 3 | 3 |
| Name | object | 0 | 891 | Braund, Mr. Owen Harris |
| Sex | object | 0 | 2 | male |
| Age | float64 | 177 | 88 | 22.0 |
| SibSp | int64 | 0 | 7 | 1 |
| Parch | int64 | 0 | 7 | 0 |
| Ticket | object | 0 | 681 | A/5 21171 |
| Fare | float64 | 0 | 248 | 7.25 |
| Cabin | object | 687 | 147 | nan |
| Embarked | object | 2 | 3 | S |

#### test.csv (418 rows, 11 columns)
| Column | Type | Missing | Unique | Example |
|---|---|---|---|---|
| PassengerId | int64 | 0 | 418 | 892 |
| Pclass | int64 | 0 | 3 | 3 |
| Name | object | 0 | 418 | Kelly, Mr. James |
| Sex | object | 0 | 2 | male |
| Age | float64 | 86 | 79 | 34.5 |
| SibSp | int64 | 0 | 7 | 0 |
| Parch | int64 | 0 | 8 | 0 |
| Ticket | object | 0 | 363 | 330911 |
| Fare | float64 | 1 | 169 | 7.8292 |
| Cabin | object | 327 | 76 | nan |
| Embarked | object | 0 | 3 | Q |


