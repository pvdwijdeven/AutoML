# AutoML

![automl icon](automl/data/icons/automl_icon_big.jpg?raw=true "AutoML icon")

Automated Machine Learning module

## 📊 What AutoML Currently Produces

AutoML generates an interactive **Exploratory Data Analysis (EDA) report** as an HTML file.

## 🧠 Exploratory Data Analysis (EDA) Report

AutoML can automatically generate an interactive **EDA report** (HTML format) from any structured dataset via CLI or via Python. This report provides a comprehensive overview of your data before modeling, supporting informed decisions in feature engineering and preprocessing.

### 🧩 Key Capabilities

- **Dataset Summary**
  - Number of features and samples
  - Feature types (numeric, categorical, string, boolean)
  - Memory usage and missing value analysis
  - Table previews (head, middle, tail)

- **Per-Feature Diagnostics**
  - Summary statistics and distributions
  - Category frequencies
  - Imputation and encoding suggestions
  - Flagging of zero-inflated, highly skewed, or constant features

- **Relation Insights**
  - Correlation matrix
  - Mutual information with the target
  - Feature-target interactions

- **Missing data overview**
  - Lists columns sorted by missing count and percentage
  - Visualizes missingness (bar plots, optional heatmap)
  - Estimates data retention if rows/columns are dropped
  - Suggests basic imputation strategies per column

- **(Optional) Competition/Isolated test Data Check**
  - Structural and statistical comparison with training data
  - Detection of mismatches and unseen categories

- **Recommendations to do before pre-processing**
  - While this tool provides a robust automated EDA, it does not incorporate domain-specific knowledge or an understanding of feature semantics. Prior to applying preprocessing steps, it is strongly advised to review the dataset with human expertise in mind.
  - The report outlines key areas where such insights can be integrated.

The report is fully HTML-based, styled for usability, and supports quick navigation through tabs and feature anchors.

---

Use CLI

## Command Line Interface (CLI) Options

```bash
automl [OPTIONS]
````

### Options

| Option                 | Description                                                   |
| ---------------------- | ------------------------------------------------------------- |
| `-h`, `--help`         | Show this help message and exit                               |
| `--title`              | Title for the project                                         |
| `--config FILE`        | Filename for configuration data                               |

---

## yaml configuration file

Here is an example of a yaml configuration file for loading the data:

```yaml
root: C:\Users\pvdwi\OneDrive\Python\ML\automl2
project_name: Titanic
training_file: personal\Titanic\data\train.csv
submission_file: personal\Titanic\data\submission.csv
report_template: personal\Titanic\export\Titanic_report.html
target: Survived
competition_file: personal\Titanic\data\test.csv
config_file: personal\yaml\Titanic.yaml
description_file: null
update_file: null
log_file: personal\Titanic\log\Titanic.log
```

## Column description file

If available, a description file can be added, so that the column description will be part of the report. The descriptions shall be of the following format:

````text
column_name
description, can be multiple lines.
so this line also. Next line, before next column name is always empty

column_name2
also here text....

column_name3

column_name4
Some column_names have no description
````

## 🚧 What’s in Progress

The AutoML module is still under active development. The following components are being built:

- 🔧 Automated preprocessing pipeline (guided by EDA insights)
- 🧠 Model selection and baseline training
- ⚙️ Hyperparameter optimization
- 📊 Evaluation metrics and comparison tools
- 💾 Exportable pipeline (as code and artifacts)

---
