# AutoML
![automl icon](data/icons/automl_icon_big.jpg?raw=true "AutoML icon")

Automated Machine Learning module


## 📊 What AutoML Currently Produces

When EDA is started from the GUI or via CLI with the `--EDA` option, AutoML generates an interactive **Exploratory Data Analysis (EDA) report** as an HTML file.

## 🧠 Exploratory Data Analysis (EDA) Report

AutoML can automatically generate an interactive **EDA report** (HTML format) from any structured dataset via GUI or CLI. This report provides a comprehensive overview of your data before modeling, supporting informed decisions in feature engineering and preprocessing.

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

- **Optional Test Data Check**
  - Structural and statistical comparison with training data
  - Detection of mismatches and unseen categories

The report is fully HTML-based, styled for usability, and supports quick navigation through tabs and feature anchors.


---


Use GUI or CLI

## Command Line Interface (CLI) Options

```bash
automl [OPTIONS]
````

### Options

| Option                 | Description                                                   |
| ---------------------- | ------------------------------------------------------------- |
| `-h`, `--help`         | Show this help message and exit                               |
| `--title`              | Title for the project                                         |
| `--training_data FILE` | Filename for training data                                    |
| `--test_data FILE`     | Filename for test data                                        |
| `--EDA`                | Start EDA (default: `False`)                                  |
| `--nogui`              | Execute without GUI (default: `False`)                        |
| `--silent`             | If executed without GUI, no console output (default: `False`) |
| `--report_file FILE`   | Filename for report file (`.html`)                            |
| `--output_file FILE`   | Filename for output file (`.csv`)                             |

---

## 🚧 What’s in Progress

The AutoML module is still under active development. The following components are being built:

* 🔧 Automated preprocessing pipeline (guided by EDA insights)
* 🧠 Model selection and baseline training
* ⚙️ Hyperparameter optimization
* 📊 Evaluation metrics and comparison tools
* 💾 Exportable pipeline (as code and artifacts)



---

```