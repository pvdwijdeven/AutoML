# AutoML
![automl icon](data/icons/automl_icon_big.jpg?raw=true "AutoML icon")

Automated Machine Learning module

Use GUI or CLI

## Command Line Interface (CLI) Options

```bash
automl [OPTIONS]
````

### Options

| Option                 | Description                                                   |
| ---------------------- | ------------------------------------------------------------- |
| `-h`, `--help`         | Show this help message and exit                               |
| `--training_data FILE` | Filename for training data                                    |
| `--test_data FILE`     | Filename for test data                                        |
| `--EDA`                | Start EDA (default: `False`)                                  |
| `--nogui`              | Execute without GUI (default: `False`)                        |
| `--silent`             | If executed without GUI, no console output (default: `False`) |
| `--report_file FILE`   | Filename for report file (`.html`)                            |
| `--output_file FILE`   | Filename for output file (`.csv`)                             |

---

## 📊 What AutoML Currently Produces

When EDA is started from the GUI or via CLI with the `--EDA` option, AutoML generates an interactive **Exploratory Data Analysis (EDA) report** as an HTML file.

### ✅ Features of the Report

* **Dataset Overview**

  * Number of samples and features
  * Types of features (numeric, categorical, string, boolean)
  * Missing values summary
  * Data memory usage and preview tables (head/middle/tail)

* **Per-Feature Details**

  * Summary statistics (mean, std, min, max, skewness)
  * Frequency counts for categorical features
  * Intelligent suggestions for:

    * Encoding (e.g. One-Hot, Ordinal, Binary)
    * Outlier and skewness handling (e.g. log transforms, binning)
    * String feature parsing (e.g. split suggestions)
    * Zero-inflated feature treatment

* **Relation Analysis**

  * Correlation matrix
  * Mutual information plot
  * Feature interaction insights

* **Test Data Compatibility**

  * Optional analysis for test data consistency

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