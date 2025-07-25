
### 🔍 1. Data Verkenning (Exploratory Data Analysis - EDA)

**Doel**: Inzicht krijgen in de data, kwaliteit en relaties.

- Inladen van de dataset  
- Vorm (dimensies), types, eerste en laatste rijen  
- Missende waarden en duplicaten  
- Statistische samenvattingen (mean, std, etc.)  
- Verdeling van variabelen (histogrammen, boxplots)  
- Correlaties en relaties tussen variabelen (pairplots, heatmaps)  
- Outliers opsporen  

---

### 🛠️ 2. Voorbewerking (Preprocessing)

**Doel**: Dataset klaarmaken voor modelleren.

- Imputatie van missende waarden  
- Encoderen van categorische variabelen (LabelEncoder, OneHotEncoder)  
- Normalisatie of standaardisatie  
- Train-test splitsing  
- Eventueel feature engineering of selectie  

---

### 🧠 3. Modelselectie en Training

**Doel**: Een of meerdere ML-modellen trainen.

- Keuze van model(len) (bijv. Decision Tree, Random Forest, SVM, Logistic Regression, etc.)  
- Trainen van model(len) op de trainingsdata  
- Hyperparameter tuning (bijv. via GridSearchCV)  

---

### 📈 4. Evaluatie van Modellen

**Doel**: Prestaties beoordelen op testdata.

- Voorspellingen op testdata  
- Evaluatiemetrics:  
  - Regressie: MAE, MSE, R²  
  - Classificatie: Accuracy, Precision, Recall, F1-score, Confusion matrix, ROC/AUC  
- Visualisatie van prestaties  

---

### 🔁 5. Optimalisatie en Herhaling

**Doel**: Verbeteren door iteratie.

- Feature selectie verbeteren  
- Andere modellen proberen  
- Balanceren van klassen (bijv. met SMOTE)  
- Hyperparameters verfijnen  

---

### 💾 6. Opslaan en Hergebruik

**Doel**: Model bewaren en toepassen in andere contexten.

- Opslaan van model (joblib, pickle)  
- Opslaan van preprocessing stappen (bijv. Pipeline)  
- Script of interface maken voor voorspellingen op nieuwe data  

---

### Extra toelichting op EDA-stappen:

#### 1. Data Verkenning

**a)** Inladen van de dataset  
> Tip: Bekijk het datatype van de ingelezen data (DataFrame, array, etc.)

**b)** Vorm (dimensies), types, eerste en laatste rijen  
> Gebruik `shape`, `dtypes`, `head()` en `tail()`

**c)** Missende waarden en duplicaten  
> Gebruik `isnull().sum()` en `duplicated()`

**d)** Statistische samenvattingen  
> `describe()`, `value_counts()`, variantie, std

**e)** Verdeling van variabelen  
> Histogrammen, boxplots, staafdiagrammen, skewness, kurtosis

**f)** Correlaties en relaties  
> Correlatiematrix, heatmaps, scatterplots

**g)** Outliers  
> Boxplots, Z-score, IQR, visuele inspectie

**h)** Univariate en bivariate analyses  
> Crosstabs, grouped boxplots

**i)** Time series  
> Trend, seizoensinvloeden, ontbrekende perioden

**j)** Data kwaliteit  
> Controleer inconsistenties zoals negatieve leeftijden

---

### Tips voor goede EDA:

- Begin breed, zoom daarna in  
- Combineer statistiek en visualisatie  
- Documenteer bevindingen  
- Werk iteratief  
- Gebruik tools zoals Jupyter Notebooks  

---

### Automatiseren van EDA (minimale gebruikersinput)

1. **Automatische analyse**:  
   Gebruik tools als `pandas-profiling`, `sweetviz`, `dataprep`.

2. **Standaardregels**:  
   Vaste aanpak voor outliers, categorische detectie, missende waarden.

3. **Feature-type detectie**:  
   Targetkolom detecteren, classificatie vs regressie.

4. **Logging & rapportage**:  
   Opslaan van samenvattingen, grafieken en bevindingen.

---

### Wanneer juist **wel** input vragen?

- Bij onzekerheid: targetkolom, probleemtype  
- Domeinkennis nodig  
- Interpretatie van grafieken  
- Keuzes zoals outlierbeleid of imputatie

---

### Praktische tip:

> Begin met volledige automatische run  
> Laat gebruiker daarna aanpassen via menu of config-file
