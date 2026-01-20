# UIDAI Digital Friction Analysis & Decision System

A production-ready, privacy-preserving ML platform for analyzing Aadhaar enrolment, demographic, and biometric data to detect digital friction, forecast system stress, and enable explainable, data-driven governance decisions.

# Project Overview

This system analyzes UIDAI (Unique Identification Authority of India) data to identify areas of digital friction in Aadhaar services, predict system stress, and provide actionable recommendations for optimizing service delivery. The platform leverages machine learning to detect anomalies, assess risks, and support decision-making for better governance.

# Architecture

1. **Data Ingestion & Cleaning** (`01_EDA/`, `02_Cleaned_Data/`)
   - Raw enrolment, demographic, and biometric data processing
   - Data quality checks and standardization
   - Merged dataset creation with temporal aggregation

2. **Exploratory Analysis** (`03_Analysis/`)
   - Univariate, bivariate, and trivariate analysis
   - Digital Friction Index calculation
   - Outlier detection and pattern identification

3. **Machine Learning Pipeline** (`Model/`)
   - Risk prediction models using XGBoost
   - Feature importance analysis with SHAP
   - Impact integration and decision modeling

4. **Decision Dashboard** (`Dashboard/`)
   - Real-time monitoring interface
   - Interactive visualizations and alerts
   - Actionable recommendation system

## 📊 Key Metrics & Features

### Data Quality Challenges Addressed

**Geographic Name Inconsistencies**
- **Issue**: 54+ and 940+  variations of state and district names respectively across datasets 
- **Examples**: "West Bangal" vs "West Bengal", "odisha" vs "ODISHA" vs "Orissa"
- **Impact**: Required careful merging and potential manual mapping
- **Status**: Identified for future standardization

**Temporal Alignment**
- **Challenge**: Different file date ranges and overlapping periods
- **Solution**: Standardized date formatting and temporal aggregation
- **Result**: Consistent monthly aggregations across all datasets

**Scale Variations**
- **Enrolment**: Daily transactional data
- **Demographic**: Periodic verification data
- **Biometric**: Quality-based processing data
- **Resolution**: Normalized to common temporal and geographic units

### Digital Friction Index (DFI)
**Formula**: `DFI = adult_system_stress / (age_18_greater + 1)`

- **Purpose**: Measures system efficiency relative to adult population
- **Interpretation**: Higher values indicate greater digital friction
- **Normalization**: Applied `log1p()` transformation for statistical normality
- **Usage**: Outlier detection and risk assessment
- **Threshold**: IQR-based method identifies high-friction outliers

### Core Features
- **Risk Scoring**: ML-powered risk assessment for different regions
- **Age Verification**: Automated age verification decisions
- **Disaster Alerts**: Early warning system for potential service disruptions
- **Kendra Optimization**: Recommendations for Common Service Centres
- **Outlier Detection**: Statistical identification of high-friction areas

## Project Structure

UIDAI/
├── 01_EDA/                     
│   ├── clean_enrolment.ipynb   
│   ├── clean_demographic.ipynb 
│   └── clean_biometric.ipynb   
├── 02_Cleaned_Data/            # Processed datasets
│   ├── enrolment_cleaned.csv
│   ├── demographic_cleaned.csv
│   ├── biometric_cleaned.csv
│   └── merged.csv              # Combined dataset
├── 03_Analysis/                # Statistical analysis notebooks
│   ├── UNIVARIATE.ipynb        
│   ├── BIVARIATE.ipynb         
│   ├── TRIVARIATE.ipynb       
│   ├── Outlier_Analysis.ipynb  # Anomaly detection
│   └── merged_cleaned_data.ipynb 
├── Model/                      # Machine learning components
│   ├── 01_model_pipeline.ipynb # ML model development
│   ├── 02_impact_integration.ipynb # 3 Impact ideas integration
│   └── Decision_model.ipynb    # Decision model for the impactful ideas
├── Dashboard/                  # Streamlit web application
│   ├── dashboard.py            # Main dashboard application
│   ├── requirements.txt        # Python dependencies
│   ├── Outputs/                # Generated analysis outputs
│   │   ├── final_decision_outputs.csv
│   │   ├── risk_predictions.csv
│   │   ├── high_friction_outliers.csv
│   │   ├── age_verification_output.csv
│   │   ├── kendra_optimization_recommendations.csv
│   │   ├── disaster_early_alerts.csv
│   │   └── feature_importance.csv
│   └── shap_plots/             # Model explainability visualizations
├── Outputs/                    # Additional analysis outputs
├── .devcontainer/              # Development environment setup
└── output*.png                 # Generated visualizations
```

## 🚀 Getting Started

### Prerequisites
- Python 3.11 or higher
- Git

### Option 1: Development Container (Recommended)
1. Open this repository in GitHub Codespaces or VS Code with Dev Containers extension
2. The container will automatically install dependencies and start the dashboard

### Option 2: Local Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd UIDAI
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r Dashboard/requirements.txt
   ```

4. **Run the dashboard**
   ```bash
   cd Dashboard
   streamlit run dashboard.py
   ```

The dashboard will be available at '' https://uidai-data-hackathon.streamlit.app/ ''

## Data Processing Pipeline

### 1. Data Cleaning & Preprocessing

#### Critical Cleaning Steps Performed:

**Enrolment Data (3 Files → 1M+ records)**
- **File Consolidation**: Merged 3 separate enrolment files covering 02-03-2025 to 31-12-2025
- **Date Standardization**: Converted DD-MM-YYYY format to consistent YYYYMMDD using `pd.to_datetime(date, dayfirst=True)`
- **Quality Check**: Zero null dates after conversion
- **Result**: Complete dataset preserved with no data loss

**Demographic Data (5 Files → 2M+ records)**
- **File Consolidation**: Merged 5 demographic files covering 01-03-2025 to 29-12-2025
- **Date Standardization**: Applied consistent date conversion across all files
- **Critical Issue Identified**: 50+ state name variations including:
  - Case inconsistencies: "odisha" vs "ODISHA" vs "Orissa"
  - Spelling variations: "West Bangal" vs "West Bengal"
  - Alternative names: "Uttaranchal" vs "Uttarakhand"
  - Invalid entries: "100000", "Jaipur", "Madanapalle" (city names instead of states)
- **Result**: Dataset intact but requires geographic name standardization

**Biometric Data (4 Files → 1,861,108 records)**
- **File Consolidation**: Merged 4 biometric files covering 01-03-2025 to 29-12-2025
- **Date Standardization**: Consistent date format conversion applied
- **Quality Issue**: Similar state name inconsistencies as demographic data
- **Result**: Complete dataset with temporal consistency achieved

#### Cleaning Results Summary:

| Dataset | Original Records | Final Records | Data Loss | Coverage Period |
|---------|----------------|---------------|-----------|----------------|
| Enrolment | 1,006,029 | 1,006,029 | 0% | Mar 2 - Dec 31, 2025 |
| Demographic | 2,071,700 | 2,071,700 | 0% | Mar 1 - Dec 29, 2025 |
| Biometric | 1,861,108 | 1,861,108 | 0% | Mar 1 - Dec 29, 2025 |

**Key Achievements:**
- ✅ **100% Data Preservation** - No records lost during cleaning
- ✅ **Date Consistency** - Standardized across all datasets
- ✅ **Temporal Alignment** - Overlapping date ranges enable cross-dataset analysis
- ✅ **File Consolidation** - Successfully merged multiple sources


### 2. Feature Engineering & Data Integration

#### Data Integration Process:
- **Merged Dataset Creation**: Combined enrolment, demographic, and biometric data on date-state-district keys
- **Temporal Aggregation**: Grouped by state, district, year, and month for analytical consistency
- **Cross-Dataset Validation**: Ensured alignment across all three data sources

#### Key Engineered Features:

**Digital Friction Index (DFI)**
```
DFI = adult_system_stress / (age_18_greater + 1)
```
- **Purpose**: Measures system efficiency relative to adult population
- **Interpretation**: Higher values indicate greater digital friction
- **Normalization**: Applied `log1p()` transformation for normality
- **Outlier Detection**: IQR-based method (Q3 + 1.5*IQR threshold)

**System Stress Metrics**
- **Adult System Stress**: Sum of adult enrolment and update activities
- **Child System Stress**: Sum of child enrolment and update activities
- **Baseline Updates**: Reference values for comparison

**Age-Based Features**
- **age_0_5**: Enrolment counts for children 0-5 years
- **age_5_17**: Enrolment counts for children 5-17 years
- **age_18_greater**: Enrolment counts for adults 18+ years
- **demo_age_5_17**: Demographic verification counts for 5-17 years
- **demo_age_17_**: Demographic verification counts for 17+ years
- **bio_age_5_17**: Biometric verification counts for 5-17 years
- **bio_age_17_**: Biometric verification counts for 17+ years

**Temporal Features**
- **Month**: 1-12 for seasonal pattern analysis
- **Year**: Temporal grouping for year-over-year comparison

**Derived Features**
- **total_enrolments**: Sum of all age-based enrolments
- **total_updates**: Combined demographic and biometric updates
- **log_digital_friction**: Log-transformed DFI for normality
- **friction_outlier**: Binary flag for high-friction regions

### 3. Machine Learning Model
- **Algorithm**: XGBoost Classifier for risk prediction
- **Features**: 18 engineered features including demographics, system metrics, and temporal indicators
- **Explainability**: SHAP values for model interpretability
- **Evaluation**: ROC-AUC score and classification metrics

## 🎯 Decision Categories

### Age Verification Decisions
- `AGE_NOT_VERIFIED`: Standard processing
- `DEFER_VERIFICATION`: Requires additional verification

### Disaster Alert Levels
- `NORMAL`: Regular operations
- Watch`: Monitor closely
- `HIGH`: Alert triggered

### Kendra Actions
- `NORMAL_OPERATIONS`: Standard service delivery
- `CHILD_UPDATE_CAMPS`: Special camps for child enrolment updates
- `ENHANCED_MONITORING`: Increased oversight required

## 📊 Key Insights & Findings

### System Stress Patterns
- Higher stress observed during peak enrolment periods
- Regional variations in digital friction across states
- Correlation between demographic profiles and system load

### Risk Factors
- Age group distribution impact on system performance
- Geographic disparities in service delivery
- Temporal trends in enrolment and update requests

### Optimization Opportunities
- Targeted interventions for high-friction regions
- Resource allocation based on predictive models
- Proactive disaster preparedness measures

## Technical Implementation

### Machine Learning Stack
- **XGBoost**: Gradient boosting for risk prediction
- **SHAP**: Model explainability and feature importance
- **Scikit-learn**: Data preprocessing and model evaluation

### Visualization Framework
- **Streamlit**: Interactive dashboard framework
- **Plotly**: Dynamic charts and graphs
- **Seaborn/Matplotlib**: Statistical visualizations

### Data Processing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scipy**: Statistical operations and outlier detection

## Model Outputs

### Risk Predictions
- Risk scores (0-1 scale) for each state-district-month combination
- Binary classification for high-friction detection
- Feature importance ranking

### Decision Outputs
- Age verification recommendations
- Disaster alert classifications
- Kendra optimization suggestions

### Explainability
- SHAP global feature importance
- Individual prediction explanations
- Decision rationale documentation

## Continuous Improvement

### Model Retraining
- Scheduled model updates with new data
- Performance monitoring and drift detection
- Feature engineering optimization

### Dashboard Enhancements
- User feedback integration
- Additional visualization capabilities
- Real-time data integration

### Quality Assurance
- Automated testing for data pipelines
- Model validation protocols
- Output consistency checks

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is part of a UIDAI data hackathon initiative and follows appropriate data governance and privacy regulations.

## 🔒 Privacy & Security

- All analysis performed on anonymized data
- No personal identifiers used in modeling
- Compliance with data protection regulations
- Secure data processing pipeline

##  Contact

For questions or suggestions regarding this project, please reach out through the project repository.

---

*Note: This is a proof-of-concept developed for a hackathon. Production deployment would require additional security measures, scalability considerations, and regulatory compliance validations.*

