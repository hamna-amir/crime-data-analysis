# crime-data-analysis
An interactive Streamlit dashboard that visualizes and analyzes Los Angeles crime data using Python. Includes statistical modeling, trend analysis, demographic breakdowns to support public safety insights.



Project Objective
To provide public safety insights by analyzing crime data through:
- Statistical models (Poisson distribution, confidence intervals)
- Predictive analysis (Linear regression)
- Probabilistic reasoning (Bayes' Theorem)
- Interactive visualizations

Key Features
- Crimes per Area**: Bar chart showing total incidents per jurisdiction
- Monthly Crime Trends**: Line chart to visualize changes over time
- Victim Demographics**: Gender, age, and ethnicity breakdowns
- Boxplot Analysis**: Victim age by crime type
- Hourly Trends: When most crimes occur
- Poisson Distribution: Frequency modeling of daily crime
- Bayesian Inference: Conditional probabilities of specific crime types
- Linear Regression:
  - Crime count over the years
  - Victim age trends
  - Crime frequency by hour

The project uses real-world crime data with the following columns:
- `DATE OCC`: Date of the crime
- `TIME OCC`: Time of occurrence
- `AREA NAME`: Police jurisdiction
- `Crm Cd Desc`: Crime description
- `Vict Age`, `Vict Sex`, `Vict Descent`: Demographics of victims

Prerequisites
- Python 3.8+
- Streamlit
- pandas, numpy, seaborn, matplotlib, scikit-learn, scipy

### Installation
git clone https://github.com/hamna-amir/crime-data-analysis.git
cd crime-data-analysis
pip install -r requirements.txt
streamlit run app.py

requirements.txt:
streamlit
pandas
matplotlib
seaborn
numpy
scipy
scikit-learn

