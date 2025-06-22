import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
st.set_page_config(page_title="Crime Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
	df = pd.read_csv("crimedata.csv", usecols=["DATE OCC", "TIME OCC", "AREA NAME", "Crm Cd Desc", "Vict Age", "Vict Sex", "Vict Descent"])
	df.columns = df.columns.str.strip().str.replace('\xa0', ' ', regex=True)
	df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], errors='coerce')
	df['Vict Age'] = pd.to_numeric(df['Vict Age'], errors='coerce')
	df.dropna(subset=["DATE OCC", "AREA NAME", "Crm Cd Desc", "Vict Age"], inplace=True)
	df = df[df['Vict Age'] > 0]
	df['Year'] = df['DATE OCC'].dt.year
	df['Month'] = df['DATE OCC'].dt.to_period('M').dt.to_timestamp()
	df['Hour'] = df['TIME OCC'] // 100
	return df
df = load_data()

# Sidebar filters
st.sidebar.header("🔍 Filter Crime Data")
selected_areas = st.sidebar.multiselect("Select Area(s)", sorted(df['AREA NAME'].unique()), default=None)
selected_crime = st.sidebar.multiselect("Select Crime Type(s)", sorted(df['Crm Cd Desc'].unique()), default=None)
year_range = st.sidebar.slider("Select Year Range", int(df['Year'].min()), int(df['Year'].max()), (2015, 2020))

# Apply filters
filtered_df = df.copy()
if selected_areas:
	filtered_df = filtered_df[filtered_df['AREA NAME'].isin(selected_areas)]
if selected_crime:
	filtered_df = filtered_df[filtered_df['Crm Cd Desc'].isin(selected_crime)]
filtered_df = filtered_df[(filtered_df['Year'] >= year_range[0]) & (filtered_df['Year'] <= year_range[1])]

# Download button
st.sidebar.download_button("Download Filtered Data", filtered_df.to_csv(index=False), file_name="filtered_crime_data.csv")

# Dashboard title
st.title("📊 Crime Data Analysis Dashboard")

# Key Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Records", f"{len(filtered_df):,}")
col2.metric("Avg. Victim Age", f"{filtered_df['Vict Age'].mean():.1f}")
col3.metric("Crime Period", f"{filtered_df['Year'].min()} - {filtered_df['Year'].max()}")

# Chart buttons
st.header("📈 Interactive Charts")
if st.button("📍 Crimes per Area (Bar Chart)"):
	st.subheader("Crimes per Area")
	st.bar_chart(filtered_df['AREA NAME'].value_counts())

if st.button("📅 Monthly Crime Trend (Line Chart)"):
	st.subheader("Monthly Crime Trend")
	monthly = filtered_df.groupby('Month').size()
	fig1, ax1 = plt.subplots()
	monthly.plot(ax=ax1)
	ax1.set_ylabel("Crimes")
	st.pyplot(fig1)

if st.button("👥 Victim Sex Distribution (Pie Chart)"):
	st.subheader("Victim Sex Distribution")
	fig2, ax2 = plt.subplots()
	filtered_df['Vict Sex'].fillna("Unknown").value_counts().plot.pie(autopct='%1.1f%%', ax=ax2)
	ax2.set_ylabel("")
	st.pyplot(fig2)

if st.button("📊 Victim Age Stats"):
	st.subheader("Victim Age: Descriptive Stats")
	mean_age = filtered_df['Vict Age'].mean()
	std_age = filtered_df['Vict Age'].std()
	n = filtered_df['Vict Age'].count()
	ci = stats.norm.interval(0.95, loc=mean_age, scale=std_age / np.sqrt(n))
	st.write(f"Mean Age: {mean_age:.2f}, Std Dev: {std_age:.2f}")
	st.write(f"95% Confidence Interval: ({ci[0]:.2f}, {ci[1]:.2f})")

if st.button("📈 Poisson Fit: Daily Crimes"):
	st.subheader("Poisson Distribution of Daily Crimes")
	daily_counts = filtered_df.groupby(filtered_df['DATE OCC'].dt.date).size()
	mu = daily_counts.mean()
	fig3, ax3 = plt.subplots()
	sns.histplot(daily_counts, stat="density", bins=30, label="Observed", ax=ax3)
	x = np.arange(0, daily_counts.max())
	ax3.plot(x, stats.poisson.pmf(x, mu), 'r-', label="Poisson Fit")
	ax3.legend()
	st.pyplot(fig3)

if st.button("🧠 Bayes' Theorem Analysis"):
	st.subheader("Bayes' Theorem Analysis")
	assaults = filtered_df[filtered_df['Crm Cd Desc'].str.contains("ASSAULT", na=False)]
	p_f = (filtered_df['Vict Sex'] == 'F').mean()
	p_a = (filtered_df['Crm Cd Desc'].str.contains("ASSAULT", na=False)).mean()
	p_f_given_a = (assaults['Vict Sex'] == 'F').mean()
	p_a_given_f = (p_f_given_a * p_a) / p_f if p_f > 0 else np.nan
	st.write(f"P(Female | Assault): {p_f_given_a:.2f}")
	st.write(f"P(Assault | Female): {p_a_given_f:.2f}")

# Regression and More Graphs
st.header("📉 Linear Regression & Extra Charts")

if st.button("📉 Linear Regression: Crime Count by Year"):
	st.subheader("Linear Regression: Crime Count by Year")
	reg_data = filtered_df.groupby(['AREA NAME', 'Year']).size().reset_index(name='Crime Count')
	X = reg_data[['Year']]
	y = reg_data['Crime Count']
	lin_model = LinearRegression().fit(X, y)
	st.write(f"Coef: {lin_model.coef_[0]:.2f}, Intercept: {lin_model.intercept_:.2f}")
	fig_lin, ax_lin = plt.subplots()
	for area in reg_data['AREA NAME'].unique():
    	area_data = reg_data[reg_data['AREA NAME'] == area]
    	ax_lin.scatter(area_data['Year'], area_data['Crime Count'], label=area, alpha=0.6)
	x_vals = np.array(sorted(reg_data['Year'].unique()))
	y_vals = lin_model.predict(x_vals.reshape(-1, 1))
	ax_lin.plot(x_vals, y_vals, color='black', linewidth=2, label='Regression Line')
	ax_lin.set_xlabel("Year")
	ax_lin.set_ylabel("Crime Count")
	ax_lin.set_title("Linear Regression: Crime Count by Year")
	ax_lin.legend()
	st.pyplot(fig_lin)

if st.button("📉 Linear Regression: Victim Age by Year"):
	st.subheader("Linear Regression: Victim Age over Years")
	age_data = filtered_df.groupby('Year')['Vict Age'].mean().reset_index()
	X_age = age_data[['Year']]
	y_age = age_data['Vict Age']
	age_model = LinearRegression().fit(X_age, y_age)
	fig_age, ax_age = plt.subplots()
	ax_age.scatter(age_data['Year'], age_data['Vict Age'], label='Avg Age')
	ax_age.plot(age_data['Year'], age_model.predict(X_age), color='green', label='Regression Line')
	ax_age.set_xlabel("Year")
	ax_age.set_ylabel("Avg Victim Age")
	ax_age.set_title("Trend of Victim Age over Time")
	ax_age.legend()
	st.pyplot(fig_age)

if st.button("📉 Linear Regression: Crime Count by Hour"):
	st.subheader("Linear Regression: Crime Count by Hour")
	hourly_data = filtered_df.groupby('Hour').size().reset_index(name='Crime Count')
	X_hr = hourly_data[['Hour']]
	y_hr = hourly_data['Crime Count']
	hour_model = LinearRegression().fit(X_hr, y_hr)
	fig_hr, ax_hr = plt.subplots()
ax_hr.scatter(hourly_data['Hour'], hourly_data['Crime Count'], label='Crimes per Hour')
	ax_hr.plot(hourly_data['Hour'], hour_model.predict(X_hr), color='red', label='Regression Line')
	ax_hr.set_xlabel("Hour")
	ax_hr.set_ylabel("Crime Count")
	ax_hr.set_title("Crime Distribution by Hour")
	ax_hr.legend()
	st.pyplot(fig_hr)

if st.button("📦 Boxplot: Victim Age by Top Crime Types"):
	st.subheader("Boxplot: Victim Age by Top Crime Types")
	top_crimes = filtered_df['Crm Cd Desc'].value_counts().head(5).index
	box_data = filtered_df[filtered_df['Crm Cd Desc'].isin(top_crimes)]
	fig_box, ax_box = plt.subplots(figsize=(10, 6))
	sns.boxplot(data=box_data, x='Crm Cd Desc', y='Vict Age', ax=ax_box)
	ax_box.set_title("Victim Age Distribution by Top Crime Types")
    ax_box.set_xticklabels(ax_box.get_xticklabels(), rotation=45, ha="right")
	st.pyplot(fig_box)

if st.button("🕒 Hourly Crime Pattern"):
	st.subheader("Hourly Crime Pattern")
	hourly = filtered_df.groupby('Hour').size()
	fig_hourly, ax_hourly = plt.subplots()
	hourly.plot(kind='line', ax=ax_hourly, marker='o')
	ax_hourly.set_xlabel("Hour")
	ax_hourly.set_ylabel("Number of Crimes")
	ax_hourly.set_title("Crime Count by Hour of the Day")
    st.pyplot(fig_hourly)
