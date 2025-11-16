# Student Intervention Prioritization Dashboard

A comprehensive Streamlit dashboard for the Portuguese Ministry of Education to identify and prioritize students requiring personalized support after school closures.

## Overview

This dashboard helps educational counselors prioritize students based on two key dimensions:
- **Current Performance**: Final grade in mathematics (0-20 scale)
- **Intervention Complexity**: How actionable/easy it is to help them based on 8 key indicators

## Features

### 🎯 Priority Matrix
- Interactive scatter plot showing all students positioned by grade vs. intervention score
- Color-coded urgency zones (High Priority, Moderate Priority, Monitor)
- Click on points to view detailed student profiles
- Quadrant lines showing critical thresholds

### 📊 Actionable Indicators
The system evaluates 8 key indicators for each student:
1. **High Absences**: Above median attendance issues
2. **Low Study Time**: Less than 2 hours per week
3. **Alcohol Consumption**: Workday or weekend consumption ≥ 3
4. **Past Failures**: Previous class failures
5. **No School Support**: Not receiving extra educational support
6. **No Family Support**: Lacking family educational support
7. **Poor Family Relations**: Family relationship quality < 3
8. **Health Issues**: Health status < 3

### 🔍 Advanced Filtering
- Filter by grade range
- Filter by minimum intervention score
- Filter by specific indicators
- View filtered student counts and percentages

### 📋 Student Profiles
- Detailed view of individual student data
- Personalized intervention recommendations
- Demographics and background information
- Visual indicators showing which areas need attention

### 📥 Data Export
- Export priority student lists to CSV
- Timestamped filenames for record-keeping

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone or download this repository**
   ```powershell
   cd student-intervention-dashboard
   ```

2. **Create a virtual environment (recommended)**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install required packages**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Create environment file**  
   Use the .env.example file in the project root to add your configuration variables to a `.env` file.

5. **Run the dashboard**
   ```powershell
   streamlit run app.py
   ```

6. **Access the dashboard**
   - The dashboard will open automatically in your default browser
   - If not, navigate to `http://localhost:8501`

## Data Format

Your `students.csv` file must include the following columns:

### Required Columns
- `StudentID`: Unique student identifier
- `FinalGrade`: Final math grade (0-20 scale)
- `absences`: Number of absences (0-93)
- `studytime`: Weekly study hours (1=<2h, 2=2-5h, 3=5-10h, 4=>10h)
- `Dalc`: Workday alcohol consumption (1-5)
- `Walc`: Weekend alcohol consumption (1-5)
- `failures`: Past class failures (0-4)
- `schoolsup`: Extra educational support (yes/no)
- `famsup`: Family educational support (yes/no)
- `famrel`: Quality of family relationships (1-5)
- `health`: Health status (1-5)

### Optional Columns (for demographics)
- `age`: Student age
- `sex`: Student gender (M/F)
- `address`: Address type (U=urban, R=rural)
- `Medu`: Mother's education level (0-4)
- `Fedu`: Father's education level (0-4)
- `famsize`: Family size (LE3=≤3, GT3=>3)
- `Pstatus`: Parent cohabitation status (T=together, A=apart)
- `paid`: Extra paid classes (yes/no)
- `activities`: Extra-curricular activities (yes/no)
- `internet`: Internet access at home (yes/no)

### Example Data Format
```csv
StudentID,FinalGrade,absences,studytime,Dalc,Walc,failures,schoolsup,famsup,famrel,health,age,sex,address
STU001,14,2,2,1,1,0,no,yes,4,5,18,F,U
STU002,8,15,1,3,4,1,yes,no,2,3,17,M,U
STU003,16,0,3,1,1,0,no,yes,5,4,18,F,R
```

## Project Structure

```
student-intervention-dashboard/
├── app.py                 # Main Streamlit application
├── data_processor.py      # Data processing and scoring functions
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── test_logging.py       # Logging test script
├── data/                 # Data folder (gitignored)
│   └── students.csv      # Your student data (not included)
└── Logs/                 # Log files (auto-created)
    ├── README.md         # Logging system documentation
    ├── dashboard.log     # Application logs
    └── data_processor.log # Processing logs
```

## Usage Guide

### 1. Initial View
Upon loading, you'll see:
- Priority matrix showing all students
- Top 30 priority students list
- Distribution of actionable indicators
- Summary statistics

### 2. Filtering Students
Use the sidebar to:
- Adjust grade range (e.g., focus on failing students)
- Set minimum intervention score threshold
- Select specific indicators to filter by
- Reset all filters with one click

### 3. Analyzing Individual Students
- Click on any point in the priority matrix, OR
- Use the dropdown selector to choose a student
- View detailed profile with:
  - Current grades and scores
  - Specific indicators flagged
  - Personalized intervention recommendations
  - Demographics and background

### 4. Exporting Data
- Click "Download Priority List (CSV)" button
- File will be saved as `priority_students_YYYYMMDD.csv`
- Contains all visible filtered students with calculated scores

## Understanding the Scores

### Intervention Score (0-1)
- **Higher = More Actionable**: Easier to help with concrete interventions
- Calculated as: (number of actionable indicators) / 8
- Score of 0.5+ indicates multiple actionable areas

### Priority Score
- **Higher = More Urgent**: Combined need and actionability
- Calculated as: (20 - FinalGrade) × intervention_score
- Balances academic need with intervention feasibility

### Urgency Zones
- **High Priority** (Red): Grade < 10 AND intervention score > 0.5
  - Failing students with many actionable interventions
- **Moderate Priority** (Orange): Grade < 12 AND intervention score > 0.3
  - At-risk students with some actionable interventions
- **Monitor** (Green): All other students
  - Passing or fewer actionable interventions needed

## Troubleshooting

### Dashboard won't start
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.8+)
- Verify virtual environment is activated

### Data file not found
- Check that `data/students.csv` exists
- Verify the file path is correct
- Ensure the CSV file is properly formatted

### Missing columns error
- Verify your CSV contains all required columns
- Check column names match exactly (case-sensitive)
- Ensure no extra spaces in column names

### Charts not displaying
- Try refreshing the browser
- Check browser console for JavaScript errors
- Ensure Plotly is properly installed

## Technical Details

### Logging System
The dashboard includes a comprehensive logging system that tracks all operations:

- **Terminal Logging**: View operations in real-time during execution
- **File Logging**: All logs saved to `Logs/` folder for review
- **Automatic Rotation**: Log files automatically rotate at 10 MB (keeps 5 archives)
- **Dual Log Files**: 
  - `dashboard.log` - Application logs (user interactions, data loading)
  - `data_processor.log` - Processing logs (calculations, classifications)

**View logs:**
```powershell
# View recent logs
Get-Content Logs/dashboard.log -Tail 20

# Monitor logs in real-time
Get-Content Logs/dashboard.log -Wait

# Search for errors
Select-String -Path Logs/*.log -Pattern "ERROR"
```

📖 **Full documentation**: See `Logs/README.md` for complete logging system details.

### Performance Optimization
- Data loading cached with `@st.cache_data`
- Score calculations cached to prevent recomputation
- Efficient filtering using pandas operations


Built with:
- [Streamlit](https://streamlit.io/) - Dashboard framework
- [Plotly](https://plotly.com/) - Interactive visualizations
- [Pandas](https://pandas.pydata.org/) - Data processing
- [NumPy](https://numpy.org/) - Numerical computing
