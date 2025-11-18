# Data Folder

**Note:** Placing files in this directory is **optional**. You can upload your student data (CSV, XLSX, or XLS) directly through the dashboard.

If you prefer to pre-load your data, you can place your data file in this directory.

## Required Data Format

Your data file (CSV, XLSX, or XLS) must include these columns:

### Core Columns (Required)
- StudentID
- FinalGrade
- absences
- studytime
- Dalc (workday alcohol consumption)
- Walc (weekend alcohol consumption)
- failures
- schoolsup
- famsup
- famrel
- health

### Optional Demographic & Enrichment Columns
Including more columns improves the data-driven indicator selection:

**Demographics:**
- age, sex, address

**Family Background:**
- Medu, Fedu (parent education)
- Mjob, Fjob (parent jobs)
- famsize, Pstatus (family size, parent status)
- guardian (primary guardian)

**School & Support:**
- traveltime, paid, activities, nursery, higher, internet, reason

**Lifestyle:**
- freetime, goout, romantic

**Note:** The dashboard automatically analyzes all available columns to select the most predictive indicators using correlation analysis. More data = better predictions! The system will:
- Test 37+ potential indicators across 8 categories
- Select the top 8 with strongest correlation to grades
- Display only these selected indicators in the dashboard
- Weight each indicator by its predictive power

## Example Format

```csv
StudentID,FinalGrade,absences,studytime,Dalc,Walc,failures,schoolsup,famsup,famrel,health,age,sex,address
STU001,14,2,2,1,1,0,no,yes,4,5,18,F,U
STU002,8,15,1,3,4,1,yes,no,2,3,17,M,U
STU003,16,0,3,1,1,0,no,yes,5,4,18,F,R
```

**Supported formats:** CSV, XLSX (Excel), XLS (Excel)

## Data Upload

✅ **Recommended:** Upload your data file directly through the dashboard interface for convenience.

## Data Privacy

⚠️ **Important**: This folder is configured to be ignored by version control (git).
Never commit actual student data to the repository.
