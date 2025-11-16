# Data Folder

Place your `students.csv` file in this directory.

## Required CSV Format

Your CSV file must include these columns:

### Core Columns (Required)
- StudentID
- FinalGrade
- absences
- studytime
- Dalc
- Walc
- failures
- schoolsup
- famsup
- famrel
- health

### Optional Demographic Columns
- age
- sex
- address
- Medu
- Fedu
- famsize
- Pstatus
- paid
- activities
- internet

## Example

```csv
StudentID,FinalGrade,absences,studytime,Dalc,Walc,failures,schoolsup,famsup,famrel,health,age,sex,address
STU001,14,2,2,1,1,0,no,yes,4,5,18,F,U
STU002,8,15,1,3,4,1,yes,no,2,3,17,M,U
STU003,16,0,3,1,1,0,no,yes,5,4,18,F,R
```

## Data Privacy

⚠️ **Important**: This folder is configured to be ignored by version control (git).
Never commit actual student data to the repository.
