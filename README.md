# Student Intervention Prioritization Dashboard

A comprehensive Streamlit dashboard for the Portuguese Ministry of Education to identify and prioritize students requiring personalized support after school closures.

## Overview

This dashboard helps educational counselors prioritize students based on two key dimensions:
- **Current Performance**: Final grade in mathematics (0-20 scale)
- **Intervention Complexity**: How actionable/easy it is to help them based on data-driven indicator selection

### Data-Driven Methodology

The dashboard employs an **exhaustive correlation-based approach** to identify the most impactful intervention indicators:

1. **Comprehensive Testing**: Analyzes 37+ potential indicators across 8 categories (demographics, family, school behavior, support, activities, leisure, health, motivation)
2. **Correlation Analysis**: Calculates Pearson correlation between each indicator and student grades (FinalGrade)
3. **Smart Selection**: Automatically selects the top 8 indicators with highest absolute correlation
4. **Weighted Scoring**: Each indicator is weighted proportionally to its predictive power (correlation strength)
5. **Validation**: Final intervention score is validated against actual grades to ensure statistical significance

This data-driven approach ensures that the most statistically significant factors are prioritized over domain assumptions, resulting in more accurate student prioritization.

## Features

### 🎯 Priority Matrix
- Interactive scatter plot showing all students positioned by grade vs. intervention score
- Color-coded urgency zones (High Priority, Moderate Priority, Monitor)
- Click on points to view detailed student profiles
- Quadrant lines showing critical thresholds

### 📊 Actionable Indicators

The system **dynamically selects** the top 8 most predictive indicators from a comprehensive set of 37+ potential factors:

**Indicator Categories Analyzed:**
- **Demographic**: Age, gender, address (urban/rural)
- **Family**: Education levels, jobs, family size, parent status, family relations
- **School Behavior**: Absences, study time, travel time, past failures
- **Support Systems**: School support, family support, paid classes
- **Activities**: Extra-curricular activities, internet access, nursery, higher education aspirations
- **Leisure**: Free time, social activities
- **Health**: Alcohol consumption (various thresholds), general health status
- **Motivation**: Reason for school choice

**Typical Top Indicators** (based on correlation with grades):
1. **Past Failures** (corr ≈ -0.28): Previous class failures strongly predict current performance
2. **No School Support** (corr ≈ +0.24): Lack of tutoring/extra help correlates with lower grades
3. **High Absences** (corr ≈ -0.20): Frequent absences (>10 days) impact learning
4. **Weekend Alcohol** (corr ≈ -0.18): Alcohol consumption affects academic focus
5. **Low Study Time** (corr ≈ -0.18): Insufficient study hours (≤2/week) limits progress
6. **High Social Activity** (corr ≈ -0.18): Excessive going out time detracts from studies
7-8. Additional indicators weighted by correlation strength

*Note: The specific indicators and their weights are recalculated for each dataset to ensure optimal accuracy.*

**Dashboard Display**: The "Actionable Indicators Overview" chart dynamically shows only the top 8 selected indicators for your dataset, with:
- Prevalence percentages (% of students affected)
- Color-coded bars by correlation strength
- Hover tooltips displaying correlation values and weights
- Different datasets will show different indicators based on what statistically matters most

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

### Option 1: Docker Deployment (Recommended)

#### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- [Docker Compose](https://docs.docker.com/compose/install/) (included with Docker Desktop)

#### Quick Start with Docker

1. **Clone or download this repository**
   ```bash
   cd Ekinox-Student-Intervention-Dashboard
   ```

2. **(Optional) Create environment file for Docker**
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit .env file if you want to pre-configure LLM settings
   nano .env
   # or
   vim .env
   ```
   
   **Note about DOCKER_ENV:** When launching with `docker-compose`, the `DOCKER_ENV` variable is automatically set to `true` in the docker-compose.yml file, so you don't need to manually set it in your `.env` file.
   
   **Note:** LLM configuration (API keys, model names) can be done directly from the dashboard homepage instead of editing the `.env` file. However, if you prefer to pre-configure your LLM settings:
   
   **For Local Ollama (No API key required):**
   ```env
   OLLAMA_MODEL_NAME=ollama/llama2
   OLLAMA_BASE_URL=http://ollama:11434
   ```
   
   **For Cloud Providers (e.g., Mistral, OpenAI):**
   ```env
   MISTRAL_API_KEY=your_api_key_here
   MISTRAL_MODEL_NAME=mistral/codestral-2508
   ```

3. **(Optional) Place your data file in the data folder**
   ```bash
   # You can optionally pre-load your data file
   cp your_students_file.csv data/
   # Or upload it directly through the dashboard interface
   ```
   
   **Note:** You can upload your student data (CSV, XLSX, or XLS) directly from the dashboard interface.

4. **Start the application**
   ```bash
   # Build and start all services
   docker-compose up --build
   
   # Or run in detached mode (background)
   docker-compose up -d --build
   ```

5. **Access the dashboard**
   - Open your browser and navigate to: `http://localhost:8501`
   - Configure your LLM provider directly from the homepage settings
   - The application will be ready in 10-30 seconds

6. **If using Ollama, pull a model (first time only)**
   ```bash
   # Connect to the Ollama container
   docker exec -it ollama-service ollama pull llama2
   
   # Or pull other models
   docker exec -it ollama-service ollama pull mistral
   docker exec -it ollama-service ollama pull codellama
   ```

#### Docker Management Commands

```bash
# Stop the application
docker-compose down

# Stop and remove all data (including Ollama models)
docker-compose down -v

# View logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f streamlit-app
docker-compose logs -f ollama

# Restart the application
docker-compose restart

# Rebuild after code changes
docker-compose up --build
```

#### Using Without Ollama (Cloud LLMs Only)

If you're using cloud-based LLMs (Mistral, OpenAI, Anthropic, Groq) and don't need local Ollama:

**Option A: Use the cloud-specific compose file**
```bash
docker-compose -f docker-compose.cloud.yml up --build
```

**Option B: Run only the streamlit service**
```bash
docker-compose up streamlit-app --build
```

---

### Option 2: Local Python Installation

#### Prerequisites
- Python 3.8 or higher
- pip package manager

#### Setup Steps

1. **Clone or download this repository**
   ```bash
   cd Ekinox-Student-Intervention-Dashboard
   ```

2. **Create a virtual environment (recommended)**
   
   **On Windows (PowerShell):**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   
   **On Windows (Command Prompt):**
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate.bat
   ```
   
   **On macOS/Linux:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Install and run Ollama locally**
   - Download from: https://ollama.ai/
   - Run: `ollama serve`
   - Pull a model: `ollama pull llama2`

5. **Run the dashboard**
   ```bash
   streamlit run app.py
   ```

6. **Access the dashboard**
   - The dashboard will open automatically in your default browser
   - If not, navigate to `http://localhost:8501`
   - Configure your LLM provider directly from the homepage settings

## Data Format

Your student data file (CSV, XLSX, or XLS) can be uploaded directly through the dashboard interface and must include the following columns:

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
Ekinox-Student-Intervention-Dashboard/
├── app.py                    # Main Streamlit application
├── data_processor.py         # Data processing and scoring functions
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker image configuration
├── docker-compose.yml        # Docker services orchestration
├── .dockerignore            # Files to exclude from Docker build
├── .env.example             # Environment variables template
├── README.md                # This file
├── test_logging.py          # Logging test script
├── data/                    # Data folder
│   └── students.csv         # Your student data
└── Logs/                    # Log files (auto-created)
    ├── README.md            # Logging system documentation
    ├── dashboard.log        # Application logs
    └── data_processor.log   # Processing logs
```

## Usage Guide

### 1. Upload Your Data
- Upload your student data file (CSV, XLSX, or XLS format) using the file uploader on the homepage
- Alternatively, you can place a data file in the `data/` folder before starting the application

### 2. Initial View
After loading data, you'll see:
- Priority matrix showing all students
- Top 30 priority students list
- **Actionable Indicators Overview**: Bar chart dynamically displaying the top 8 indicators selected for your dataset
  - Shows prevalence percentage for each indicator
  - Hover over bars to see correlation strength and weight
  - Indicators vary by dataset based on statistical analysis
- Summary statistics

### 3. Filtering Students
Use the sidebar to:
- Adjust grade range (e.g., focus on failing students)
- Set minimum intervention score threshold
- Select specific indicators to filter by
- Reset all filters with one click

### 4. Analyzing Individual Students
- Click on any point in the priority matrix, OR
- Use the dropdown selector to choose a student
- View detailed profile with:
  - Current grades and scores
  - Specific indicators flagged
  - Personalized intervention recommendations
  - Demographics and background

### 5. Exporting Data
- Click "Download Priority List (CSV)" button
- File will be saved as `priority_students_YYYYMMDD.csv`
- Contains all visible filtered students with calculated scores

## Understanding the Scores

### Intervention Score (0-1)
- **Higher = More Actionable**: Easier to help with concrete interventions
- **Calculation Method**: Weighted sum of top 8 indicators, where each indicator's contribution is proportional to its correlation strength with grades
  - Formula: `Σ(indicator_present × weight_i)` where weights sum to 1.0
  - Weights determined by: `weight_i = |correlation_i| / Σ|correlations|`
- **Score Interpretation**:
  - 0.0-0.3: Few actionable interventions
  - 0.3-0.5: Moderate intervention opportunities
  - 0.5+: Multiple high-impact intervention areas
- **Validation**: Correlation with grades typically ≈ -0.25 to -0.30 (strong negative relationship confirms predictive validity)

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

### Docker Issues

#### Container won't start
```bash
# Check if Docker is running
docker ps

# Check logs for errors
docker-compose logs

# Rebuild containers
docker-compose down
docker-compose up --build
```

#### Ollama service not accessible
```bash
# Check if Ollama container is running
docker ps | grep ollama

# Check Ollama logs
docker-compose logs ollama

# Test Ollama API
curl http://localhost:11434/api/tags
```

#### Port already in use
```bash
# Find process using port 8501
lsof -i :8501
# or
netstat -tuln | grep 8501

# Change port in docker-compose.yml
# ports:
#   - "8502:8501"  # Use different host port
```

#### Volume/data persistence issues
```bash
# Check volumes
docker volume ls

# Remove all volumes and start fresh
docker-compose down -v
docker-compose up --build
```

### Application Issues

#### Dashboard won't start (Local Python)
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.8+)
- Verify virtual environment is activated

#### Data file not found
- Use the file uploader on the dashboard homepage to upload your data file (CSV, XLSX, or XLS)
- Alternatively, place a file in the `data/` folder
- Ensure the file is properly formatted with required columns
- In Docker: Check volume mount is correct if using the data folder

#### Missing columns error
- Verify your CSV contains all required columns
- Check column names match exactly (case-sensitive)
- Ensure no extra spaces in column names

#### Charts not displaying
- Try refreshing the browser
- Check browser console for JavaScript errors
- Ensure Plotly is properly installed

#### LLM/AI features not working

**For Ollama:**
```bash
# Verify Ollama is running
docker exec -it ollama-service ollama list

# Pull a model if none available
docker exec -it ollama-service ollama pull llama2

# Check Ollama logs
docker-compose logs ollama
```

**For Cloud APIs:**
- Verify API key is set in `.env` file
- Check API key has proper permissions
- Verify model name is correct
- Check network connectivity

## Technical Details

### Docker Architecture

The application is containerized using Docker with the following architecture:

**Services:**
- **streamlit-app**: Main application container running the Streamlit dashboard
  - Exposes port 8501
  - Mounts volumes for data persistence (data/, Logs/)
  - Automatically detects Docker environment and adjusts Ollama URL
  
- **ollama** (optional): Local LLM service container
  - Exposes port 11434
  - Stores downloaded models in persistent volume
  - Accessible by streamlit-app via Docker network as `http://ollama:11434`

**Networking:**
- Both containers connected via `dashboard-network` bridge network
- Internal DNS resolution allows service-to-service communication
- Ollama accessible from host at `localhost:11434`
- Dashboard accessible from host at `localhost:8501`

**Volume Mounts:**
- `./data:/app/data` - Student data files (bind mount)
- `./Logs:/app/Logs` - Application logs (bind mount)
- `./.env:/app/.env` - Environment variables (bind mount, read-only)
- `ollama-data:/root/.ollama` - Ollama models (named volume)

**Environment Detection:**
The application automatically detects when running in Docker via the `DOCKER_ENV` environment variable and adjusts the Ollama base URL accordingly:
- **Docker**: `http://ollama:11434` (service name)
- **Local**: `http://localhost:11434` (localhost)

### Logging System
The dashboard includes a comprehensive logging system that tracks all operations:

- **Terminal Logging**: View operations in real-time during execution
- **File Logging**: All logs saved to `Logs/` folder for review
- **Automatic Rotation**: Log files automatically rotate at 10 MB (keeps 5 archives)
- **Dual Log Files**: 
  - `dashboard.log` - Application logs (user interactions, data loading)
  - `data_processor.log` - Processing logs (calculations, classifications)

**View logs:**
```bash
# View recent logs
tail -n 20 Logs/dashboard.log

# Monitor logs in real-time
tail -f Logs/dashboard.log

# Search for errors
grep "ERROR" Logs/*.log
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
