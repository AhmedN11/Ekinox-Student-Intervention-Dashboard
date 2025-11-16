# Logging System Documentation

## Overview

The Student Intervention Dashboard includes a comprehensive logging system that tracks all operations, errors, and user interactions. Logs are visible in the terminal during execution and are also saved to files in the `Logs/` folder for later review.

## Log Files

### Location
All log files are stored in the `Logs/` directory at the root of the project.

### Log File Types

1. **dashboard.log** - Main application logs from `app.py`
   - User interactions
   - Data loading operations
   - Filter applications
   - Student profile views
   - Errors and warnings

2. **data_processor.log** - Data processing logs from `data_processor.py`
   - Student data processing operations
   - Score calculations
   - Urgency zone classifications
   - Processing statistics

## Log Rotation and Archiving

### Automatic Rotation
Log files are automatically rotated when they reach **10 MB** in size using Python's `RotatingFileHandler`.

### Archive Management
- **Backup Count**: 5 archived files are kept
- **Archive Naming**: Archives are automatically named with numeric suffixes
  - `dashboard.log` (current)
  - `dashboard.log.1` (most recent archive)
  - `dashboard.log.2`
  - `dashboard.log.3`
  - `dashboard.log.4`
  - `dashboard.log.5` (oldest archive)

### Archive Process
When a log file reaches 10 MB:
1. Current log is renamed to `.1`
2. Previous archives are incremented (`.1` → `.2`, `.2` → `.3`, etc.)
3. Oldest archive (`.5`) is deleted if it exists
4. New empty log file is created

## Log Levels

### Console Output (Terminal)
- **INFO**: General information about operations
- **WARNING**: Non-critical issues that should be noted
- **ERROR**: Critical errors that affect functionality

### File Output
- **DEBUG**: Detailed diagnostic information
- **INFO**: General information about operations
- **WARNING**: Non-critical issues
- **ERROR**: Critical errors with stack traces

## Log Format

Each log entry follows this format:
```
YYYY-MM-DD HH:MM:SS - LoggerName - LEVEL - Message
```

**Example:**
```
2025-11-16 14:04:09 - DataProcessor - INFO - Starting student data processing for 3 records
```

## What Gets Logged

### Application Events (dashboard.log)
- Application startup
- Data file selection and loading
- Number of students loaded
- Filter applications (grade range, intervention threshold, indicators)
- Student profile views
- Data loading errors
- Filter operations

### Data Processing Events (data_processor.log)
- Module initialization
- Data processing start/completion
- Intervention score calculations
- Priority score calculations
- Urgency zone classifications (High/Moderate/Monitor counts)
- Processing statistics (mean scores, max values, etc.)
- Data processing errors with stack traces

## Usage Examples

### Viewing Recent Logs
```powershell
# View last 20 lines of dashboard log
Get-Content Logs/dashboard.log -Tail 20

# View last 20 lines of data processor log
Get-Content Logs/data_processor.log -Tail 20
```

### Searching Logs
```powershell
# Find all error messages
Select-String -Path Logs/*.log -Pattern "ERROR"

# Find logs for a specific student
Select-String -Path Logs/dashboard.log -Pattern "STU001"

# Find logs from today
Get-Content Logs/dashboard.log | Select-String "2025-11-16"
```

### Monitoring Logs in Real-Time
```powershell
# Watch dashboard log in real-time
Get-Content Logs/dashboard.log -Wait

# Watch data processor log in real-time
Get-Content Logs/data_processor.log -Wait
```

### Checking Log File Sizes
```powershell
# Check sizes of all log files
Get-ChildItem Logs/ -Recurse | Select-Object Name, @{Name="Size(MB)";Expression={[math]::Round($_.Length/1MB, 2)}}
```

## Troubleshooting

### Log Files Not Created
1. Ensure the `Logs/` directory exists (it's created automatically)
2. Check write permissions on the project directory
3. Verify Python logging module is available

### Large Log Files
- Rotation occurs automatically at 10 MB
- If you want to manually archive: rename the log file and restart the application
- To reduce log size: decrease logging level in code

### Missing Log Entries
- Console shows INFO level and above
- File logs include DEBUG level for more detail
- Check if the operation completed successfully (errors would be logged)

## Best Practices

### For Administrators
1. **Regular Monitoring**: Check logs weekly for errors or unusual patterns
2. **Archive Management**: Archived logs (`.1` through `.5`) can be compressed or moved for long-term storage
3. **Error Response**: Review ERROR level logs promptly
4. **Storage**: Monitor disk space as logs accumulate

### For Developers
1. **Error Investigation**: Check both terminal output and log files for full context
2. **Debug Mode**: File logs include DEBUG level for detailed troubleshooting
3. **Stack Traces**: Errors are logged with full stack traces in files
4. **Performance**: Log rotation prevents unbounded growth

## Maintenance

### Clearing Old Logs
```powershell
# Remove all archived logs (keep current)
Remove-Item Logs/*.log.[1-5]

# Remove all logs and start fresh
Remove-Item Logs/*.log*
```

### Backing Up Logs
```powershell
# Create dated backup of all logs
$date = Get-Date -Format "yyyyMMdd"
Compress-Archive -Path Logs/*.log* -DestinationPath "LogBackup_$date.zip"
```

## Configuration

### Changing Log Rotation Settings

To modify rotation behavior, edit the `RotatingFileHandler` parameters in both files:

**In app.py and data_processor.py:**
```python
file_handler = RotatingFileHandler(
    log_file,
    maxBytes=10*1024*1024,  # Change this (currently 10 MB)
    backupCount=5,           # Change this (currently 5 archives)
    encoding='utf-8'
)
```

### Changing Log Levels

**Console output:**
```python
console_handler.setLevel(logging.INFO)  # Change to DEBUG, WARNING, or ERROR
```

**File output:**
```python
file_handler.setLevel(logging.DEBUG)  # Change to INFO, WARNING, or ERROR
```

## Technical Details

### Logging Architecture
- **Logger Names**: 
  - `StudentInterventionDashboard` (app.py)
  - `DataProcessor` (data_processor.py)
- **Handlers**: Console (StreamHandler) and File (RotatingFileHandler)
- **Formatter**: Timestamp, logger name, level, and message
- **Rotation**: Size-based (10 MB) with 5 backups

### Thread Safety
The logging system is thread-safe and can handle concurrent operations.

### Performance Impact
Logging has minimal performance impact:
- Console output: Negligible
- File writing: Asynchronous, non-blocking
- Rotation: Occurs only when threshold reached