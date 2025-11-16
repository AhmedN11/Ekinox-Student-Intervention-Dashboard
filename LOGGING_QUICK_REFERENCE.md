# Logging Quick Reference Card

## 📋 Quick Commands

### View Logs
```powershell
# Last 20 lines
Get-Content Logs/dashboard.log -Tail 20
Get-Content Logs/data_processor.log -Tail 20

# All logs
Get-Content Logs/dashboard.log
Get-Content Logs/data_processor.log
```

### Monitor in Real-Time
```powershell
# Watch logs as they're written
Get-Content Logs/dashboard.log -Wait
Get-Content Logs/data_processor.log -Wait
```

### Search Logs
```powershell
# Find errors
Select-String -Path Logs/*.log -Pattern "ERROR"

# Find warnings
Select-String -Path Logs/*.log -Pattern "WARNING"

# Find specific student
Select-String -Path Logs/dashboard.log -Pattern "STU001"

# Find today's logs
Get-Content Logs/dashboard.log | Select-String "2025-11-16"
```

### Check Log Sizes
```powershell
# View sizes in KB
Get-ChildItem Logs/ | Select-Object Name, @{Name="Size(KB)";Expression={[math]::Round($_.Length/1KB, 2)}}

# View sizes in MB
Get-ChildItem Logs/ | Select-Object Name, @{Name="Size(MB)";Expression={[math]::Round($_.Length/1MB, 2)}}
```

### Maintenance
```powershell
# Remove old archives
Remove-Item Logs/*.log.[1-5]

# Backup logs
$date = Get-Date -Format "yyyyMMdd"
Compress-Archive -Path Logs/*.log* -DestinationPath "LogBackup_$date.zip"

# Clear all logs (use with caution!)
Remove-Item Logs/*.log*
```

## 📁 Log Files

| File | Purpose | Level |
|------|---------|-------|
| `dashboard.log` | Main app operations | INFO+ |
| `data_processor.log` | Data processing | INFO+ |
| `*.log.1` to `*.log.5` | Archived logs | - |

## 🔍 What Gets Logged

### dashboard.log
- Application startup
- Data file loading
- User interactions
- Filter applications
- Student selections
- Errors and warnings

### data_processor.log
- Data processing operations
- Score calculations
- Classifications
- Processing statistics
- Errors with stack traces

## ⚙️ Configuration

**Log Rotation:**
- Size: 10 MB per file
- Archives: 5 backups kept
- Automatic rotation

**Log Levels:**
- Console: INFO
- File: DEBUG
- Includes timestamps

## 📖 Full Documentation

See `Logs/README.md` for complete details.

## ⚠️ Important Notes

- Logs rotate automatically at 10 MB
- Old logs are NOT deleted (archived as .log.1, .log.2, etc.)
- Console shows INFO level, files have DEBUG level
- All timestamps in local time
- Log files excluded from git

## 💡 Tips

1. **Regular Monitoring**: Check logs weekly for issues
2. **Error Investigation**: Search for "ERROR" keyword
3. **Performance Tracking**: Review processing statistics
4. **Backup Strategy**: Archive logs monthly
5. **Disk Space**: Monitor log folder size

---

**Need help?** See `Logs/README.md` for detailed documentation.
