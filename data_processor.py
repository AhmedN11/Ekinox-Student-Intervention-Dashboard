"""
Data processing module for student intervention prioritization.

This module contains functions to calculate intervention scores, priority scores,
and classify students into urgency zones based on their academic performance
and actionable indicators.
"""

import pandas as pd
import numpy as np
from typing import Tuple
import logging
import os
from logging.handlers import RotatingFileHandler

# Setup logging
def setup_logging():
    """Configure logging for data processor module."""
    # Create logs directory if it doesn't exist
    log_dir = 'Logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create logger
    logger = logging.getLogger('DataProcessor')
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation (10MB max file size, keep 5 backup files)
    log_file = os.path.join(log_dir, 'data_processor.log')
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# Initialize logger
logger = setup_logging()
logger.info("Data Processor module initialized")


def calculate_intervention_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate an Intervention Complexity Score for each student.
    
    The score ranges from 0 to 1, where higher values indicate more actionable
    interventions (easier to help the student). The score is based on the presence
    of 8 actionable indicators:
    
    1. High absences (above median)
    2. Low study time (less than 2 hours per week)
    3. Alcohol consumption issues (workday or weekend >= 3)
    4. Past failures (one or more)
    5. No school support
    6. No family support
    7. Poor family relations (score < 3)
    8. Health issues (score < 3)
    
    Args:
        df: DataFrame containing student data with required columns
        
    Returns:
        DataFrame with added 'intervention_score' column
        
    Raises:
        KeyError: If required columns are missing from the DataFrame
    """
    logger.debug(f"Calculating intervention scores for {len(df)} students")
    df = df.copy()
    
    # Calculate median absences for comparison
    absences_median = df['absences'].median()
    
    # Initialize indicator flags
    indicators = pd.DataFrame(index=df.index)
    
    # 1. High absences (above median)
    indicators['high_absences'] = (df['absences'] > absences_median).astype(int)
    
    # 2. Low study time (studytime == 1, which means <2 hours/week)
    indicators['low_studytime'] = (df['studytime'] == 1).astype(int)
    
    # 3. Alcohol consumption issues (Dalc >= 3 OR Walc >= 3)
    indicators['alcohol_issues'] = ((df['Dalc'] >= 3) | (df['Walc'] >= 3)).astype(int)
    
    # 4. Past failures (failures > 0)
    indicators['past_failures'] = (df['failures'] > 0).astype(int)
    
    # 5. No school support (schoolsup == 'no')
    indicators['no_school_support'] = (df['schoolsup'] == 'no').astype(int)
    
    # 6. No family support (famsup == 'no')
    indicators['no_family_support'] = (df['famsup'] == 'no').astype(int)
    
    # 7. Poor family relations (famrel < 3)
    indicators['poor_family_relations'] = (df['famrel'] < 3).astype(int)
    
    # 8. Health issues (health < 3)
    indicators['health_issues'] = (df['health'] < 3).astype(int)
    
    # Calculate intervention score (sum of indicators / 8.0)
    df['intervention_score'] = indicators.sum(axis=1) / 8.0
    
    # Store individual indicator columns for later reference
    for col in indicators.columns:
        df[col] = indicators[col]
    
    logger.info(f"Intervention scores calculated. Mean score: {df['intervention_score'].mean():.3f}")
    return df


def calculate_priority_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate overall priority score for student intervention.
    
    The priority score combines the student's academic need (based on FinalGrade)
    with the intervention complexity score. Higher priority scores indicate
    students who both need help and can be helped with actionable interventions.
    
    Formula: priority_score = (20 - FinalGrade) * intervention_score
    
    Args:
        df: DataFrame with 'FinalGrade' and 'intervention_score' columns
        
    Returns:
        DataFrame with added 'priority_score' column
        
    Raises:
        KeyError: If required columns are missing from the DataFrame
    """
    logger.debug(f"Calculating priority scores for {len(df)} students")
    df = df.copy()
    
    # Calculate priority score
    # Higher FinalGrade deficit (20 - grade) combined with higher intervention score
    # gives higher priority
    df['priority_score'] = (20 - df['FinalGrade']) * df['intervention_score']
    
    logger.info(f"Priority scores calculated. Max priority score: {df['priority_score'].max():.3f}")
    return df


def classify_urgency_zone(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify students into urgency zones based on their grade and intervention score.
    
    Zones are defined as:
    - "High Priority": FinalGrade < 10 AND intervention_score > 0.5
      (Failing students with many actionable interventions)
    - "Moderate Priority": FinalGrade < 12 AND intervention_score > 0.3
      (At-risk students with some actionable interventions)
    - "Monitor": All other students
      (Passing students or those with fewer actionable interventions)
    
    Args:
        df: DataFrame with 'FinalGrade' and 'intervention_score' columns
        
    Returns:
        DataFrame with added 'urgency_zone' column
        
    Raises:
        KeyError: If required columns are missing from the DataFrame
    """
    logger.debug(f"Classifying urgency zones for {len(df)} students")
    df = df.copy()
    
    # Initialize urgency zone with default value
    df['urgency_zone'] = 'Monitor'
    
    # Classify High Priority students
    high_priority_mask = (df['FinalGrade'] < 10) & (df['intervention_score'] > 0.5)
    df.loc[high_priority_mask, 'urgency_zone'] = 'High Priority'
    
    # Classify Moderate Priority students (excluding those already marked as High Priority)
    moderate_priority_mask = (
        (df['FinalGrade'] < 12) & 
        (df['intervention_score'] > 0.3) & 
        (~high_priority_mask)
    )
    df.loc[moderate_priority_mask, 'urgency_zone'] = 'Moderate Priority'
    
    high_count = high_priority_mask.sum()
    moderate_count = moderate_priority_mask.sum()
    monitor_count = len(df) - high_count - moderate_count
    logger.info(f"Urgency classification complete: High Priority={high_count}, Moderate Priority={moderate_count}, Monitor={monitor_count}")
    
    return df


def get_actionable_indicators_list(row: pd.Series) -> str:
    """
    Generate a comma-separated list of actionable indicators present for a student.
    
    Args:
        row: Series containing indicator columns for a single student
        
    Returns:
        Comma-separated string of indicator names present for the student
    """
    indicators = []
    
    if row.get('high_absences', 0) == 1:
        indicators.append('High Absences')
    if row.get('low_studytime', 0) == 1:
        indicators.append('Low Study Time')
    if row.get('alcohol_issues', 0) == 1:
        indicators.append('Alcohol Issues')
    if row.get('past_failures', 0) == 1:
        indicators.append('Past Failures')
    if row.get('no_school_support', 0) == 1:
        indicators.append('No School Support')
    if row.get('no_family_support', 0) == 1:
        indicators.append('No Family Support')
    if row.get('poor_family_relations', 0) == 1:
        indicators.append('Poor Family Relations')
    if row.get('health_issues', 0) == 1:
        indicators.append('Health Issues')
    
    return ', '.join(indicators) if indicators else 'None'


def get_top_indicators(row: pd.Series, top_n: int = 3) -> str:
    """
    Get the top N actionable indicators for a student.
    
    Args:
        row: Series containing indicator columns for a single student
        top_n: Number of top indicators to return (default: 3)
        
    Returns:
        Comma-separated string of top N indicator names
    """
    indicators = []
    
    if row.get('high_absences', 0) == 1:
        indicators.append('High Absences')
    if row.get('low_studytime', 0) == 1:
        indicators.append('Low Study Time')
    if row.get('alcohol_issues', 0) == 1:
        indicators.append('Alcohol Issues')
    if row.get('past_failures', 0) == 1:
        indicators.append('Past Failures')
    if row.get('no_school_support', 0) == 1:
        indicators.append('No School Support')
    if row.get('no_family_support', 0) == 1:
        indicators.append('No Family Support')
    if row.get('poor_family_relations', 0) == 1:
        indicators.append('Poor Family Relations')
    if row.get('health_issues', 0) == 1:
        indicators.append('Health Issues')
    
    return ', '.join(indicators[:top_n]) if indicators else 'None'


def count_actionable_indicators(row: pd.Series) -> int:
    """
    Count the number of actionable indicators present for a student.
    
    Args:
        row: Series containing indicator columns for a single student
        
    Returns:
        Integer count of actionable indicators
    """
    indicator_cols = [
        'high_absences', 'low_studytime', 'alcohol_issues', 'past_failures',
        'no_school_support', 'no_family_support', 'poor_family_relations', 'health_issues'
    ]
    
    count = sum(row.get(col, 0) for col in indicator_cols)
    return int(count)


def process_student_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw student data to calculate all scores and classifications.
    
    This is the main processing function that applies all transformations:
    1. Calculates intervention score
    2. Calculates priority score
    3. Classifies urgency zones
    4. Adds helper columns for indicators
    
    Args:
        df: Raw student DataFrame
        
    Returns:
        Processed DataFrame with all calculated columns
        
    Raises:
        Exception: If processing fails at any step
    """
    logger.info(f"Starting student data processing for {len(df)} records")
    try:
        # Handle missing values
        df = df.copy()
        
        # Fill numeric missing values with median
        numeric_cols = ['absences', 'studytime', 'Dalc', 'Walc', 'failures', 'famrel', 'health']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # Fill categorical missing values with mode or default
        if 'schoolsup' in df.columns:
            df['schoolsup'] = df['schoolsup'].fillna(df['schoolsup'].mode()[0] if not df['schoolsup'].mode().empty else 'no')
        if 'famsup' in df.columns:
            df['famsup'] = df['famsup'].fillna(df['famsup'].mode()[0] if not df['famsup'].mode().empty else 'no')
        
        # Calculate scores
        df = calculate_intervention_score(df)
        df = calculate_priority_score(df)
        df = classify_urgency_zone(df)
        
        # Add helper columns
        logger.debug("Adding helper columns for indicators")
        df['num_indicators'] = df.apply(count_actionable_indicators, axis=1)
        df['indicators_list'] = df.apply(get_actionable_indicators_list, axis=1)
        df['top_indicators'] = df.apply(lambda x: get_top_indicators(x, 3), axis=1)
        
        logger.info(f"Data processing complete successfully. Processed {len(df)} students with {df.columns.size} features")
        return df
        
    except Exception as e:
        logger.error(f"Error processing student data: {str(e)}", exc_info=True)
        raise Exception(f"Error processing student data: {str(e)}")


def get_suggested_interventions(row: pd.Series) -> list:
    """
    Generate suggested interventions based on student's actionable indicators.
    
    Args:
        row: Series containing indicator columns for a single student
        
    Returns:
        List of intervention recommendations
    """
    logger.debug(f"Generating interventions for student {row.get('StudentID', 'unknown')}")
    interventions = []
    
    if row.get('high_absences', 0) == 1:
        interventions.append("📞 Contact family about attendance patterns and barriers")
    
    if row.get('low_studytime', 0) == 1:
        interventions.append("📚 Enroll in study skills workshop and time management training")
    
    if row.get('alcohol_issues', 0) == 1:
        interventions.append("🏥 Refer to school counseling services for substance use support")
    
    if row.get('past_failures', 0) == 1:
        interventions.append("🎯 Provide targeted tutoring in subjects with previous failures")
    
    if row.get('no_school_support', 0) == 1:
        interventions.append("🏫 Enroll in school tutoring program and extra educational support")
    
    if row.get('no_family_support', 0) == 1:
        interventions.append("👨‍👩‍👧 Engage family support services and parent education programs")
    
    if row.get('poor_family_relations', 0) == 1:
        interventions.append("💬 Offer family counseling and relationship building support")
    
    if row.get('health_issues', 0) == 1:
        interventions.append("⚕️ Coordinate with health services for medical evaluation and support")
    
    if not interventions:
        interventions.append("✅ Student shows no critical intervention needs - continue monitoring")
    
    return interventions
