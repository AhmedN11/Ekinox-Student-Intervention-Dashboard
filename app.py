"""
Student Intervention Prioritization Dashboard

A Streamlit dashboard for the Portuguese Ministry of Education to identify
and prioritize students requiring personalized support after school closures.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from smolagents import tool
import numpy as np
from datetime import datetime
import os
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
from data_processor import (
    process_student_data,
    get_suggested_interventions,
    count_actionable_indicators
)

# Load environment variables
load_dotenv()

# Helper function to get Ollama base URL
def get_ollama_base_url():
    """
    Get the appropriate Ollama base URL based on environment.
    Returns:
        str: Ollama API endpoint URL
    """
    # Priority 1: Check if OLLAMA_BASE_URL is explicitly set
    ollama_url = os.getenv('OLLAMA_BASE_URL')
    if ollama_url:
        return ollama_url
    
    # Priority 2: Check if running in Docker
    if os.getenv('DOCKER_ENV') == 'true':
        return 'http://ollama:11434'
    
    # Priority 3: Default to localhost for local development
    return 'http://localhost:11434'

# Setup logging
def setup_logging():
    """Configure logging with both console and file handlers with rotation."""
    # Create logs directory if it doesn't exist
    log_dir = 'Logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create logger
    logger = logging.getLogger('StudentInterventionDashboard')
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
    log_file = os.path.join(log_dir, 'dashboard.log')
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
logger.info("="*80)
logger.info("Student Intervention Dashboard Starting")
logger.info("="*80)

try:
    from smolagents import CodeAgent, LiteLLMModel
    SMOLAGENTS_AVAILABLE = True
    logger.info("smolagents library loaded successfully")
except ImportError as e:
    SMOLAGENTS_AVAILABLE = False
    logger.warning(f"smolagents not available: {e}")


# Page configuration
st.set_page_config(
    page_title="Student Intervention Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Color scheme constants
COLORS = {
    'High Priority': '#d32f2f',
    'Moderate Priority': '#f57c00',
    'Monitor': '#388e3c'
}

# Data value decoders
def decode_studytime(value):
    """Convert studytime numeric code to description."""
    mapping = {
        1: '<2 hours',
        2: '2 to 5 hours',
        3: '5 to 10 hours',
        4: '>10 hours'
    }
    return mapping.get(value, f'Level {value}')

def decode_traveltime(value):
    """Convert traveltime numeric code to description."""
    mapping = {
        1: '<15 min',
        2: '15 to 30 min',
        3: '30 min to 1 hour',
        4: '>1 hour'
    }
    return mapping.get(value, f'Level {value}')

def decode_education(value):
    """Convert education numeric code to description."""
    mapping = {
        0: 'None',
        1: 'Primary (4th grade)',
        2: '5th to 9th grade',
        3: 'Secondary',
        4: 'Higher education'
    }
    return mapping.get(value, 'N/A')

def decode_rating(value, scale_type='quality'):
    """Convert 1-5 rating to description."""
    if scale_type == 'quality':
        mapping = {1: 'Very bad', 2: 'Bad', 3: 'Fair', 4: 'Good', 5: 'Excellent'}
    elif scale_type == 'health':
        mapping = {1: 'Very bad', 2: 'Bad', 3: 'Fair', 4: 'Good', 5: 'Very good'}
    elif scale_type == 'consumption':
        mapping = {1: 'Very low', 2: 'Low', 3: 'Moderate', 4: 'High', 5: 'Very high'}
    elif scale_type == 'frequency':
        mapping = {1: 'Very low', 2: 'Low', 3: 'Moderate', 4: 'High', 5: 'Very high'}
    else:
        mapping = {1: 'Very low', 2: 'Low', 3: 'Moderate', 4: 'High', 5: 'Very high'}
    return mapping.get(value, 'N/A')

def decode_address(value):
    """Convert address code to description."""
    return 'Urban' if value == 'U' else 'Rural' if value == 'R' else value

def decode_famsize(value):
    """Convert famsize code to description."""
    return '≤3 members' if value == 'LE3' else '>3 members' if value == 'GT3' else value

def decode_pstatus(value):
    """Convert parent status code to description."""
    return 'Living together' if value == 'T' else 'Living apart' if value == 'A' else value


@tool
@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load and process student data from CSV file.
    
    Args:
        file_path: Path to the CSV file containing student data
        
    Returns:
        Processed DataFrame with all calculated scores and classifications
        
    Raises:
        FileNotFoundError: If the data file doesn't exist
        Exception: If data processing fails
    """
    logger.info(f"Loading data from: {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully read CSV file with {len(df)} rows")
        df = process_student_data(df)
        logger.info(f"Data processing complete. Final dataset: {len(df)} students")
        return df
    except FileNotFoundError:
        logger.error(f"Data file not found at {file_path}")
        raise FileNotFoundError(f"Data file not found at {file_path}")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}", exc_info=True)
        raise Exception(f"Error loading data: {str(e)}")


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Apply user-selected filters to the student dataframe.
    
    Args:
        df: Full student DataFrame
        filters: Dictionary containing filter parameters
        
    Returns:
        Filtered DataFrame
    """
    logger.debug(f"Applying filters: {filters}")
    filtered_df = df.copy()
    
    # Grade range filter
    filtered_df = filtered_df[
        (filtered_df['FinalGrade'] >= filters['grade_range'][0]) &
        (filtered_df['FinalGrade'] <= filters['grade_range'][1])
    ]
    
    # Intervention score threshold
    filtered_df = filtered_df[
        filtered_df['intervention_score'] >= filters['intervention_threshold']
    ]
    
    # Specific indicator filters
    if filters['selected_indicators']:
        indicator_map = {
            'High absences': 'high_absences',
            'Low study time': 'low_studytime',
            'Alcohol consumption': 'alcohol_issues',
            'Past failures': 'past_failures',
            'No school support': 'no_school_support',
            'No family support': 'no_family_support',
            'Poor family relations': 'poor_family_relations',
            'Health issues': 'health_issues'
        }
        
        for indicator in filters['selected_indicators']:
            col_name = indicator_map[indicator]
            filtered_df = filtered_df[filtered_df[col_name] == 1]
    
    logger.info(f"Filtered dataset: {len(filtered_df)} students (from {len(df)} total)")
    return filtered_df


def create_priority_matrix(df: pd.DataFrame) -> go.Figure:
    """
    Create interactive scatter plot showing student priority matrix.
    
    Args:
        df: Student DataFrame with calculated scores
        
    Returns:
        Plotly figure object
    """
    # Prepare hover data
    hover_data = {
        'StudentID': True,
        'FinalGrade': ':.1f',
        'intervention_score': ':.2f',
        'priority_score': ':.2f',
        'top_indicators': True
    }
    
    # Create scatter plot
    fig = px.scatter(
        df,
        x='FinalGrade',
        y='intervention_score',
        color='urgency_zone',
        size='num_indicators',
        hover_data=hover_data,
        color_discrete_map=COLORS,
        labels={
            'FinalGrade': 'Final Grade (0-20)',
            'intervention_score': 'Intervention Score (0-1)',
            'urgency_zone': 'Urgency Zone'
        },
        title='Student Priority Matrix'
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            title='Final Grade (0-20)',
            range=[-0.5, 20.5],
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title='Intervention Score (0-1)',
            range=[-0.05, 1.05],
            gridcolor='lightgray'
        )
    )
    
    # Add quadrant lines
    # Vertical line at FinalGrade = 10 (passing threshold)
    fig.add_vline(
        x=10,
        line_dash="dash",
        line_color="gray",
        annotation_text="Passing Threshold",
        annotation_position="top"
    )
    
    # Horizontal line at intervention_score = 0.5
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="gray",
        annotation_text="High Actionability",
        annotation_position="right"
    )
    
    # Update marker size
    fig.update_traces(
        marker=dict(
            size=df['num_indicators'] * 3 + 5,
            line=dict(width=1, color='white'),
            opacity=0.7
        )
    )
    
    return fig


def create_indicator_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create bar chart showing distribution of actionable indicators.
    Dynamically displays the top indicators selected by correlation analysis.
    
    Args:
        df: Student DataFrame with indicator columns and top_indicators metadata
        
    Returns:
        Plotly figure object
    """
    # Get dynamically selected indicators from dataframe metadata
    top_indicators = df.attrs.get('top_indicators', [])
    
    if not top_indicators:
        # Fallback to legacy indicators if metadata not available
        indicator_map = {
            'high_absences': 'High Absences',
            'low_studytime': 'Low Study Time',
            'alcohol_issues': 'Alcohol Issues',
            'past_failures': 'Past Failures',
            'no_school_support': 'No School Support',
            'no_family_support': 'No Family Support',
            'poor_family_relations': 'Poor Family Relations',
            'health_issues': 'Health Issues'
        }
        indicators_to_show = [(col, label) for col, label in indicator_map.items() if col in df.columns]
    else:
        # Use dynamically selected indicators
        indicators_to_show = [(ind['name'], ind['display_name']) for ind in top_indicators]
    
    # Calculate percentages
    percentages = []
    labels = []
    colors = []
    hover_texts = []
    
    for col, label in indicators_to_show:
        if col in df.columns:
            percentage = (df[col].sum() / len(df)) * 100
            percentages.append(percentage)
            labels.append(label)
            
            # Color code by correlation if available
            if top_indicators:
                ind_meta = next((ind for ind in top_indicators if ind['name'] == col), None)
                if ind_meta:
                    # Stronger correlation = darker color
                    colors.append(f"rgba(31, 119, 180, {0.5 + abs(ind_meta['correlation']) * 0.5})")
                    hover_texts.append(
                        f"{label}<br>" +
                        f"Prevalence: {percentage:.1f}%<br>" +
                        f"Correlation: {ind_meta['correlation']:.3f}<br>" +
                        f"Weight: {ind_meta['weight']*100:.1f}%"
                    )
                else:
                    colors.append('#1f77b4')
                    hover_texts.append(f"{label}<br>Prevalence: {percentage:.1f}%")
            else:
                colors.append('#1f77b4')
                hover_texts.append(f"{label}<br>Prevalence: {percentage:.1f}%")
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=percentages,
            y=labels,
            orientation='h',
            marker=dict(color=colors),
            text=[f'{p:.1f}%' for p in percentages],
            textposition='outside',
            hovertext=hover_texts,
            hoverinfo='text'
        )
    ])
    
    title = 'Top 8 Indicators Selected by Correlation Analysis' if top_indicators else 'Distribution of Actionable Indicators'
    
    fig.update_layout(
        title=title,
        xaxis_title='Percentage of Students (%)',
        yaxis_title='Indicator',
        height=400,
        showlegend=False,
        xaxis=dict(range=[0, max(percentages) * 1.15] if percentages else [0, 100])
    )
    
    return fig


def display_student_profile(student_row: pd.Series, df: pd.DataFrame):
    """
    Display detailed profile for a selected student.
    
    Args:
        student_row: Series containing data for the selected student
        df: Full DataFrame for context
    """
    logger.info(f"Displaying profile for student: {student_row['StudentID']}")
    # st.markdown("---")
    
    # Create student name display
    student_name = f"{student_row.get('FirstName', '')} {student_row.get('FamilyName', '')}".strip()
    student_gender = student_row.get('sex', 'N/A')
    gender_symbol = '👩' if student_gender == 'F' else '👨' if student_gender == 'M' else '👤'
    
    if student_name:
        st.subheader(f"📋 Student Profile: {gender_symbol} {student_name} ({student_row['StudentID']})")
    else:
        st.subheader(f"📋 Student Profile: {gender_symbol} {student_row['StudentID']}")
    
    # Student overview card
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Final Grade", f"{student_row['FinalGrade']:.1f}/20")
    
    with col2:
        urgency_color = COLORS.get(student_row['urgency_zone'], '#gray')
        st.markdown(
            f"<div style='padding: 10px; background-color: {urgency_color}; "
            f"color: white; border-radius: 5px; text-align: center;'>"
            f"<strong>{student_row['urgency_zone']}</strong></div>",
            unsafe_allow_html=True
        )
    
    with col3:
        st.metric("Priority Score", f"{student_row['priority_score']:.2f}")
    
    with col4:
        st.metric("Intervention Score", f"{student_row['intervention_score']:.2f}")
    
    # Create two columns for detailed information
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("#### 🎯 Actionable Indicators")
        
        # Get decoded values
        studytime_val = student_row.get('studytime')
        studytime_desc = decode_studytime(studytime_val) if pd.notna(studytime_val) else 'N/A'
        
        dalc_val = student_row.get('Dalc')
        walc_val = student_row.get('Walc')
        dalc_desc = decode_rating(dalc_val, 'consumption') if pd.notna(dalc_val) else 'N/A'
        walc_desc = decode_rating(walc_val, 'consumption') if pd.notna(walc_val) else 'N/A'
        
        famrel_val = student_row.get('famrel')
        famrel_desc = decode_rating(famrel_val, 'quality') if pd.notna(famrel_val) else 'N/A'
        
        health_val = student_row.get('health')
        health_desc = decode_rating(health_val, 'health') if pd.notna(health_val) else 'N/A'
        
        indicator_map = {
            'high_absences': ('High Absences', f"{student_row.get('absences', 'N/A')} absences"),
            'low_studytime': ('Low Study Time', studytime_desc),
            'alcohol_issues': ('Alcohol Issues', f"Workday: {dalc_desc} / Weekend: {walc_desc}"),
            'past_failures': ('Past Failures', f"{student_row.get('failures', 'N/A')} failure(s)"),
            'no_school_support': ('No School Support', student_row.get('schoolsup', 'N/A').upper() if pd.notna(student_row.get('schoolsup')) else 'N/A'),
            'no_family_support': ('No Family Support', student_row.get('famsup', 'N/A').upper() if pd.notna(student_row.get('famsup')) else 'N/A'),
            'poor_family_relations': ('Poor Family Relations', famrel_desc),
            'health_issues': ('Health Issues', health_desc)
        }
        
        # Display indicators in a table format
        for col, (label, value) in indicator_map.items():
            status = student_row.get(col, 0)
            if status == 1:
                st.markdown(
                    f"🔴 **{label}**: {value}",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"🟢 **{label}**: {value}",
                    unsafe_allow_html=True
                )
    
    with col_right:
        st.markdown("#### 💡 Suggested Interventions")
        interventions = get_suggested_interventions(student_row)
        for intervention in interventions:
            st.markdown(f"- {intervention}")
    
    # Radar chart section
    st.markdown("---")
    st.markdown("#### 📊 Indicator Overview - Radar Chart")
    
    # Prepare data for radar chart
    indicator_labels = [
        'High Absences',
        'Low Study Time',
        'Alcohol Issues',
        'Past Failures',
        'No School Support',
        'No Family Support',
        'Poor Family Relations',
        'Health Issues'
    ]
    
    indicator_values = [
        student_row.get('high_absences', 0),
        student_row.get('low_studytime', 0),
        student_row.get('alcohol_issues', 0),
        student_row.get('past_failures', 0),
        student_row.get('no_school_support', 0),
        student_row.get('no_family_support', 0),
        student_row.get('poor_family_relations', 0),
        student_row.get('health_issues', 0)
    ]
    
    # Create radar chart
    fig_radar = go.Figure()
    
    fig_radar.add_trace(go.Scatterpolar(
        r=indicator_values,
        theta=indicator_labels,
        fill='toself',
        fillcolor='rgba(211, 47, 47, 0.3)',
        line=dict(color='#d32f2f', width=2),
        name='Risk Indicators'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0, 1],
                ticktext=['No Risk', 'At Risk'],
                showticklabels=True
            )
        ),
        showlegend=False,
        height=500,
        margin=dict(l=80, r=80, t=40, b=40)
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Demographics section (collapsible)
    with st.expander("👥 Demographics & Background"):
        demo_col1, demo_col2, demo_col3 = st.columns(3)
        
        with demo_col1:
            st.markdown("**Basic Information**")
            st.write(f"Age: {student_row.get('age', 'N/A')} years")
            sex_val = student_row.get('sex', 'N/A')
            sex_display = 'Female' if sex_val == 'F' else 'Male' if sex_val == 'M' else sex_val
            st.write(f"Sex: {sex_display}")
            address_val = student_row.get('address')
            st.write(f"Address: {decode_address(address_val) if pd.notna(address_val) else 'N/A'}")
            traveltime_val = student_row.get('traveltime')
            st.write(f"Travel Time: {decode_traveltime(traveltime_val) if pd.notna(traveltime_val) else 'N/A'}")
            st.write(f"Guardian: {student_row.get('guardian', 'N/A').capitalize() if pd.notna(student_row.get('guardian')) else 'N/A'}")
        
        with demo_col2:
            st.markdown("**Family Background**")
            medu_val = student_row.get('Medu')
            st.write(f"Mother's Education: {decode_education(medu_val) if pd.notna(medu_val) else 'N/A'}")
            fedu_val = student_row.get('Fedu')
            st.write(f"Father's Education: {decode_education(fedu_val) if pd.notna(fedu_val) else 'N/A'}")
            st.write(f"Mother's Job: {student_row.get('Mjob', 'N/A').replace('_', ' ').title() if pd.notna(student_row.get('Mjob')) else 'N/A'}")
            st.write(f"Father's Job: {student_row.get('Fjob', 'N/A').replace('_', ' ').title() if pd.notna(student_row.get('Fjob')) else 'N/A'}")
            famsize_val = student_row.get('famsize')
            st.write(f"Family Size: {decode_famsize(famsize_val) if pd.notna(famsize_val) else 'N/A'}")
            pstatus_val = student_row.get('Pstatus')
            st.write(f"Parent Status: {decode_pstatus(pstatus_val) if pd.notna(pstatus_val) else 'N/A'}")
        
        with demo_col3:
            st.markdown("**Additional Information**")
            st.write(f"Extra Paid Classes: {student_row.get('paid', 'N/A').upper() if pd.notna(student_row.get('paid')) else 'N/A'}")
            st.write(f"Extra Activities: {student_row.get('activities', 'N/A').upper() if pd.notna(student_row.get('activities')) else 'N/A'}")
            st.write(f"Internet Access: {student_row.get('internet', 'N/A').upper() if pd.notna(student_row.get('internet')) else 'N/A'}")
            st.write(f"Nursery School: {student_row.get('nursery', 'N/A').upper() if pd.notna(student_row.get('nursery')) else 'N/A'}")
            st.write(f"Higher Education Goal: {student_row.get('higher', 'N/A').upper() if pd.notna(student_row.get('higher')) else 'N/A'}")
            st.write(f"Romantic Relationship: {student_row.get('romantic', 'N/A').upper() if pd.notna(student_row.get('romantic')) else 'N/A'}")
            reason_val = student_row.get('reason')
            st.write(f"School Choice Reason: {reason_val.capitalize() if pd.notna(reason_val) else 'N/A'}")
        
        st.markdown("---")
        st.markdown("**Lifestyle & Social**")
        lifestyle_col1, lifestyle_col2, lifestyle_col3 = st.columns(3)
        
        with lifestyle_col1:
            freetime_val = student_row.get('freetime')
            st.write(f"Free Time: {decode_rating(freetime_val, 'frequency') if pd.notna(freetime_val) else 'N/A'}")
        
        with lifestyle_col2:
            goout_val = student_row.get('goout')
            st.write(f"Going Out: {decode_rating(goout_val, 'frequency') if pd.notna(goout_val) else 'N/A'}")
        
        with lifestyle_col3:
            famrel_val = student_row.get('famrel')
            st.write(f"Family Relations: {decode_rating(famrel_val, 'quality') if pd.notna(famrel_val) else 'N/A'}")


def get_ollama_models():
    """Get list of available Ollama models."""
    try:
        # If running in Docker, use HTTP API
        if os.getenv('DOCKER_ENV') == 'true':
            import requests
            ollama_url = get_ollama_base_url()
            response = requests.get(f"{ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                return models
            return []
        else:
            # Use CLI for local environment
            import subprocess
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                models = []
                for line in lines:
                    if line.strip():
                        model_name = line.split()[0]
                        models.append(model_name)
                return models
            return []
    except Exception as e:
        logger.warning(f"Failed to get Ollama models: {e}")
        return []


def pull_ollama_model(model_name: str) -> bool:
    """Pull an Ollama model."""
    try:
        # If running in Docker, use HTTP API
        if os.getenv('DOCKER_ENV') == 'true':
            import requests
            ollama_url = get_ollama_base_url()
            response = requests.post(
                f"{ollama_url}/api/pull",
                json={"name": model_name},
                timeout=300
            )
            return response.status_code == 200
        else:
            # Use CLI for local environment
            import subprocess
            result = subprocess.run(
                ['ollama', 'pull', model_name],
                capture_output=True,
                text=True,
                timeout=300
            )
            return result.returncode == 0
    except Exception as e:
        logger.error(f"Failed to pull Ollama model: {e}")
        return False


def save_to_env_file(provider: str, model_name: str, api_key: str = None):
    """Save LLM configuration to .env file."""
    env_path = '.env'
    env_vars = {}
    
    # Read existing .env file
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    
    # Update with new values
    if provider.lower() == 'ollama':
        env_vars['OLLAMA_MODEL_NAME'] = f'ollama/{model_name}'
        # Remove API key if exists
        env_vars.pop('MISTRAL_API_KEY', None)
        env_vars.pop('OPENAI_API_KEY', None)
    else:
        # For other providers (mistral, openai, etc.)
        env_vars[f'{provider.upper()}_MODEL_NAME'] = f'{provider.lower()}/{model_name}'
        if api_key:
            env_vars[f'{provider.upper()}_API_KEY'] = api_key
    
    # Write back to .env file
    with open(env_path, 'w') as f:
        for key, value in env_vars.items():
            f.write(f'{key}={value}\n')
    
    logger.info(f"Saved LLM configuration to .env: {provider}/{model_name}")
    return True


def get_current_llm_config():
    """Get current LLM configuration from .env file."""
    config = {
        'provider': None,
        'model_name': None,
        'api_key_set': False
    }
    
    # Check for different providers
    providers_to_check = ['MISTRAL', 'OPENAI', 'OLLAMA', 'ANTHROPIC', 'GROQ']
    
    for provider in providers_to_check:
        model_key = f'{provider}_MODEL_NAME'
        api_key = f'{provider}_API_KEY'
        
        model_value = os.getenv(model_key)
        if model_value:
            config['provider'] = provider.lower()
            config['model_name'] = model_value
            config['api_key_set'] = bool(os.getenv(api_key))
            break
    
    return config


def display_data_selection_page():
    """Display the data file selection and upload page."""
    logger.info("Displaying data selection page")
    st.title("🎓 Student Intervention Prioritization Dashboard")
    st.markdown(
        "**Portuguese Ministry of Education** | "
        "Identify students requiring personalized support after school closures"
    )
    
    st.markdown("---")
    
    # LLM Configuration Section
    st.header("🤖 LLM Configuration (Optional)")
    st.markdown(
        "Configure your AI provider for the intelligent data query feature. "
        "You can skip this and proceed to load data - the AI feature will be disabled."
    )
    
    # Show current configuration
    current_config = get_current_llm_config()
    
    if current_config['provider']:
        st.success(
            f"✅ **Current Configuration:** {current_config['provider'].upper()} - "
            f"{current_config['model_name']} "
            f"{'(API Key Set)' if current_config['api_key_set'] else '(No API Key)'}"
        )
    else:
        st.info("ℹ️ No LLM provider configured. AI query feature will be unavailable.")
    
    with st.expander("⚙️ Configure LLM Provider", expanded=not current_config['provider']):
        col_provider, col_model = st.columns(2)
        
        with col_provider:
            provider_options = ['Ollama', 'Mistral', 'OpenAI', 'Anthropic', 'Groq']
            selected_provider = st.selectbox(
                "Select Provider",
                options=provider_options,
                help="Choose your LLM provider. Ollama runs locally, others require API keys."
            )
        
        with col_model:
            if selected_provider == 'Ollama':
                # Get available Ollama models
                ollama_models = get_ollama_models()
                
                if ollama_models:
                    selected_model = st.selectbox(
                        "Select Model",
                        options=ollama_models,
                        help="Choose from your locally available Ollama models"
                    )
                else:
                    st.warning("⚠️ No Ollama models found. Please install Ollama and pull a model.")
                    selected_model = st.text_input(
                        "Model Name",
                        placeholder="e.g., llama2, mistral, codellama",
                        help="Enter the name of the model you want to pull"
                    )
                    
                    if selected_model and st.button("📥 Pull Model from Ollama"):
                        with st.spinner(f"Pulling {selected_model} from Ollama..."):
                            if pull_ollama_model(selected_model):
                                st.success(f"✅ Successfully pulled {selected_model}")
                                st.rerun()
                            else:
                                st.error(f"❌ Failed to pull {selected_model}. Make sure Ollama is installed and running.")
            else:
                # For API-based providers
                if selected_provider == 'Mistral':
                    default_model = 'codestral-2508'
                    model_examples = "e.g., codestral-2508, mistral-large-latest"
                elif selected_provider == 'OpenAI':
                    default_model = 'gpt-4'
                    model_examples = "e.g., gpt-4, gpt-3.5-turbo"
                elif selected_provider == 'Anthropic':
                    default_model = 'claude-3-sonnet'
                    model_examples = "e.g., claude-3-opus, claude-3-sonnet"
                elif selected_provider == 'Groq':
                    default_model = 'mixtral-8x7b'
                    model_examples = "e.g., mixtral-8x7b, llama2-70b"
                else:
                    default_model = ''
                    model_examples = "Enter model name"
                
                selected_model = st.text_input(
                    "Model Name",
                    value=default_model,
                    placeholder=model_examples,
                    help=f"Enter the {selected_provider} model name"
                )
        
        # API Key input (not needed for Ollama)
        api_key = None
        if selected_provider != 'Ollama':
            api_key = st.text_input(
                f"{selected_provider} API Key",
                type="password",
                placeholder=f"Enter your {selected_provider} API key",
                help=f"Your API key will be saved to the .env file"
            )
        
        # Save configuration button
        col_save, col_skip = st.columns(2)
        
        with col_save:
            if st.button("💾 Save LLM Configuration", type="primary"):
                if selected_provider and selected_model:
                    if selected_provider != 'Ollama' and not api_key:
                        st.error("❌ API key is required for non-Ollama providers")
                    else:
                        try:
                            save_to_env_file(selected_provider, selected_model, api_key)
                            st.success(f"✅ Configuration saved! Reload environment variables to use.")
                            # Reload environment variables
                            load_dotenv(override=True)
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error saving configuration: {e}")
                else:
                    st.error("❌ Please select both provider and model")
        
        with col_skip:
            if st.button("⏭️ Skip Configuration"):
                st.info("You can configure LLM later. AI query feature will be disabled.")
    
    st.markdown("---")
    st.header("📁 Data File Selection")
    
    # Check for existing files in data folder
    data_folder = 'data'
    existing_files = []
    
    if os.path.exists(data_folder):
        existing_files = [f for f in os.listdir(data_folder) 
                         if f.endswith('.csv') and not f.startswith('.')]
    
    # Display existing files
    st.subheader("📋 Existing Files in Data Folder")
    
    if existing_files:
        st.write(f"Found {len(existing_files)} CSV file(s):")
        
        # Create a selection box for existing files
        selected_file = st.selectbox(
            "Select a file to load:",
            options=['-- Choose a file --'] + existing_files,
            key='existing_file_select'
        )
        
        if selected_file != '-- Choose a file --':
            file_path = os.path.join(data_folder, selected_file)
            file_size = os.path.getsize(file_path)
            st.info(f"📄 **{selected_file}** ({file_size / 1024:.2f} KB)")
            
            if st.button("✅ Load Selected File", type="primary"):
                st.session_state.data_file = file_path
                st.session_state.data_source = 'existing'
                st.rerun()
    else:
        st.info("No CSV files found in the data folder.")
    
    st.markdown("---")
    
    # File upload section
    st.subheader("📤 Upload New Data File")
    st.markdown(
        "Upload a CSV file containing student data. The file should include the following columns: "
        "`StudentID`, `FinalGrade`, `absences`, `studytime`, `Dalc`, `Walc`, `failures`, "
        "`schoolsup`, `famsup`, `famrel`, `health`"
    )
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['xlsx', 'xls', 'csv'],
        help="Upload a CSV file with student data"
    )
    
    if uploaded_file is not None:
        # Show file details
        st.success(f"✅ File uploaded: **{uploaded_file.name}** ({uploaded_file.size / 1024:.2f} KB)")
        
        # Option to save to data folder
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save to Data Folder & Load", type="primary"):
                try:
                    # Save uploaded file to data folder
                    save_path = os.path.join(data_folder, uploaded_file.name)
                    with open(save_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    
                    st.session_state.data_file = save_path
                    st.session_state.data_source = 'uploaded'
                    st.success(f"File saved to: {save_path}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error saving file: {str(e)}")
        
        with col2:
            if st.button("📊 Load Temporarily (Don't Save)"):
                # Load directly from uploaded file without saving
                st.session_state.uploaded_data = uploaded_file
                st.session_state.data_source = 'temporary'
                st.rerun()
    
    # Help section
    with st.expander("ℹ️ Required Data Format"):
        st.markdown("""
        ### Core Columns (Required)
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
        
        ### Optional Demographic Columns
        - `age`, `sex`, `address`, `Medu`, `Fedu`, `famsize`, `Pstatus`, 
          `paid`, `activities`, `internet`
        
        ### Example Format
        ```
        StudentID,FinalGrade,absences,studytime,Dalc,Walc,failures,schoolsup,famsup,famrel,health
        STU001,14,2,2,1,1,0,no,yes,4,5
        STU002,8,15,1,3,4,1,yes,no,2,3
        ```
        """)


def main():
    """Main application function."""
    logger.info("Main application started")
    
    # Initialize session state
    if 'data_file' not in st.session_state:
        st.session_state.data_file = None
    if 'data_source' not in st.session_state:
        st.session_state.data_source = None
    if 'selected_student' not in st.session_state:
        st.session_state.selected_student = None
    
    # Show data selection page if no data is loaded
    if st.session_state.data_file is None and st.session_state.data_source != 'temporary':
        logger.debug("No data loaded, showing data selection page")
        display_data_selection_page()
        return
    
    # Load data based on source
    try:
        if st.session_state.data_source == 'temporary':
            with st.spinner('Loading student data...'):
                df = pd.read_csv(st.session_state.uploaded_data)
                df = process_student_data(df)
        else:
            with st.spinner('Loading student data...'):
                df = load_data(st.session_state.data_file)
        
        # Display header with data source info
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title("🎓 Student Intervention Prioritization Dashboard")
            st.markdown(
                "**Portuguese Ministry of Education** | "
                "Identify students requiring personalized support after school closures"
            )
        with col2:
            st.markdown("###")  # Spacing
            if st.button("🔄 Change Data File"):
                # Clear data and return to selection page
                st.session_state.data_file = None
                st.session_state.data_source = None
                if 'uploaded_data' in st.session_state:
                    del st.session_state.uploaded_data
                st.rerun()
        
        st.success(f"✅ Loaded data for {len(df)} students")
        logger.info(f"Successfully loaded and displayed data for {len(df)} students")
        
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}", exc_info=True)
        st.error(f"❌ Error loading data: {str(e)}")
        if st.button("← Back to File Selection"):
            st.session_state.data_file = None
            st.session_state.data_source = None
            if 'uploaded_data' in st.session_state:
                del st.session_state.uploaded_data
            st.rerun()
        st.stop()
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Grade range filter
    grade_range = st.sidebar.slider(
        "Final Grade Range",
        min_value=0,
        max_value=20,
        value=(0, 20),
        step=1,
        help="Filter students by their final grade (0-20 scale)"
    )
    
    # Intervention score threshold
    intervention_threshold = st.sidebar.slider(
        "Minimum Intervention Score",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Filter students by minimum intervention complexity score"
    )
    
    # Specific indicators multi-select
    indicator_options = [
        'High absences',
        'Low study time',
        'Alcohol consumption',
        'Past failures',
        'No school support',
        'No family support',
        'Poor family relations',
        'Health issues'
    ]
    
    selected_indicators = st.sidebar.multiselect(
        "Specific Indicators",
        options=indicator_options,
        default=[],
        help="Filter to show only students with selected indicators"
    )
    
    # Reset filters button
    if st.sidebar.button("🔄 Reset Filters"):
        st.rerun()
    
    # Clear selection button
    if st.sidebar.button("❌ Clear Student Selection"):
        st.session_state.selected_student = None
        st.rerun()
    
    # Apply filters
    filters = {
        'grade_range': grade_range,
        'intervention_threshold': intervention_threshold,
        'selected_indicators': selected_indicators
    }
    
    logger.debug(f"User applied filters: grade_range={grade_range}, threshold={intervention_threshold}, indicators={len(selected_indicators)}")
    filtered_df = apply_filters(df, filters)
    
    # Display filter summary
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Filter Results")
    st.sidebar.metric(
        "Filtered Students",
        f"{len(filtered_df)} / {len(df)}",
        f"{(len(filtered_df)/len(df)*100):.1f}%"
    )
    
    # Main Section 1: Priority Matrix
    st.markdown("---")
    st.subheader("📍 Student Priority Matrix")
    st.markdown(
        "*Click on any point to view detailed student profile. "
        "Points are sized by number of actionable indicators.*"
    )
    
    fig_matrix = create_priority_matrix(filtered_df)
    
    # Display the chart and capture click events
    selected_point = st.plotly_chart(
        fig_matrix,
        use_container_width=True,
        key="priority_matrix",
        on_select="rerun"
    )
    
    # Handle point selection (Streamlit's plotly_events might need additional setup)
    # For now, we'll use a selectbox as an alternative selection method
    
    # Main Section 2: Actionable Indicators Distribution
    st.markdown("---")
    st.subheader("📊 Actionable Indicators Overview")
    
    col_chart, col_stats = st.columns([2, 1])
    
    with col_chart:
        fig_indicators = create_indicator_distribution_chart(filtered_df)
        st.plotly_chart(fig_indicators, use_container_width=True)
    
    with col_stats:
        st.markdown("#### Summary Statistics")
        
        total_students = len(filtered_df)
        high_priority = len(filtered_df[filtered_df['urgency_zone'] == 'High Priority'])
        moderate_priority = len(filtered_df[filtered_df['urgency_zone'] == 'Moderate Priority'])
        monitor = len(filtered_df[filtered_df['urgency_zone'] == 'Monitor'])
        
        st.metric("Total Students", total_students)
        
        st.markdown(f"**High Priority**")
        st.markdown(f"{high_priority} students ({high_priority/total_students*100:.1f}%)")
        
        st.markdown(f"**Moderate Priority**")
        st.markdown(f"{moderate_priority} students ({moderate_priority/total_students*100:.1f}%)")
        
        st.markdown(f"**Monitor**")
        st.markdown(f"{monitor} students ({monitor/total_students*100:.1f}%)")
        
        st.markdown("---")
        
        st.metric(
            "Avg Intervention Score",
            f"{filtered_df['intervention_score'].mean():.2f}"
        )
        st.metric(
            "Median Final Grade",
            f"{filtered_df['FinalGrade'].median():.1f}"
        )
    
    # Main Section 3: Priority Student List
    st.markdown("---")
    st.subheader("🎯 Top Priority Students")
    
    # Prepare display dataframe
    columns_to_display = ['StudentID', 'FinalGrade', 'priority_score', 'intervention_score',
                          'num_indicators', 'indicators_list', 'urgency_zone']
    
    # Add name and gender if available
    if 'FirstName' in filtered_df.columns and 'FamilyName' in filtered_df.columns:
        filtered_df['FullName'] = filtered_df['FirstName'].fillna('') + ' ' + filtered_df['FamilyName'].fillna('')
        filtered_df['FullName'] = filtered_df['FullName'].str.strip()
        columns_to_display.insert(0, 'FullName')
    
    if 'sex' in filtered_df.columns:
        columns_to_display.insert(1 if 'FullName' in columns_to_display else 0, 'sex')
    
    display_df = filtered_df.nlargest(30, 'priority_score')[columns_to_display].copy()
    
    # Rename columns for display
    column_names = [col for col in display_df.columns]
    display_names = []
    for col in column_names:
        if col == 'FullName':
            display_names.append('Student Name')
        elif col == 'sex':
            display_names.append('Gender')
        elif col == 'StudentID':
            display_names.append('Student ID')
        elif col == 'FinalGrade':
            display_names.append('Final Grade')
        elif col == 'priority_score':
            display_names.append('Priority Score')
        elif col == 'intervention_score':
            display_names.append('Intervention Score')
        elif col == 'num_indicators':
            display_names.append('Num Indicators')
        elif col == 'indicators_list':
            display_names.append('Actionable Indicators')
        elif col == 'urgency_zone':
            display_names.append('Urgency Zone')
        else:
            display_names.append(col)
    
    display_df.columns = display_names
    
    # Apply styling based on urgency zone
    def highlight_urgency(row):
        if row['Urgency Zone'] == 'High Priority':
            return ['background-color: #ffcdd2; color: black'] * len(row)
        elif row['Urgency Zone'] == 'Moderate Priority':
            return ['background-color: #ffe0b2; color: black'] * len(row)
        else:
            return ['background-color: #c8e6c9; color: black'] * len(row)
    
    styled_df = display_df.style.apply(highlight_urgency, axis=1)
    
    # Display dataframe
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        height=400
    )
    
    # Download button
    csv = display_df.to_csv(index=False)
    timestamp = datetime.now().strftime('%Y%m%d')
    st.download_button(
        label="📥 Download Priority List (CSV)",
        data=csv,
        file_name=f"priority_students_{timestamp}.csv",
        mime="text/csv"
    )
    
    # Student selection dropdown (alternative to clicking)
    st.markdown("---")
    selected_student_id = st.selectbox(
        "🔍 Select a student to view detailed profile:",
        options=['None'] + sorted(filtered_df['StudentID'].tolist()),
        index=0
    )
    
    if selected_student_id != 'None':
        st.session_state.selected_student = selected_student_id
        logger.info(f"User selected student: {selected_student_id}")
    
    # Main Section 4: Individual Student Profile
    if st.session_state.selected_student is not None:
        try:
            student_row = filtered_df[
                filtered_df['StudentID'] == st.session_state.selected_student
            ].iloc[0]
            display_student_profile(student_row, filtered_df)
        except IndexError:
            logger.warning(f"Selected student {st.session_state.selected_student} not found in filtered data")
            st.warning(
                f"⚠️ Student {st.session_state.selected_student} not found in filtered results. "
                "Try adjusting filters or clearing selection."
            )
    
    # Main Section 5: AI-Powered Data Query
    st.markdown("---")
    st.subheader("🤖 Ask Questions About the Data")
    
    if not SMOLAGENTS_AVAILABLE:
        st.warning("⚠️ AI Query feature requires smolagents. Install with: `pip install smolagents`")
    else:
        st.markdown(
            "Ask natural language questions about the student data. "
            "The AI agent will analyze the dataset and provide answers."
        )
        
        # Initialize chat history in session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Input for new question
        st.markdown("#### Ask a New Question")
        
        # Example questions
        # st.markdown("**Example questions:**")
        example_col1, example_col2 = st.columns(2)
        
        example_question = None
        with example_col1:
            if st.button("📊 What's the average grade by urgency zone?"):
                example_question = "What's the average grade by urgency zone?"
            if st.button("👥 How many male vs female students in high priority?"):
                example_question = "How many male vs female students in high priority?"
        
        with example_col2:
            if st.button("🎯 Which indicators are most common in failing students?"):
                example_question = "Which indicators are most common in failing students?"
            if st.button("📈 Show correlation between absences and final grade"):
                example_question = "Show correlation between absences and final grade"
        
        col_input, col_button = st.columns([4, 1])
        
        with col_input:
            user_question = st.text_input(
                "Your question:",
                placeholder="e.g., How many students have high absences and low grades?",
                key="user_question_input",
                label_visibility="collapsed"
            )
        
        with col_button:
            ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
        
        # Use example question if button was clicked, otherwise use user input
        final_question = example_question if example_question else user_question
        
        # Process question
        if (ask_button and user_question) or example_question:
            with st.spinner("🤔 Thinking..."):
                try:
                    # Get current LLM configuration
                    llm_config = get_current_llm_config()
                    
                    if not llm_config['provider']:
                        st.error("❌ No LLM provider configured. Please configure in the data selection page.")
                        st.stop()
                    
                    # Get API key and model name from environment
                    provider = llm_config['provider'].upper()
                    model_name = llm_config['model_name']
                    api_key = os.getenv(f'{provider}_API_KEY')
                    
                    # For Ollama, API key is not required
                    if provider != 'OLLAMA' and not api_key:
                        st.error(f"❌ {provider} API key not found. Please configure in the data selection page.")
                        st.stop()
                    
                    # Initialize the LiteLLM model
                    if provider == 'OLLAMA':
                        ollama_url = get_ollama_base_url()
                        logger.info(f"Using Ollama endpoint: {ollama_url}")
                        model = LiteLLMModel(
                            model_id=model_name,
                            api_base=ollama_url
                        )
                    else:
                        model = LiteLLMModel(
                            model_id=model_name,
                            api_key=api_key
                        )
                    
                    # Create the CodeAgent with the model
                    agent = CodeAgent(
                        tools=[load_data],
                        model=model,
                        additional_authorized_imports=['*'],
                        max_steps=5,
                        name='Data_Analysis_Agent',
                        description='Helpful assistant for student intervention data'
                    )
                    
                    # Prepare the data context
                    data_info = f"""
                    You are a helpful data analysis assistant.
                    You have access to pandas DataFrames with the following columns: {', '.join(df.columns.tolist())}
                    
                    Key information:
                    - FinalGrade: 0-20 scale (passing is >= 10)
                    - intervention_score: 0-1 scale (higher = more actionable)
                    - urgency_zone: 'High Priority', 'Moderate Priority', or 'Monitor'
                    - Actionable indicators: high_absences, low_studytime, alcohol_issues, past_failures, 
                      no_school_support, no_family_support, poor_family_relations, health_issues
                    
                    the path to the data file is: {st.session_state.data_file}
                    if asked about the data, use the tool 'load_data' to load the data in order to answer the question.
                    
                    Question: {final_question}
                    
                    Provide a clear, concise and complete coherent answer based on the data analysis.
                    """
                    
                    # Run the agent
                    result = agent.run(data_info)
                    
                    # Store in chat history
                    st.session_state.chat_history.append((final_question, str(result)))
                    
                    # Display the answer
                    # st.success("✅ Analysis complete!")
                    st.markdown("### Answer:")
                    st.markdown(result)
                    
                    # Display previous chat history below current answer
                    if len(st.session_state.chat_history) > 1:
                        st.markdown("---")
                        st.markdown("#### Previous Questions & Answers")
                        for i, (prev_question, prev_answer) in enumerate(st.session_state.chat_history[:-1]):
                            with st.expander(f"Q{i+1}: {prev_question}", expanded=False):
                                # st.markdown(f"**Question:** {prev_question}")
                                st.markdown(f"**Answer:** {prev_answer}")
                    
                except Exception as e:
                    error_msg = f"Error processing question: {str(e)}"
                    st.error(error_msg)
                    # st.info(
                    #     "💡 **Tip:** Try rephrasing your question or make it more specific. "
                    #     "The AI works best with clear, data-focused questions."
                    # )
        
        # Clear history button
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>" 
        "Student Intervention Dashboard v1.0 | Portuguese Ministry of Education"
        "</div>",
        unsafe_allow_html=True
    )
if __name__ == "__main__":
    main()
