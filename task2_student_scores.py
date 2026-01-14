"""
Task 2: Data Processing and Visualization
==========================================
This script fetches student test score data from an API, calculates statistics
(including average score), and creates a bar chart visualization.

Assumptions:
1. Using JSONPlaceholder or a mock API endpoint for demonstration
2. Since no specific student score API is provided, we'll use a simulated API
   with realistic student data
3. Visualization will be saved as PNG and also displayed if GUI is available

Author: Divyesh Mutha
Date: January 2026
"""

import requests
import json
import statistics
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple


# =============================================================================
# API DATA FETCHING FUNCTIONS
# =============================================================================

def fetch_student_scores_from_api(api_url=None):
    """
    Fetch student test scores from an external API.

    Args:
        api_url (str): URL of the API endpoint (optional)

    Returns:
        list: List of dictionaries containing student score data
    """
    # Attempt to fetch from a mock API service
    # Using JSONPlaceholder's posts endpoint and transforming data
    if api_url is None:
        api_url = "https://jsonplaceholder.typicode.com/users"

    try:
        print(f"Fetching data from API: {api_url}")
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()

        # Transform API data into student scores format
        api_data = response.json()
        students = []

        # Generate realistic scores based on API data
        np.random.seed(42)  # For reproducibility

        for i, user in enumerate(api_data[:10]):  # Limit to 10 students
            student = {
                "id": user.get("id", i + 1),
                "name": user.get("name", f"Student {i + 1}"),
                "math_score": int(np.random.normal(75, 15)),
                "science_score": int(np.random.normal(72, 12)),
                "english_score": int(np.random.normal(78, 10)),
                "history_score": int(np.random.normal(70, 14))
            }
            # Clamp scores between 0 and 100
            for key in ["math_score", "science_score", "english_score", "history_score"]:
                student[key] = max(0, min(100, student[key]))

            students.append(student)

        print(f"Successfully fetched and transformed {len(students)} student records")
        return students

    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        print("Using fallback sample data...")
        return get_sample_student_data()


def get_sample_student_data():
    """
    Return sample student score data as fallback.

    Returns:
        list: List of student score dictionaries
    """
    return [
        {"id": 1, "name": "Alice Johnson", "math_score": 85, "science_score": 78, "english_score": 92, "history_score": 88},
        {"id": 2, "name": "Bob Smith", "math_score": 72, "science_score": 85, "english_score": 68, "history_score": 75},
        {"id": 3, "name": "Charlie Brown", "math_score": 90, "science_score": 88, "english_score": 85, "history_score": 92},
        {"id": 4, "name": "Diana Ross", "math_score": 65, "science_score": 70, "english_score": 75, "history_score": 68},
        {"id": 5, "name": "Edward Wilson", "math_score": 78, "science_score": 82, "english_score": 80, "history_score": 85},
        {"id": 6, "name": "Fiona Garcia", "math_score": 95, "science_score": 92, "english_score": 88, "history_score": 90},
        {"id": 7, "name": "George Martinez", "math_score": 58, "science_score": 62, "english_score": 70, "history_score": 55},
        {"id": 8, "name": "Hannah Lee", "math_score": 82, "science_score": 79, "english_score": 85, "history_score": 80},
        {"id": 9, "name": "Ivan Petrov", "math_score": 88, "science_score": 90, "english_score": 72, "history_score": 78},
        {"id": 10, "name": "Julia Chen", "math_score": 76, "science_score": 74, "english_score": 82, "history_score": 79},
    ]


# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================

def calculate_student_average(student: Dict) -> float:
    """
    Calculate the average score for a single student across all subjects.

    Args:
        student (dict): Student record with scores

    Returns:
        float: Average score rounded to 2 decimal places
    """
    scores = [
        student.get("math_score", 0),
        student.get("science_score", 0),
        student.get("english_score", 0),
        student.get("history_score", 0)
    ]
    return round(statistics.mean(scores), 2)


def calculate_subject_averages(students: List[Dict]) -> Dict[str, float]:
    """
    Calculate average scores for each subject across all students.

    Args:
        students (list): List of student dictionaries

    Returns:
        dict: Dictionary with subject names and their averages
    """
    subjects = {
        "Math": [s.get("math_score", 0) for s in students],
        "Science": [s.get("science_score", 0) for s in students],
        "English": [s.get("english_score", 0) for s in students],
        "History": [s.get("history_score", 0) for s in students]
    }

    averages = {}
    for subject, scores in subjects.items():
        averages[subject] = round(statistics.mean(scores), 2)

    return averages


def calculate_overall_statistics(students: List[Dict]) -> Dict:
    """
    Calculate comprehensive statistics for all students.

    Args:
        students (list): List of student dictionaries

    Returns:
        dict: Dictionary containing various statistics
    """
    all_scores = []
    for student in students:
        all_scores.extend([
            student.get("math_score", 0),
            student.get("science_score", 0),
            student.get("english_score", 0),
            student.get("history_score", 0)
        ])

    student_averages = [calculate_student_average(s) for s in students]

    return {
        "overall_average": round(statistics.mean(all_scores), 2),
        "overall_median": round(statistics.median(all_scores), 2),
        "overall_std_dev": round(statistics.stdev(all_scores), 2),
        "highest_score": max(all_scores),
        "lowest_score": min(all_scores),
        "top_student_avg": max(student_averages),
        "bottom_student_avg": min(student_averages),
        "class_average": round(statistics.mean(student_averages), 2)
    }


def process_student_data(students: List[Dict]) -> Tuple[List[Dict], Dict, Dict]:
    """
    Process student data and compute all relevant statistics.

    Args:
        students (list): Raw student data from API

    Returns:
        tuple: (processed_students, subject_averages, overall_stats)
    """
    # Add average score to each student
    processed_students = []
    for student in students:
        processed = student.copy()
        processed["average_score"] = calculate_student_average(student)
        processed_students.append(processed)

    # Sort by average score (descending)
    processed_students.sort(key=lambda x: x["average_score"], reverse=True)

    # Calculate subject and overall statistics
    subject_averages = calculate_subject_averages(students)
    overall_stats = calculate_overall_statistics(students)

    return processed_students, subject_averages, overall_stats


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_student_scores_bar_chart(students: List[Dict], save_path="student_scores_chart.png"):
    """
    Create a bar chart visualizing student average scores.

    Args:
        students (list): Processed student data with average scores
        save_path (str): Path to save the chart image
    """
    # Extract data for plotting
    names = [s["name"].split()[0] for s in students]  # Use first names for clarity
    averages = [s["average_score"] for s in students]

    # Create figure with larger size
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create color gradient based on scores
    colors = plt.cm.RdYlGn([score / 100 for score in averages])

    # Create bar chart
    bars = ax.bar(names, averages, color=colors, edgecolor='black', linewidth=0.7)

    # Add value labels on bars
    for bar, avg in zip(bars, averages):
        height = bar.get_height()
        ax.annotate(f'{avg:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

    # Add average line
    class_avg = statistics.mean(averages)
    ax.axhline(y=class_avg, color='red', linestyle='--', linewidth=2, label=f'Class Average: {class_avg:.1f}')

    # Customize chart
    ax.set_xlabel('Students', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Score', fontsize=12, fontweight='bold')
    ax.set_title('Student Test Score Analysis\nAverage Scores by Student', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)  # Leave room for labels
    ax.legend(loc='upper right')

    # Add grid for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Tight layout
    plt.tight_layout()

    # Save the chart
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Chart saved to: {save_path}")

    # Show the chart (if display is available)
    try:
        plt.show()
    except Exception:
        print("Display not available, chart saved to file only")

    plt.close()


def create_subject_comparison_chart(subject_averages: Dict, save_path="subject_averages_chart.png"):
    """
    Create a bar chart comparing average scores across subjects.

    Args:
        subject_averages (dict): Dictionary with subject names and averages
        save_path (str): Path to save the chart image
    """
    subjects = list(subject_averages.keys())
    averages = list(subject_averages.values())

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bars with different colors for each subject
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    bars = ax.bar(subjects, averages, color=colors, edgecolor='black', linewidth=0.7)

    # Add value labels
    for bar, avg in zip(bars, averages):
        height = bar.get_height()
        ax.annotate(f'{avg:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold')

    # Customize chart
    ax.set_xlabel('Subjects', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Score', fontsize=12, fontweight='bold')
    ax.set_title('Subject-wise Average Score Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)

    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Chart saved to: {save_path}")

    try:
        plt.show()
    except Exception:
        pass

    plt.close()


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def display_student_data(students: List[Dict]):
    """
    Display student data in a formatted table.

    Args:
        students (list): Processed student data
    """
    print("\n" + "=" * 95)
    print(f"{'Rank':<5} {'Name':<20} {'Math':<8} {'Science':<10} {'English':<10} {'History':<10} {'Average':<8}")
    print("=" * 95)

    for rank, student in enumerate(students, 1):
        print(f"{rank:<5} {student['name']:<20} {student['math_score']:<8} "
              f"{student['science_score']:<10} {student['english_score']:<10} "
              f"{student['history_score']:<10} {student['average_score']:<8.2f}")

    print("=" * 95)


def display_statistics(subject_averages: Dict, overall_stats: Dict):
    """
    Display calculated statistics.

    Args:
        subject_averages (dict): Subject-wise averages
        overall_stats (dict): Overall statistics
    """
    print("\n" + "-" * 50)
    print("SUBJECT-WISE AVERAGES")
    print("-" * 50)
    for subject, avg in subject_averages.items():
        print(f"  {subject:<15}: {avg:.2f}")

    print("\n" + "-" * 50)
    print("OVERALL STATISTICS")
    print("-" * 50)
    print(f"  Overall Average Score  : {overall_stats['overall_average']:.2f}")
    print(f"  Overall Median Score   : {overall_stats['overall_median']:.2f}")
    print(f"  Standard Deviation     : {overall_stats['overall_std_dev']:.2f}")
    print(f"  Highest Score          : {overall_stats['highest_score']}")
    print(f"  Lowest Score           : {overall_stats['lowest_score']}")
    print(f"  Top Student Average    : {overall_stats['top_student_avg']:.2f}")
    print(f"  Bottom Student Average : {overall_stats['bottom_student_avg']:.2f}")
    print(f"  Class Average          : {overall_stats['class_average']:.2f}")
    print("-" * 50)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to orchestrate data fetching, processing, and visualization.
    """
    print("\n" + "=" * 60)
    print("  TASK 2: Data Processing and Visualization")
    print("  Student Test Score Analysis")
    print("=" * 60 + "\n")

    # Step 1: Fetch student data from API
    print("STEP 1: Fetching student score data from API...")
    raw_students = fetch_student_scores_from_api()

    # Step 2: Process data and calculate statistics
    print("\nSTEP 2: Processing data and calculating statistics...")
    processed_students, subject_averages, overall_stats = process_student_data(raw_students)

    # Step 3: Display data in tabular format
    print("\nSTEP 3: Displaying student data...")
    display_student_data(processed_students)
    display_statistics(subject_averages, overall_stats)

    # Step 4: Create visualizations
    print("\nSTEP 4: Creating visualizations...")
    create_student_scores_bar_chart(processed_students, "student_scores_chart.png")
    create_subject_comparison_chart(subject_averages, "subject_averages_chart.png")

    print("\n" + "=" * 60)
    print("  Analysis Complete!")
    print("  Charts have been saved to the current directory.")
    print("=" * 60)


if __name__ == "__main__":
    main()
