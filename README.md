# AccuKnox AI/ML Assignment

**Candidate:** Divyesh Mutha
**Date:** January 2026

---

## Project Structure

```
Assignment2/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── sample_users.csv             # Sample data for Task 3
├── task1_api_books.py           # Task 1: API Data Retrieval
├── task2_student_scores.py      # Task 2: Data Visualization
├── task3_csv_import.py          # Task 3: CSV to Database
└── Assignment2_Responses.md     # Written responses for Problem Statement 2
```

---

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Navigate to the project directory:
   ```bash
   cd Assignment2
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Scripts

### Task 1: API Data Retrieval and Storage (Books)

Fetches book data from Open Library API, stores in SQLite database, and displays results.

```bash
python task1_api_books.py
```

**Output:**
- Creates `books.db` SQLite database
- Displays fetched books in formatted table

---

### Task 2: Data Processing and Visualization (Student Scores)

Fetches student data, calculates statistics, and creates bar chart visualizations.

```bash
python task2_student_scores.py
```

**Output:**
- Creates `student_scores_chart.png` - Bar chart of student averages
- Creates `subject_averages_chart.png` - Subject comparison chart
- Displays statistics in console

---

### Task 3: CSV Data Import to Database

Reads user data from CSV file and imports into SQLite database with validation.

```bash
python task3_csv_import.py
```

**Output:**
- Creates `users.db` SQLite database
- Displays import summary and database contents

---

## Portfolio Links (Problem Statement 1 - Additional Requirements)

**Most Complex Python Code:**
- **Project:** AI Interview Question Generator
- **Link:** https://github.com/divyeshmutha12/AI-Interview-Question-Generator
- **Description:** An intelligent RAG-based system that generates personalized interview questions by analyzing candidate resumes and job profiles using LangGraph, FAISS vector database, and OpenAI API.

**Most Complex Database Code:**
- **Project:** AI Interview Question Generator (FAISS Vector Database Implementation)
- **Link:** https://github.com/divyeshmutha12/AI-Interview-Question-Generator
- **Description:** Implements FAISS vector store for semantic search and knowledge base retrieval, demonstrating advanced database concepts including vector embeddings and similarity search.

---

## Problem Statement 2 Responses

See `Assignment2_Responses.md` for detailed written responses covering:
1. Self-rating on LLM, Deep Learning, AI, ML technologies
2. LLM-based chatbot architecture explanation
3. Vector databases analysis and selection

---

## Assumptions Made

1. **Task 1:** Using Open Library API (free, no authentication required)
2. **Task 2:** Using JSONPlaceholder API with transformed data for demonstration
3. **Task 3:** CSV file has headers: name, email, phone, age, city
4. All databases are created locally in the project directory

---

## Contact

For any questions regarding this assignment, please contact me.
