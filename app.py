
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
from datetime import date
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# COLLEGE PLACEMENT MANAGEMENT SYSTEM
# Streamlit + SQLite + Pandas + Scikit-learn
# ============================================================

st.set_page_config(
    page_title="College Placement Management System",
    page_icon="🎓",
    layout="wide"
)

DB_NAME = "placement.db"


# ---------------------- DATABASE ----------------------------

def get_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            roll_no TEXT UNIQUE NOT NULL,
            branch TEXT NOT NULL,
            year INTEGER NOT NULL,
            cgpa REAL NOT NULL,
            skills TEXT,
            internship INTEGER DEFAULT 0,
            backlogs INTEGER DEFAULT 0,
            aptitude_score REAL DEFAULT 0
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS companies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            job_role TEXT NOT NULL,
            package REAL NOT NULL,
            min_cgpa REAL NOT NULL,
            eligible_branches TEXT NOT NULL,
            skills_required TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            company_id INTEGER,
            application_date TEXT,
            status TEXT DEFAULT 'Applied',
            FOREIGN KEY(student_id) REFERENCES students(id),
            FOREIGN KEY(company_id) REFERENCES companies(id)
        )
    """)

    conn.commit()
    conn.close()


init_db()


# ---------------------- HELPERS -----------------------------

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def load_students():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM students", conn)
    conn.close()
    return df


def load_companies():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM companies", conn)
    conn.close()
    return df


def load_applications():
    conn = get_connection()
    query = """
        SELECT
            a.id,
            s.name AS student,
            s.roll_no,
            c.name AS company,
            c.job_role,
            c.package,
            a.application_date,
            a.status
        FROM applications a
        JOIN students s ON a.student_id = s.id
        JOIN companies c ON a.company_id = c.id
        ORDER BY a.id DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def calculate_eligibility(student, company):
    if student["cgpa"] < company["min_cgpa"]:
        return False

    if student["backlogs"] > 0:
        return False

    student_branch = student["branch"].strip().lower()
    allowed_branches = [
        x.strip().lower()
        for x in company["eligible_branches"].split(",")
    ]

    if "all" not in allowed_branches and student_branch not in allowed_branches:
        return False

    required = [
        x.strip().lower()
        for x in str(company["skills_required"]).split(",")
        if x.strip()
    ]

    student_skills = [
        x.strip().lower()
        for x in str(student["skills"]).split(",")
        if x.strip()
    ]

    if required:
        if not any(skill in student_skills for skill in required):
            return False

    return True


# ---------------------- SIDEBAR -----------------------------

st.sidebar.title("🎓 Placement Portal")
st.sidebar.write("College Placement Management System")

menu = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard",
        "Student Management",
        "Company & Job Management",
        "Eligibility Checker",
        "Apply for Job",
        "Applications",
        "Placement Prediction",
        "Reports"
    ]
)


# ====================== DASHBOARD ===========================

if menu == "Dashboard":
    st.title("🎓 College Placement Management System")
    st.subheader("Dashboard")

    students = load_students()
    companies = load_companies()
    applications = load_applications()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Students", len(students))
    col2.metric("Companies", len(companies))
    col3.metric("Applications", len(applications))

    placed = len(applications[applications["status"] == "Selected"]) if not applications.empty else 0
    col4.metric("Students Selected", placed)

    st.divider()

    st.info(
        "Use the sidebar to add students and companies, check eligibility, "
        "apply for jobs, track applications, and predict placement chances."
    )

    if not applications.empty:
        st.subheader("Application Status")

        status_count = applications["status"].value_counts()

        fig, ax = plt.subplots()
        ax.bar(status_count.index, status_count.values)
        ax.set_xlabel("Status")
        ax.set_ylabel("Number of Applications")
        ax.set_title("Application Status")
        st.pyplot(fig)


# ================= STUDENT MANAGEMENT ======================

elif menu == "Student Management":
    st.title("👨‍🎓 Student Management")

    tab1, tab2 = st.tabs(["Add Student", "View Students"])

    with tab1:
        with st.form("student_form"):
            name = st.text_input("Student Name")
            roll_no = st.text_input("Roll Number")
            branch = st.selectbox(
                "Branch",
                [
                    "CSE",
                    "CSE-AIML",
                    "CSE-DS",
                    "ECE",
                    "EEE",
                    "MECH",
                    "CIVIL",
                    "CHEMICAL"
                ]
            )
            year = st.selectbox("Year", [1, 2, 3, 4])
            cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.1)
            skills = st.text_input(
                "Skills",
                placeholder="Python, Java, SQL"
            )
            internship = st.checkbox("Completed Internship")
            backlogs = st.number_input("Number of Backlogs", min_value=0, step=1)
            aptitude = st.number_input(
                "Aptitude Score (%)",
                min_value=0.0,
                max_value=100.0,
                step=1.0
            )

            submit = st.form_submit_button("Add Student")

            if submit:
                if not name or not roll_no:
                    st.error("Please enter student name and roll number.")
                else:
                    try:
                        conn = get_connection()
                        conn.execute(
                            """
                            INSERT INTO students
                            (name, roll_no, branch, year, cgpa, skills,
                             internship, backlogs, aptitude_score)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                name,
                                roll_no,
                                branch,
                                year,
                                cgpa,
                                skills,
                                int(internship),
                                backlogs,
                                aptitude
                            )
                        )
                        conn.commit()
                        conn.close()
                        st.success("Student added successfully!")
                    except sqlite3.IntegrityError:
                        st.error("Roll number already exists.")

    with tab2:
        students = load_students()

        if students.empty:
            st.warning("No students added yet.")
        else:
            display = students.copy()
            display["internship"] = display["internship"].map(
                {0: "No", 1: "Yes"}
            )
            st.dataframe(display, use_container_width=True)


# ================= COMPANY MANAGEMENT ======================

elif menu == "Company & Job Management":
    st.title("🏢 Company & Job Management")

    tab1, tab2 = st.tabs(["Add Company", "View Companies"])

    with tab1:
        with st.form("company_form"):
            company_name = st.text_input("Company Name")
            job_role = st.text_input("Job Role")
            package = st.number_input(
                "Package (LPA)",
                min_value=0.0,
                step=0.5
            )
            min_cgpa = st.number_input(
                "Minimum CGPA",
                min_value=0.0,
                max_value=10.0,
                step=0.1
            )
            branches = st.text_input(
                "Eligible Branches",
                value="CSE,ECE"
            )
            required_skills = st.text_input(
                "Required Skills",
                placeholder="Python, Java, SQL"
            )

            submit_company = st.form_submit_button("Add Company")

            if submit_company:
                if not company_name or not job_role:
                    st.error("Enter company name and job role.")
                else:
                    conn = get_connection()
                    conn.execute(
                        """
                        INSERT INTO companies
                        (name, job_role, package, min_cgpa,
                         eligible_branches, skills_required)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            company_name,
                            job_role,
                            package,
                            min_cgpa,
                            branches,
                            required_skills
                        )
                    )
                    conn.commit()
                    conn.close()
                    st.success("Company added successfully!")

    with tab2:
        companies = load_companies()

        if companies.empty:
            st.warning("No companies added yet.")
        else:
            st.dataframe(companies, use_container_width=True)


# ================= ELIGIBILITY CHECKER =====================

elif menu == "Eligibility Checker":
    st.title("✅ Student Eligibility Checker")

    students = load_students()
    companies = load_companies()

    if students.empty or companies.empty:
        st.warning("Please add students and companies first.")
    else:
        student_options = {
            f"{row['name']} ({row['roll_no']})": row["id"]
            for _, row in students.iterrows()
        }

        company_options = {
            f"{row['name']} - {row['job_role']}": row["id"]
            for _, row in companies.iterrows()
        }

        selected_student = st.selectbox(
            "Select Student",
            list(student_options.keys())
        )

        selected_company = st.selectbox(
            "Select Company / Job",
            list(company_options.keys())
        )

        if st.button("Check Eligibility"):
            student_id = student_options[selected_student]
            company_id = company_options[selected_company]

            student = students[students["id"] == student_id].iloc[0]
            company = companies[companies["id"] == company_id].iloc[0]

            eligible = calculate_eligibility(student, company)

            if eligible:
                st.success(
                    f"🎉 {student['name']} is ELIGIBLE for "
                    f"{company['name']} - {company['job_role']}."
                )
            else:
                st.error(
                    f"❌ {student['name']} is NOT eligible for "
                    f"{company['name']} - {company['job_role']}."
                )

            st.write("### Student Details")
            st.write(
                f"**CGPA:** {student['cgpa']}  |  "
                f"**Branch:** {student['branch']}  |  "
                f"**Backlogs:** {student['backlogs']}"
            )

            st.write("### Company Requirements")
            st.write(
                f"**Minimum CGPA:** {company['min_cgpa']}  |  "
                f"**Branches:** {company['eligible_branches']}  |  "
                f"**Skills:** {company['skills_required']}"
            )


# ================= APPLY FOR JOB ===========================

elif menu == "Apply for Job":
    st.title("📝 Apply for Job")

    students = load_students()
    companies = load_companies()

    if students.empty or companies.empty:
        st.warning("Please add students and companies first.")
    else:
        student_options = {
            f"{row['name']} ({row['roll_no']})": row["id"]
            for _, row in students.iterrows()
        }

        company_options = {
            f"{row['name']} - {row['job_role']}": row["id"]
            for _, row in companies.iterrows()
        }

        selected_student = st.selectbox(
            "Student",
            list(student_options.keys())
        )

        selected_company = st.selectbox(
            "Company / Job",
            list(company_options.keys())
        )

        if st.button("Submit Application"):
            student_id = student_options[selected_student]
            company_id = company_options[selected_company]

            student = students[students["id"] == student_id].iloc[0]
            company = companies[companies["id"] == company_id].iloc[0]

            if not calculate_eligibility(student, company):
                st.error("Student is not eligible for this job.")
            else:
                conn = get_connection()

                existing = conn.execute(
                    """
                    SELECT id FROM applications
                    WHERE student_id = ? AND company_id = ?
                    """,
                    (student_id, company_id)
                ).fetchone()

                if existing:
                    st.warning("Student has already applied for this job.")
                else:
                    conn.execute(
                        """
                        INSERT INTO applications
                        (student_id, company_id, application_date, status)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            student_id,
                            company_id,
                            str(date.today()),
                            "Applied"
                        )
                    )
                    conn.commit()
                    st.success("Application submitted successfully!")

                conn.close()


# ================= APPLICATIONS =============================

elif menu == "Applications":
    st.title("📋 Application Management")

    applications = load_applications()

    if applications.empty:
        st.warning("No applications available.")
    else:
        st.dataframe(applications, use_container_width=True)

        st.divider()
        st.subheader("Update Application Status")

        application_id = st.selectbox(
            "Select Application ID",
            applications["id"].tolist()
        )

        new_status = st.selectbox(
            "New Status",
            [
                "Applied",
                "Shortlisted",
                "Interview",
                "Selected",
                "Rejected"
            ]
        )

        if st.button("Update Status"):
            conn = get_connection()
            conn.execute(
                "UPDATE applications SET status = ? WHERE id = ?",
                (new_status, application_id)
            )
            conn.commit()
            conn.close()
            st.success("Application status updated successfully!")
            st.rerun()


# ================= PLACEMENT PREDICTION =====================

elif menu == "Placement Prediction":
    st.title("🤖 Placement Prediction")

    st.write(
        "This module uses a simple Random Forest model to estimate "
        "whether a student has a higher or lower placement chance."
    )

    students = load_students()

    if students.empty:
        st.warning("Add students first.")
    else:
        student_options = {
            f"{row['name']} ({row['roll_no']})": row["id"]
            for _, row in students.iterrows()
        }

        selected = st.selectbox(
            "Select Student",
            list(student_options.keys())
        )

        student_id = student_options[selected]
        student = students[students["id"] == student_id].iloc[0]

        # Synthetic training data for demonstration.
        # In a real college system, replace this with historical placement data.
        np.random.seed(42)
        n = 300

        train = pd.DataFrame({
            "cgpa": np.random.uniform(5.0, 10.0, n),
            "aptitude": np.random.uniform(30, 100, n),
            "internship": np.random.randint(0, 2, n),
            "backlogs": np.random.randint(0, 4, n),
        })

        train["placed"] = (
            (train["cgpa"] >= 7.0).astype(int)
            + (train["aptitude"] >= 60).astype(int)
            + train["internship"]
            - (train["backlogs"] > 0).astype(int)
            >= 2
        ).astype(int)

        X = train[
            ["cgpa", "aptitude", "internship", "backlogs"]
        ]
        y = train["placed"]

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        model.fit(X, y)

        student_data = pd.DataFrame([{
            "cgpa": student["cgpa"],
            "aptitude": student["aptitude_score"],
            "internship": student["internship"],
            "backlogs": student["backlogs"]
        }])

        prediction = model.predict(student_data)[0]
        probability = model.predict_proba(student_data)[0][1] * 100

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Placement Probability",
                f"{probability:.1f}%"
            )

        with col2:
            if prediction == 1:
                st.success("Higher placement chance")
            else:
                st.warning("Lower placement chance")

        st.write("### Student Information")
        st.write(
            f"**CGPA:** {student['cgpa']}  \n"
            f"**Aptitude Score:** {student['aptitude_score']}%  \n"
            f"**Internship:** {'Yes' if student['internship'] else 'No'}  \n"
            f"**Backlogs:** {student['backlogs']}"
        )

        st.caption(
            "Note: This is a demonstration prediction model using "
            "synthetic training data. It should not be treated as a "
            "real admission or hiring decision."
        )


# ================= REPORTS ==================================

elif menu == "Reports":
    st.title("📊 Placement Reports")

    students = load_students()
    companies = load_companies()
    applications = load_applications()

    if students.empty:
        st.warning("No student data available.")
    else:
        col1, col2, col3 = st.columns(3)

        col1.metric("Students", len(students))
        col2.metric("Average CGPA", f"{students['cgpa'].mean():.2f}")

        if not applications.empty:
            selected_count = len(
                applications[applications["status"] == "Selected"]
            )
        else:
            selected_count = 0

        col3.metric("Selected Applications", selected_count)

        st.divider()

        st.subheader("Students by Branch")
        branch_count = students["branch"].value_counts()

        fig1, ax1 = plt.subplots()
        ax1.bar(branch_count.index, branch_count.values)
        ax1.set_xlabel("Branch")
        ax1.set_ylabel("Students")
        ax1.set_title("Students by Branch")
        plt.xticks(rotation=30)
        st.pyplot(fig1)

        if not applications.empty:
            st.subheader("Company-wise Applications")

            company_count = applications["company"].value_counts()

            fig2, ax2 = plt.subplots()
            ax2.bar(company_count.index, company_count.values)
            ax2.set_xlabel("Company")
            ax2.set_ylabel("Applications")
            ax2.set_title("Applications by Company")
            plt.xticks(rotation=30)
            st.pyplot(fig2)

            st.subheader("Application Status Distribution")

            fig3, ax3 = plt.subplots()
            sns.countplot(
                data=applications,
                x="status",
                ax=ax3
            )
            ax3.set_title("Application Status Distribution")
            ax3.set_xlabel("Status")
            ax3.set_ylabel("Count")
            plt.xticks(rotation=20)
            st.pyplot(fig3)

        st.divider()
        st.subheader("Download Data")

        csv_students = students.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Student Data",
            csv_students,
            "students.csv",
            "text/csv"
        )

        if not applications.empty:
            csv_apps = applications.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Application Data",
                csv_apps,
                "applications.csv",
                "text/csv"
            )
