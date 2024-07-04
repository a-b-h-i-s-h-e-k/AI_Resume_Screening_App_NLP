import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from streamlit.components.v1 import html

nltk.download('punkt')
nltk.download('stopwords')

# Load models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', txt)
    cleanText = re.sub(r'@\S+', '  ', cleanText)
    cleanText = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText

def extract_candidate_name(text):
    # Look for the name following "My name is" or similar phrases
    name_pattern = re.compile(r'My name is\s([A-Z][a-z]+\s[A-Z][a-z]+)', re.IGNORECASE)
    match = name_pattern.search(text)
    if match:
        return match.group(1)
    
    # Fallback to searching for any name pattern at the top of the resume
    name_pattern = re.compile(r'^[A-Z][a-z]+\s[A-Z][a-z]+', re.MULTILINE)
    match = name_pattern.search(text)
    return match.group(0) if match else "Unknown"

def extract_technologies(text):
    # Define categorized lists of common technologies
    tech_dict = {
        "Data Science": ["Python", "R", "Sklearn", "TensorFlow", "PyTorch", "Keras", "Pandas", "NumPy", "Matplotlib", "Seaborn"],
        "Java Developer": ["Java", "Spring", "Hibernate", "Maven", "Gradle", "JSP", "Servlets"],
        "Automation Testing": ["Selenium", "QTP", "LoadRunner", "JMeter", "Cucumber", "TestNG", "JUnit"],
        "Civil Engineer": ["AutoCAD", "STAAD Pro", "Revit", "SAP2000", "ETABS", "Primavera", "MS Project"],
        "DevOps Engineer": ["Docker", "Kubernetes", "Jenkins", "Ansible", "Puppet", "Chef", "Nagios", "Git", "AWS", "Azure"],
        "Web Development": ["HTML", "CSS", "JavaScript", "React", "Angular", "Vue.js", "Node.js", "Django", "Flask"],
        "Database": ["SQL", "MySQL", "PostgreSQL", "Oracle", "MongoDB", "SQLite"],
        "Mechanical Engineer": ["SolidWorks", "AutoCAD", "ANSYS", "CATIA", "MATLAB"],
        "Electrical Engineer": ["MATLAB", "Simulink", "PSpice", "Multisim", "LabVIEW"],
    }

    known_technologies = []
    for category, technologies in tech_dict.items():
        for tech in technologies:
            if tech.lower() in text.lower():
                known_technologies.append(tech)
    
    return known_technologies

def generate_job_search_links(category_name):
    base_urls = {
        "Indeed": "https://www.indeed.com/jobs?q=",
        "Glassdoor": "https://www.glassdoor.com/Job/jobs.htm?sc.keyword=",
        "Monster": "https://www.monster.com/jobs/search/?q="
    }
    query = category_name.replace(" ", "+")
    links = {site: base_url + query for site, base_url in base_urls.items()}
    return links

# Web app
def main():
    # Set the background color to black and text color to white
    st.markdown(
        """
        <style>
        .stApp {
            background-color: black;
            color: white;
        }
        .red-button {
            background-color: #FF4136;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Resume Screening App")
    st.write("Upload a resume to predict the category and extract key information.")

    uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])
    
    if st.button("Refresh", key="refresh_button", help="Refresh to upload a new resume"):
        st.experimental_rerun()
    
    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try decoding with 'latin-1'
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = cleanResume(resume_text)
        input_features = tfidf.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]

        # Map category ID to category name
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and Fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }

        category_name = category_mapping.get(prediction_id, "Unknown")

        # Extract candidate name and technologies known
        candidate_name = extract_candidate_name(resume_text)
        known_technologies = extract_technologies(resume_text)
        
        # Generate job search links
        job_search_links = generate_job_search_links(category_name)

        # Display the results
        st.write(f"Predicted Category: {category_name}")
        st.write("Candidate Name:", candidate_name)
        st.write("Technologies Known:", ", ".join(known_technologies))
        
        st.write("Search for jobs in the predicted category:")
        for site, link in job_search_links.items():
            if st.button(f"Search on {site}", key=site, help=f"Search for {category_name} jobs on {site}"):
                st.markdown(f"[{link}]({link})", unsafe_allow_html=True)

# Python main
if __name__ == "__main__":
    main()
