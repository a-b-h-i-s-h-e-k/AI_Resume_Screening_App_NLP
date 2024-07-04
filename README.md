# AI_Resume_Screening_App_NLP

Resume_Screeing_AI

Resume Screening App
This application is designed to screen resumes, predict the category of the resume, extract key information, and provide job search links based on the predicted category.

Features
Upload Resume: Supports .txt and .pdf file formats.
Predict Resume Category: Uses a machine learning model to predict the category of the resume.
Extract Candidate Information: Extracts candidate name and known technologies from the resume.
Job Search Links: Provides links to job search websites for the predicted category.

Installation


Clone the repository:
git clone https://gitlab.inf.unibz.it/Abhishek.Bargujar/resume_screeing_ai/-/edit/main/README.md?ref_type=heads
cd <Resume_Screening_AI>


Create and activate a virtual environment:
python3 -m venv myenv
source myenv/bin/activate  # On Windows use myenv\Scripts\activate


Install dependencies:
Install require dependencies and import them correctly.



Usage


Run the application:
streamlit run app.py


Upload a resume:
Use the web interface to upload a resume in .txt or .pdf format.


View predictions and extract information:
The app will display the predicted category, candidate name, known technologies, and provide job search links.



File Descriptions

app.py: The main application script.
clf.pkl: The pickled classifier model.
tfidf.pkl: The pickled TF-IDF vectorizer.
UpdatedResumeDataSet.csv: The dataset used for training the model.
extra.txt: Additional setup instructions.
Resume Screeing App.ipynb: Jupyter notebook with the code for training and evaluating the model.



Additional Setup

Ensure the required NLTK data files are downloaded:
import nltk
nltk.download('punkt')
nltk.download('stopwords')