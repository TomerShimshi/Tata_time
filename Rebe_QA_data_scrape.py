

import requests
from bs4 import BeautifulSoup
import codecs
import re

def get_Q_and_A_from_text(text):
    paragraphs = re.split(r'\n+', text.strip())

    # Initialize variables to store questions and answers
    #question =re.search(r'^(.*\n)?[^\n?]*\?', text, re.MULTILINE).group(0).strip()
    
    answer_started = False
    question_started = False
    answer_lines = []
    question_lines = []
    

    # Iterate through paragraphs to identify questions and answers
    for paragraph in paragraphs:
       
        if not answer_started:
            # Look for the start of the answer
            if "תשובה" in paragraph:
                answer_started = True
                question_started = False
        else:
            if 'חזרה ' in paragraph:
             break
            else:
                # Collect lines as part of the answer
                answer_lines.append(paragraph)
        if not question_started:
            if "תשובות מאת" in paragraph:
                question_started = True
        else:
            question_lines.append(paragraph)
    # Join answer lines to form the answer
    answer = "\n".join(answer_lines[:-2])
    question = "\n".join(question_lines[1:])
    return question, answer



if __name__ == "__main__":
    # Replace this URL with the URL of the Hebrew web page you want to scrape
    output_file_path = "Rebe_Q_and_A_dataset_just_rebe_questions.txt"
    for i in range(534):
        url = f"https://www.meshivat-nefesh.org.il/post-{i}/"

        # Send an HTTP request to the web page
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract the text content from the page
            page_text = soup.get_text()

            # Define the output file path (change it as needed)
            
            question, answer = get_Q_and_A_from_text(page_text)
            save_txt = f"### שאלה  \n {question}. ### תשובה  \n {answer}"

            # Save the text to a file with proper encoding for Hebrew (utf-8)
            with codecs.open(output_file_path, 'a', encoding='utf-8') as output_file:
                output_file.write(save_txt)

            print(f"Text saved to {output_file_path} saved quastion num {i}")
        else:
            print(f"Failed to retrieve the web page. Status code: {response.status_code}")