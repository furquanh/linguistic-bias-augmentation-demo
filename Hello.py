# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger
from openai import OpenAI
import pandas as pd
import time

LOGGER = get_logger(__name__)

@st.cache_data(persist="disk") 
def load_submissions():
    try:
        return pd.read_csv("submissions.csv").reset_index(drop=True)
    except FileNotFoundError:
        # Return an empty DataFrame if the file doesn't exist
        return pd.DataFrame(columns=["Augmentation Name", "Explanation"]).reset_index(drop=True)

@st.cache_data(persist="disk")    
def add_submission(submissions, entry):
    new_entry = pd.DataFrame([entry])
    updated_submissions = pd.concat([submissions, new_entry], axis=0, ignore_index=True)
    updated_submissions.to_csv("submissions.csv", index=False)
    return updated_submissions

@st.cache_resource
def get_client():
    client = OpenAI(api_key=st.secrets["openai_api_key"])
    return client

def augment_text(augmentation_type, prompt, client):
 with st.spinner(f'Applying {augmentation_type}...'):   
    response = client.chat.completions.create(
      model="gpt-4-1106-preview",
      messages=[
        {"role": "system", "content": "You are an expert linguist. Do not include explaination, only reply back with the sentence."},
        {"role": "user", "content": prompt},
      ]
    )
    return response.choices[0].message.content


def main():
    st.title('Linguistic Bias Dataset Project - Sentence Augmentation Demo')
    sentence = st.text_area('Enter a sentence:', height=100)
    augment_button = st.button('Generate Augmentations')
    client = get_client()
    submissions = load_submissions()


    if augment_button and sentence:
          prompt_map = {"African American English": "Convert the following sentence to African American Vernacular English: ", 
            "Filler Words": "Insert filler words into this sentence: ",
            "Hashtags": "Add hashtags into this sentence to mimic language on twitter: ",
            "Emojify ": "Add emojis the following sentence mimicing language used in texting and social media: ",
            "Formalize": "Convert the text style from informal to formal english: ",
            "Misspelling": "Insert common misspelling within this sentence: ",
            "Mixed Language": "Pick a single word from this sentece and replace it with it's translation to one other language, mimicing a mistake a bilingual person might do: ",
            "Subject-Verb Agreement Errors": "Modify the sentence it so that the subject and verb do not agree in number: ",
            "Run-on Sentences and Comma Splices": "Transform the following sentence into a run-on sentence or a comma splice: ",
            "Sentence Fragments": "Convert this sentence into a sentence fragment by removing essential elements: ",
            "Incorrect Tense Use": "Change the tense in the following sentence inappropriately: ",
            "Misplaced or Dangling Modifiers": "Rearrange the following sentence to create a misplaced or dangling modifier: ",
            "Wrong Word Order": "Rearrange the words in this sentence into an incorrect but syntactically possible order: ",
            "Pronoun-Antecedent Agreement Errors": "modify this sentence so that the pronouns do not agree in number with their antecedents: ",
            "Incorrect Use of Articles": "modify this sentence so to show incorrect use of articles: ",
            "Mixed Constructions": "Change this sentence to have mixed constructions, starting in one grammatical structure and ending in another: "
            }


          augmented_sentences = []
          progress_text = "Applying augmentations. Please wait."
          my_bar = st.progress(0, text=progress_text)
          i = 6
          for augmentation_type, prompt_format in prompt_map.items():
              prompt = prompt_format + sentence
              augmented_sentence = augment_text(augmentation_type, prompt, client)
              my_bar.progress(i, text=progress_text)
              i += 6  
              augmented_sentences.append((augmentation_type, augmented_sentence))
          my_bar.empty()
          df = pd.DataFrame(augmented_sentences, columns=['Augmentation Type', 'Augmented Text'])
          st.table(df)

          with st.expander("See explanation"):
            st.markdown('''
    **Subject-Verb Agreement Errors**: Occur when the verb form does not agree with the subject in number (singular or plural). For example, "She walk to school every day" instead of "She walks to school every day."

    **Run-on Sentences and Comma Splices**: These happen when two or more independent clauses are incorrectly joined without proper punctuation or conjunction. For example, "I went shopping I bought a dress."

    **Sentence Fragments**: This error involves incomplete sentences that lack either a subject, a verb, or a complete thought. For instance, "Because I went to the store."
                        
    **Incorrect Tense Use:** Using the wrong tense can lead to syntactic confusion, like using past tense instead of present, or vice versa.

    **Misplaced or Dangling Modifiers**: These mistakes occur when a modifier (a word, phrase, or clause that describes something else) is not clearly or logically related to the word it modifies. For example, "Running quickly, the goal seemed impossible to reach" (misplaced modifier).

    **Wrong Word Order**: English typically follows a Subject-Verb-Object (SVO) order. Deviations can cause confusion, e.g., "The cat the mouse chased."

    **Pronoun-Antecedent Agreement Errors**: This occurs when a pronoun does not agree in number with its antecedent. For example, "Every student must bring their pencil." ('Their' should be 'his or her' to agree with 'every student').

    **Incorrect Use of Articles**: Mistakes involving the use of 'a', 'an', and 'the' can affect sentence structure and meaning.

    **Mixed Constructions**: These errors happen when a sentence starts with one construction and then abruptly changes to another, leading to confusion.
                        ''')

    ##
    st.header("Suggest a New Text Augmentation")
    st.write("The augmentation should be syntatic, think of the common mistakes people do when writing english.")
    with st.form(key='augmentation_form'):
        aug_name = st.text_input("Augmentation Name")
        aug_reason = st.text_area("Explain why", height=100)
        submit_button = st.form_submit_button(label='Submit')

        if submit_button and aug_name and aug_reason:
            new_submission = {"Augmentation Name": aug_name, "Explanation": aug_reason}
            submissions = add_submission(submissions, new_submission)
            #e#dited_df.append(new_submission, ignore_index=True)
            st.success("Thank you for your suggestion!")
            time.sleep(2)
            st.rerun()

    st.header("Suggested Augmentations:")
    st.dataframe(submissions, use_container_width=False)

if __name__ == "__main__":
    main()
