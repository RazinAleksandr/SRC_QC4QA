import re
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import yaml
from ast import literal_eval

# Load the YAML file
"""with open('/home/arazin/main/work/HUAWEI/SRC/QCQA/question_classification/configs/text_with_code_classes.yaml', 'r') as file:
    data = yaml.load(file, Loader=yaml.FullLoader)
"""
# Access the dictionary
#tags_classification = data
tags_classification = 'check'

def preprocess_text(text):
    # Remove newlines and carriage returns
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\r', ' ', text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove non-word characters except spaces
    text = re.sub(r'[^\w\s]', '', text)

    # Convert the text to lower case
    text = text.lower()

    # Remove stop words
    text = ' '.join([word for word in text.split() if word not in stop_words])

    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def get_labels(tags):
    labels = []
    tags_set = set(tags)  # Convert the list of tags to a set
    for class_name, class_tags in tags_classification.items():
        if bool(tags_set & set(class_tags)):
            labels.append(class_name)
    return labels

class DataPreprocessing:
    def __init__(self, df):
        self.df = df

    def preprocess_tags(self):
        self.df['Tags_Q'] = self.df['Tags_Q'].apply(literal_eval) #convert to list type
        #self.df['Tags_Q'] = self.df['Tags_Q'].apply(lambda x: x.split(','))
        #self.df['Tags_A'] = self.df['Tags_A'].astype(str).str.strip('[]').str.replace("'", "").split(', ')

    def preprocess_title(self, func):
        self.df['Title_Q'] = self.df['Title_Q'].apply(func)
    
    def preprocess_body(self, func):
        self.df['Body_Q'] = self.df['Body_Q'].apply(func)
        #self.df['Body_A'] = self.df['Body_A'].apply(func)

    def preprocess_QC4QA(self):
        question_data = self.df[['Title_Q', 'Body_Q', 'Tags_Q']]
        question_data.drop_duplicates('Title_Q',inplace=True)
        question_data.reset_index(inplace=True, drop=True)

        tags_my = []
        for _, v in tags_classification.items(): tags_my.extend(v)
        question_data['choose'] = question_data['Tags_Q'].apply(lambda x: any(i in tags_my for i in x))
        question_data[question_data['choose'] == True]

        question_data['Text'] = "title: " + question_data['Title_Q'] + "\nquestion: " + question_data['Body_Q']
        marked_data = question_data[question_data['choose'] == True]
        marked_data.drop(columns=['choose'], inplace=True)

        marked_data['Label'] = marked_data['Tags_Q'].apply(get_labels)
        #marked_data = marked_data[marked_data['Label'].apply(len) == 1] # for single class
        marked_data.reset_index(drop=True, inplace=True)

        labels = {}
        for i, k in enumerate(list(tags_classification.keys())):
            labels[k] = i
        
        # function to convert list of labels to one-hot encoding
        def convert_labels(label_list):
            one_hot = [0]*len(labels)
            for label in label_list:
                one_hot[labels[label]] = 1
            return one_hot
        
        marked_data['Label'] = marked_data['Label'].apply(convert_labels)
        marked_data.reset_index(drop=True, inplace=True)
        return marked_data[['Text', 'Label']]

    def perform_preprocessing(self, preproc_text=preprocess_text):
        self.preprocess_tags()
        self.preprocess_body(preproc_text)
        self.preprocess_title(preproc_text)
