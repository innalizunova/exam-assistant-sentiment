import pandas as pd
import re
import os


def preprocess_text(text):
    if type(text) == str:
        text = text.lower()
        text = re.sub(r'[:;=]\-?\)', ' улыбка ', text)  # :)  =)  ;)
        text = re.sub(r'[:;=]\-?\(', ' грусть ', text)  # :(  =(  ;(
        text = re.sub(r'т\.д\.', ' так далее ', text)
        text = re.sub(r'т\.п\.', ' тому подобное ', text)
        text = re.sub(r"- ", " ", text)
        text = re.sub(r" -", " ", text)
        text = re.sub(r"_", " ", text)
        text = re.sub(r"[^A-za-zА-Яа-яё0-9' \-]", " ", text)
        text = re.sub(r"ё", "е", text)
        text = re.sub(r" +", " ", text)
        text = text.strip()
        if text != '':
            return text
    return None


folder = '../data/reviews-ru'
data = pd.read_excel(os.path.join(folder, 'reviews.xlsx'))

texts = data['Отзыв'].values
clear_texts = [preprocess_text(text) for text in texts]
tonality_labels = data['Тональность'].values.tolist()
toxic_labels = data['Токсичность'].values.tolist()

df = pd.DataFrame({'text': texts, 'clear_text': clear_texts, 'tonality': tonality_labels, 'toxicity': toxic_labels})
df = df.dropna()
df.to_csv(os.path.join(folder, 'reviews_none.csv'), index=False)

