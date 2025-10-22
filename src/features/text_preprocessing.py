import re # Bu import'ları ekleyin
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# Gerekliyse:
# import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')

def clean_text(text: str) -> str:
    """Metni küçük harfe çevirir, noktalama ve sayıları kaldırır."""
    # Notebook'taki ilgili lambda fonksiyonlarınız [cite: 218]
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

def remove_stopwords(text: str) -> str:
    """Durak kelimeleri kaldırır."""
    # Notebook'taki stop-word kodunuz [cite: 218]
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

def lemmatize_text(text: str) -> str:
    """Kelimeleri köklerine indirger (lemmatization)."""
    # Notebook'taki lemmatization kodunuz [cite: 219]
    lemmatizer = WordNetLemmatizer()
    tokens = text.split()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized_tokens)

def full_text_pipeline(text: str) -> str:
    """Tüm metin işleme adımlarını birleştirir."""
    # Bu, notebook'taki zincirleme .apply() çağrılarınızı birleştirir
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text