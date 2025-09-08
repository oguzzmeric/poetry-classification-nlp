import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import nltk
import re
import warnings
import pickle
import os
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings('ignore')

try : 
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

class PoemClassifer:
    def __init__(self):
        self.models = {}
        self.vectorizer = None
        self.label_encoder = None


    def load_data(self,train_path,test_path):

        self.train_data = pd.read_csv(train_path)
        self.test_data = pd.read_csv(test_path)

        print("sınıf dağılımı ")
        print(self.train_data['Genre'].value_counts())

        return self.train_data,self.test_data

    def preprocess_text(self,text):
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def prepare_data(self):
        print("temizlik")

        self.train_data['cleaned_poem'] = self.train_data['Poem'].apply(self.preprocess_text)
        self.test_data['cleaned_poem'] = self.test_data['Poem'].apply(self.preprocess_text)

        self.train_data = self.train_data[self.train_data['cleaned_poem'].str.len()> 0 ]
        self.test_data = self.test_data[self.test_data['cleaned_poem'].str.len()> 0 ]

        print("temizlendi")

        return self.train_data,self.test_data

    def extract_features(self,method='tfidf'):
        print("özellik çıkarıyoruz")

        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1,2),
                min_df = 2,
                max_df = 0.95,
                stop_words='english'
            )

            x_train = self.vectorizer.fit_transform(self.train_data['cleaned_poem'])
            x_test = self.vectorizer.transform(self.test_data['cleaned_poem'])

        print("özellik çıkarıldı")

        return x_train,x_test

    def train_models(self,X_train,y_train):
        print("modeller eğitiliyor")

        models = {
            'Naive Bayes': MultinomialNB(alpha=0.1),
            'SVM': SVC(kernel='linear',C=1.0,random_state=42,probability=True),
            'Random Forest': RandomForestClassifier(n_estimators=100,random_state=42),
        }

        for name,model in models.items():
            print(f"📚 {name} eğitiliyor...")
            model.fit(X_train,y_train)
            self.models[name] = model
            print(f"✅ {name} eğitimi tamamlandı")
        
        return self.models
    
    def evaluate_models(self,X_test,y_test):
        results = {

        }

        for name,model in self.models.items():
            print(f"\n🔍 {name} değerlendiriliyor...")

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test,y_pred)
            results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred
            }

            print(f"📈 Accuracy: {accuracy:.4f}")
            
            print(f"📋 Classification Report:")
            print(classification_report(y_test,y_pred))
        
        return results
    def visualize_results(self, results, y_test):
        """Sonuçları görselleştir"""
        print("📊 Sonuçlar görselleştiriliyor...")
        
        # Accuracy karşılaştırması
        plt.figure(figsize=(12, 8))
        
        # 1. Accuracy karşılaştırması
        plt.subplot(2, 2, 1)
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        
        bars = plt.bar(model_names, accuracies, color=['skyblue', 'lightgreen', 'lightcoral'])
        plt.title('Model Accuracy Karşılaştırması', fontsize=14, fontweight='bold')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        
        # Değerleri bar üzerine yaz
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Confusion Matrix (en iyi model için)
        best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
        plt.subplot(2, 2, 2)
        
        cm = confusion_matrix(y_test, results[best_model]['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=sorted(y_test.unique()),
                   yticklabels=sorted(y_test.unique()))
        plt.title(f'Confusion Matrix - {best_model}', fontweight='bold')
        plt.ylabel('Gerçek')
        plt.xlabel('Tahmin')
        
        # 3. Sınıf dağılımı
        plt.subplot(2, 2, 3)
        y_test.value_counts().plot(kind='bar', color='lightblue')
        plt.title('Test Verisi Sınıf Dağılımı', fontweight='bold')
        plt.xlabel('Sınıf')
        plt.ylabel('Sayı')
        plt.xticks(rotation=45)
        
        # 4. Model performans detayları
        plt.subplot(2, 2, 4)
        plt.text(0.1, 0.8, f'En İyi Model: {best_model}', fontsize=12, fontweight='bold')
        plt.text(0.1, 0.7, f'Accuracy: {results[best_model]["accuracy"]:.4f}', fontsize=10)
        
        # Tüm modellerin accuracy'si
        y_pos = 0.6
        for name, acc in zip(model_names, accuracies):
            plt.text(0.1, y_pos, f'{name}: {acc:.4f}', fontsize=10)
            y_pos -= 0.1
        
        plt.axis('off')
        plt.title('Model Performans Özeti', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('poem_classification_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return best_model

    def save_model(self,model_name,filename=None):
        if filename is None:
            filename = f'best_poem_classifier_{model_name.lower().replace(" ", "_")}.pkl'
        
        model_data = {
            'model': self.models[model_name],
            'vectorizer': self.vectorizer,
            'model_name': model_name
        }
        with open(filename,'wb') as f:
            pickle.dump(model_data,f)
        
        print(f"💾 Model kaydedildi: {filename}")
        return filename
    
    def predict_new_poem(self,poem_text,model_name=None):
        if model_name is None:
            model_name = list(self.models.keys())[0]

        cleaned_text = self.preprocess_text(poem_text)

        features = self.vectorizer.transform([cleaned_text])

        prediction = self.models[model_name].predict(features)[0]
        probability = self.models[model_name].predict_proba(features)[0]

        return prediction,probability

def main():

    classifier = PoemClassifer()

    train_data,test_data = classifier.load_data(
        'Poem_classification - train_data.csv',
        'Poem_classification - test_data.csv'
    )

    train_data,test_data = classifier.prepare_data()
    x_train,x_test = classifier.extract_features('tfidf')

    y_train = train_data['Genre']
    y_test = test_data['Genre']

    models = classifier.train_models(x_train,y_train)

    results = classifier.evaluate_models(x_test,y_test)

    best_model = classifier.visualize_results(results,y_test)

    classifier.save_model(best_model)

    return classifier,results,best_model




if __name__ == "__main__":
    classifier,results,best_model = main()
    
    print("\n✨ Proje başarıyla tamamlandı!")