import pickle
from dl4 import PoemClassifer

def load_model(model_path):
    """Model dosyasını yükle"""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        print(f"Model dosyası bulunamadı: {model_path}")
        return None
    except Exception as e:
        print(f"Model yüklenirken hata: {e}")
        return None

def main():
    print("🎭 Şiir Modeli Kullanılıyor")
    print("=" * 30)

    # Model dosyasını yükle
    model_data = load_model("best_poem_classifier_svm.pkl")
    if model_data is None:
        print("❌ Model yüklenemedi!")
        return

    # Model ve vectorizer'ı al
    model = model_data['model']
    vectorizer = model_data['vectorizer']
    model_name = model_data['model_name']
    
    print(f"✅ {model_name} modeli yüklendi!")
    print("💡 Çıkmak için 'q' yazın")
    print("-" * 30)
    
    # Şiir sınıflandırma döngüsü
    while True:
        user_input = input("\n📝 Şiir giriniz: ")
        
        if user_input.lower() == "q":
            print("👋 Program sonlandırılıyor...")
            break

        if not user_input.strip():
            print("⚠️ Lütfen bir şiir giriniz!")
            continue

        try:
            # Metni temizle
            cleaned_text = user_input.lower()
            cleaned_text = cleaned_text.replace('\n', ' ').strip()
            
            # Özellik çıkar
            features = vectorizer.transform([cleaned_text])
            
            # Tahmin yap
            prediction = model.predict(features)[0]
            
            # Olasılık tahmini (SVM için özel kontrol)
            try:
                probability = model.predict_proba(features)[0]
                confidence = max(probability)
                
                # Tüm sınıfların olasılıklarını göster
                classes = model.classes_
                print("\n📈 Tüm Sınıf Olasılıkları:")
                for i, cls in enumerate(classes):
                    print(f"   {cls}: {probability[i]:.3f}")
                    
            except AttributeError:
                # SVM için olasılık tahmini yoksa sadece tahmin göster
                confidence = "N/A"
                print("\n⚠️ Bu model olasılık tahmini desteklemiyor")
            
            # Sonuçları göster
            print(f"\n🎯 Tahmin: {prediction}")
            print(f"📊 Güven: {confidence}")
                
        except Exception as e:
            print(f"❌ Hata: {e}")
            print("Lütfen geçerli bir metin giriniz.")
    
if __name__ == "__main__":
    main()
