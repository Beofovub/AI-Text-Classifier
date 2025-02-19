from transformers import pipeline

def classify_text(text):
    """Classifies text using a pre-trained DistilBERT model."""
    classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    results = classifier(text)
    return results

if __name__ == "__main__":
    text = input("Enter text to classify: ")
    classification = classify_text(text)
    print("Classification Result:", classification)
