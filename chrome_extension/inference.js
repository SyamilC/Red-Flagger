import * as tf from '@tensorflow/tfjs';

export class ProfileAnalyzer {
  constructor() {
    this.nlpConfig = null;
    this.tabularModel = null;
    this.visualModel = null;
    this.isLoaded = false;
  }

  async loadModels() {
    if (this.isLoaded) return;

    // 1. Load NLP JSON Config
    const nlpRes = await fetch(chrome.runtime.getURL('models/nlp_model.json'));
    this.nlpConfig = await nlpRes.json();

    // 2. Load Tabular Keras Model
    this.tabularModel = await tf.loadLayersModel(chrome.runtime.getURL('models/tabular_model_tfjs/model.json'));

    // 3. Load Visual YOLOv8 Graph Model
    this.visualModel = await tf.loadGraphModel(chrome.runtime.getURL('models/yolo_model_tfjs/model.json'));
    
    this.isLoaded = true;
  }

  async analyzeNLP(text) {
    const words = text.toLowerCase().match(/\b\w+\b/g) || [];
    const docLength = words.length || 1;
    const counts = {};
    words.forEach(w => counts[w] = (counts[w] || 0) + 1);

    const scores = {};
    this.nlpConfig.labels.forEach((label, i) => {
      let logit = this.nlpConfig.intercepts[i];
      for (const [word, count] of Object.entries(counts)) {
        const featIdx = this.nlpConfig.vocab[word];
        if (featIdx !== undefined) {
          const idf = this.nlpConfig.idf[featIdx] || 1.0;
          const tfidf = (count / docLength) * idf;
          logit += tfidf * this.nlpConfig.coefs[i][featIdx];
        }
      }
      // Sigmoid activation
      scores[label] = 1 / (1 + Math.exp(-logit));
    });
    return scores;
  }

  async analyzeTabular(profileData) {
    // Note: In production, apply the exact same OneHot/Standard scaling as the Python preprocessor here.
    // For this demo, we create a dummy tensor matching the trained model's expected input shape (e.g., 237 features)
    const dummyInput = tf.zeros([1, 237]); 
    
    const prediction = this.tabularModel.predict(dummyInput);
    const probs = await prediction.data();
    
    dummyInput.dispose();
    prediction.dispose();

    const labels = ["aggressive_tone", "hookup_focus", " Negativity", "sarcasm_cynicism", "substance_risk", "incomplete_profile"];
    const scores = {};
    labels.forEach((label, i) => scores[label] = probs[i]);
    
    return scores;
  }

  async analyzeVisual(imageSrc) {
    return new Promise((resolve) => {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = async () => {
        const tensor = tf.browser.fromPixels(img)
          .resizeNearestNeighbor([640, 640])
          .expandDims(0)
          .toFloat()
          .div(255.0);

        const output = this.visualModel.predict(tensor);
        const data = await output.data();
        
        tensor.dispose();
        output.dispose();
        
        // Simplified YOLOv8 post-processing (Returns max confidence for "person" class as demo)
        resolve({ person_confidence: Math.max(...data.slice(0, 100)) }); 
      };
      img.src = imageSrc;
    });
  }
}