import { ProfileAnalyzer } from './inference.js';

const analyzer = new ProfileAnalyzer();
const statusEl = document.getElementById('status');
const resultsEl = document.getElementById('results');
const analyzeBtn = document.getElementById('analyzeBtn');

// Initialize models on popup open
analyzer.loadModels().then(() => {
  statusEl.textContent = "Models loaded. Ready to analyze.";
  analyzeBtn.disabled = false;
}).catch(err => {
  statusEl.textContent = "Error loading models.";
  console.error(err);
});

analyzeBtn.addEventListener('click', async () => {
  statusEl.textContent = "Extracting data and analyzing...";
  resultsEl.innerHTML = "";
  analyzeBtn.disabled = true;

  try {
    // 1. Get active tab
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    // 2. Extract DOM data via content script
    const responseData = await chrome.tabs.sendMessage(tab.id, { action: 'GET_PROFILE_DATA' });
    
    // 3. Run Inference
    const nlpScores = await analyzer.analyzeNLP(responseData.text);
    const tabularScores = await analyzer.analyzeTabular(responseData.tabular);
    
    let visualScores = {};
    if (responseData.images.length > 0) {
      visualScores = await analyzer.analyzeVisual(responseData.images[0]);
    }

    // 4. Aggregate and Display Results
    displayResults(nlpScores, tabularScores, visualScores);
    
  } catch (error) {
    statusEl.textContent = "Failed to analyze. Are you on a profile page?";
    console.error(error);
  } finally {
    analyzeBtn.disabled = false;
  }
});

function displayResults(nlp, tabular, visual) {
  statusEl.textContent = "Analysis Complete:";
  
  // Simple thresholding logic (e.g., > 0.6 is a red flag)
  const flags = [];
  const allScores = { ...nlp, ...tabular };
  
  for (const [label, score] of Object.entries(allScores)) {
    if (score > 0.6) {
      flags.push(`<div class="flag">⚠️ ${label.replace('_', ' ').toUpperCase()} (${(score * 100).toFixed(1)}%)</div>`);
    }
  }

  if (flags.length === 0) {
    resultsEl.innerHTML = '<div class="safe">✅ No major red flags detected.</div>';
  } else {
    resultsEl.innerHTML = flags.join('');
  }
}