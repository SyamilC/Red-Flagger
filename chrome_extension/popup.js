import { ProfileAnalyzer } from './inference.js';

const analyzer = new ProfileAnalyzer();
const statusEl = document.getElementById('status');
const resultsEl = document.getElementById('results');
const analyzeBtn = document.getElementById('analyzeBtn');
const moduleEl = document.getElementById('moduleStatus');

function cleanText(value) {
  return String(value || '').replace(/\s+/g, ' ').trim();
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function percent(value) {
  return `${Math.round(Number(value || 0) * 1000) / 10}%`;
}

function riskClass(level) {
  const text = String(level || '').toLowerCase();
  if (text.includes('high')) return 'high';
  if (text.includes('moderate')) return 'warn';
  return 'ok';
}

function visibleTextOf(selector) {
  return Array.from(document.querySelectorAll(selector))
    .filter((element) => {
      const style = window.getComputedStyle(element);
      const rect = element.getBoundingClientRect();
      return style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 0 && rect.height > 0;
    })
    .map((element) => cleanText(element.innerText || element.textContent || ''))
    .filter(Boolean);
}

function firstMatch(regex, text, fallback = '') {
  const match = String(text || '').match(regex);
  return match ? cleanText(match[1] || match[0]) : fallback;
}

function inferChoice(text, rules, fallback) {
  const lower = String(text || '').toLowerCase();
  for (const [value, patterns] of Object.entries(rules)) {
    if (patterns.some((pattern) => pattern.test(lower))) return value;
  }
  return fallback;
}

function extractAge(text) {
  const direct = firstMatch(/\b(?:age[:\s]*)?([1-9][0-9])\b/i, text, '');
  const age = Number(direct);
  return age >= 18 && age <= 90 ? age : 27;
}

function extractProfileDataInPage() {
  const localCleanText = (value) => String(value || '').replace(/\s+/g, ' ').trim();
  const localVisibleTextOf = (selector) => Array.from(document.querySelectorAll(selector))
    .filter((element) => {
      const style = window.getComputedStyle(element);
      const rect = element.getBoundingClientRect();
      return style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 0 && rect.height > 0;
    })
    .map((element) => localCleanText(element.innerText || element.textContent || ''))
    .filter(Boolean);
  const localFirstMatch = (regex, text, fallback = '') => {
    const match = String(text || '').match(regex);
    return match ? localCleanText(match[1] || match[0]) : fallback;
  };
  const localInferChoice = (text, rules, fallback) => {
    const lower = String(text || '').toLowerCase();
    for (const [value, patterns] of Object.entries(rules)) {
      if (patterns.some((pattern) => pattern.test(lower))) return value;
    }
    return fallback;
  };
  const localExtractAge = (text) => {
    const direct = localFirstMatch(/\b(?:age[:\s]*)?([1-9][0-9])\b/i, text, '');
    const age = Number(direct);
    return age >= 18 && age <= 90 ? age : 27;
  };
  const prioritySelectors = [
    '[data-test*="bio" i]',
    '[data-testid*="bio" i]',
    '[class*="bio" i]',
    '[class*="about" i]',
    '[id*="bio" i]',
    '[id*="about" i]',
    'main',
    'article',
    'section'
  ];
  const profileText = prioritySelectors.flatMap(localVisibleTextOf).join(' ');
  const fallbackText = localCleanText(document.body?.innerText || '');
  const text = localCleanText(profileText || fallbackText).slice(0, 6000);
  const fullText = localCleanText(`${document.title || ''} ${fallbackText}`);
  const imageElements = Array.from(document.images || []);
  const metaImage = document.querySelector('meta[property="og:image"], meta[name="twitter:image"]')?.content;
  const images = imageElements
    .map((img) => ({
      src: img.currentSrc || img.src || '',
      alt: localCleanText(img.alt || ''),
      className: localCleanText(img.className || ''),
      width: Number(img.naturalWidth || img.width || 0),
      height: Number(img.naturalHeight || img.height || 0)
    }))
    .filter((img) => img.src)
    .sort((a, b) => (b.width * b.height) - (a.width * a.height));
  if (metaImage && !images.some((img) => img.src === metaImage)) {
    images.unshift({ src: metaImage, alt: 'open graph image', className: 'meta', width: 0, height: 0 });
  }
  const tabular = {
    age: localExtractAge(fullText),
    height: 170,
    income: -1,
    drinks: localInferChoice(fullText, { 'very often': [/drinks?\s+very often/, /heavy drink/], often: [/drinks?\s+often/, /part(y|ies) often/], socially: [/drinks?\s+socially/, /social drink/], rarely: [/drinks?\s+rarely/], 'not at all': [/does not drink/, /no alcohol/, /sober/] }, 'socially'),
    drugs: localInferChoice(fullText, { often: [/drugs?\s+often/], sometimes: [/drugs?\s+sometimes/, /420 friendly/, /weed/, /cannabis/], never: [/no drugs/, /drug free/, /never.*drugs/] }, 'never'),
    smokes: localInferChoice(fullText, { yes: [/smokes?\s+yes/, /smoker/], sometimes: [/smokes?\s+sometimes/, /trying to quit/], no: [/non smoker/, /does not smoke/, /no smoking/] }, 'no'),
    status: localInferChoice(fullText, { married: [/married/], 'seeing someone': [/seeing someone/, /in a relationship/], single: [/single/] }, 'single'),
    sex: localInferChoice(fullText, { f: [/\bwoman\b/, /\bfemale\b/], m: [/\bman\b/, /\bmale\b/] }, 'm'),
    orientation: localInferChoice(fullText, { bisexual: [/bisexual/, /\bbi\b/], gay: [/\bgay\b/, /lesbian/], straight: [/straight/] }, 'straight'),
    body_type: localInferChoice(fullText, { athletic: [/athletic/], fit: [/\bfit\b/], thin: [/\bthin\b/], curvy: [/curvy/], average: [/average/] }, 'average'),
    diet: localInferChoice(fullText, { vegan: [/vegan/], vegetarian: [/vegetarian/], halal: [/halal/], kosher: [/kosher/], 'mostly anything': [/anything/, /foodie/] }, 'mostly anything'),
    education: localInferChoice(fullText, { 'graduated from masters program': [/masters/, /master's/], 'graduated from college/university': [/college/, /university/, /degree/], 'graduated from high school': [/high school/] }, 'graduated from college/university'),
    ethnicity: localFirstMatch(/\b(asian|white|black|hispanic|latino|middle eastern|indian|pacific islander)\b/i, fullText, 'not specified'),
    job: localFirstMatch(/\b(?:job|work|works as|career)[:\s]+([a-zA-Z /-]{2,40})/i, fullText, 'other'),
    offspring: localInferChoice(fullText, { 'has kids': [/has kids/, /has children/, /\bdad\b/, /\bmom\b/], "doesn't want kids": [/doesn't want kids/, /does not want kids/, /no kids/], "doesn't have kids": [/no children/, /no kids yet/] }, "doesn't have kids"),
    pets: localInferChoice(fullText, { 'likes dogs and likes cats': [/dogs?.*cats?|cats?.*dogs?/], 'likes dogs': [/dog/], 'likes cats': [/cat/] }, 'likes dogs'),
    religion: localFirstMatch(/\b(agnostic|atheist|christian|muslim|islam|hindu|buddhist|jewish|catholic)\b/i, fullText, 'agnosticism'),
    sign: localFirstMatch(/\b(aries|taurus|gemini|cancer|leo|virgo|libra|scorpio|sagittarius|capricorn|aquarius|pisces)\b/i, fullText, 'leo')
  };
  const missingFieldCount = Object.keys(tabular).filter((field) => tabular[field] === 'not specified' || tabular[field] === '').length;
  return {
    text,
    tabular: { ...tabular, missingFieldCount },
    images: images.slice(0, 8),
    title: document.title || '',
    url: location.href
  };
}

async function getProfileFromActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) throw new Error('No active tab found.');

  try {
    return await chrome.tabs.sendMessage(tab.id, { action: 'GET_PROFILE_DATA' });
  } catch (messageError) {
    const [result] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: extractProfileDataInPage
    });
    if (!result?.result) throw messageError;
    return result.result;
  }
}

function renderModuleStatus() {
  const status = analyzer.getStatus();
  moduleEl.innerHTML = `
    <div class="module-pill ${status.tfjsReady ? 'ok' : 'warn'}">TFJS ${status.tfjsReady ? 'ready' : 'loading'}</div>
    <div class="module-pill ${status.nlpReady ? 'ok' : 'warn'}">NLP JSON ${status.nlpReady ? 'loaded' : 'missing'}</div>
    <div class="module-pill warn">Visual ${escapeHtml(status.visualMode)}</div>
  `;
}

function renderFlagList(title, labels, combined) {
  if (!labels.length) {
    return `<section class="panel"><h4>${escapeHtml(title)}</h4><p class="muted">No major flags crossed threshold.</p></section>`;
  }
  const rows = labels.slice(0, 8).map((label) => {
    const info = combined[label] || {};
    return `
      <div class="flag-row">
        <div>
          <strong>${escapeHtml(label)}</strong>
          <small>${escapeHtml(info.description || '')}</small>
          ${info.evidence?.length ? `<small>Evidence: ${escapeHtml(info.evidence.join(', '))}</small>` : ''}
        </div>
        <span>${percent(info.probability)}</span>
      </div>
    `;
  }).join('');
  return `<section class="panel"><h4>${escapeHtml(title)}</h4>${rows}</section>`;
}

function renderResults(audit) {
  const visualFlags = audit.detected_visual_flags || [];
  const visual = audit.module_outputs.visual_dom || {};
  const profile = audit.profile_summary || {};
  const riskClassName = riskClass(audit.overall_risk_level);

  resultsEl.innerHTML = `
    <section class="score-card ${riskClassName}">
      <div>
        <span class="score">${escapeHtml(audit.overall_risk_score)}</span>
        <span class="level">${escapeHtml(audit.overall_risk_level)}</span>
      </div>
      <div class="score-bar"><span></span></div>
    </section>

    ${renderFlagList('Text and Metadata Flags', audit.detected_text_tabular_flags || [], audit.combined_label_scores || {})}

    <section class="panel">
      <h4>Visual Evidence</h4>
      <div class="chips">${visualFlags.map((flag) => `<span>${escapeHtml(flag)}</span>`).join('') || '<span>clean</span>'}</div>
      <p class="muted">${escapeHtml(visual.note || 'Visual DOM heuristic completed.')}</p>
      ${(visual.evidence || []).map((item) => `<p class="muted">- ${escapeHtml(item)}</p>`).join('')}
    </section>

    <section class="panel compact">
      <h4>Extracted Profile</h4>
      <p><b>Words:</b> ${escapeHtml(profile.bio_word_count)}</p>
      <p><b>Images:</b> ${escapeHtml(profile.images_found)}</p>
      <p><b>Page:</b> ${escapeHtml(profile.title || 'current tab')}</p>
    </section>

    <details class="panel">
      <summary>Bio excerpt</summary>
      <p class="excerpt">${escapeHtml(audit.bio_excerpt || 'No bio text found.')}</p>
    </details>
  `;
  const bar = resultsEl.querySelector('.score-bar span');
  if (bar) {
    bar.style.width = `${Math.min(100, Number(audit.overall_risk_score || 0))}%`;
  }
}

async function boot() {
  analyzeBtn.disabled = true;
  statusEl.textContent = 'Loading client-side models...';
  try {
    await analyzer.loadModels();
    renderModuleStatus();
    statusEl.textContent = 'Ready. Open a profile-like page and run the audit.';
    analyzeBtn.disabled = false;
  } catch (error) {
    statusEl.textContent = 'Model load failed. Check extension/models/nlp_model.json.';
    moduleEl.innerHTML = `<div class="module-pill high">${escapeHtml(error.message)}</div>`;
    console.error(error);
  }
}

analyzeBtn.addEventListener('click', async () => {
  analyzeBtn.disabled = true;
  statusEl.textContent = 'Reading page DOM and running client-side audit...';
  resultsEl.innerHTML = '';

  try {
    const profileData = await getProfileFromActiveTab();
    const audit = analyzer.aggregate(profileData);
    statusEl.textContent = 'Analysis complete.';
    renderResults(audit);
  } catch (error) {
    statusEl.textContent = 'Could not analyze this tab.';
    resultsEl.innerHTML = `<div class="error">${escapeHtml(error.message || error)}</div>`;
    console.error(error);
  } finally {
    analyzeBtn.disabled = false;
  }
});

boot();
