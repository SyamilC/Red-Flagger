import { ProfileAnalyzer } from './inference.js';

const analyzer = new ProfileAnalyzer();
const statusEl = document.getElementById('status');
const resultsEl = document.getElementById('results');
const analyzeBtn = document.getElementById('analyzeBtn');
const moduleEl = document.getElementById('moduleStatus');
const manualBioEl = document.getElementById('manualBio');
const auditModeInputs = Array.from(document.querySelectorAll('input[name="auditMode"]'));

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

function currentMode() {
  return auditModeInputs.find((input) => input.checked)?.value || 'bio_only';
}

function extractProfileDataInPage() {
  const localCleanText = (value) => String(value || '').replace(/\s+/g, ' ').trim();
  const selectedText = localCleanText(window.getSelection?.().toString() || '');
  const isVisible = (element) => {
    try {
      const style = window.getComputedStyle(element);
      const rect = element.getBoundingClientRect();
      return style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 0 && rect.height > 0;
    } catch {
      return false;
    }
  };
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
  const scoreBioElement = (element) => {
    const text = localCleanText(element.innerText || element.textContent || '');
    if (text.length < 35 || text.length > 1800) return null;
    const marker = `${element.className || ''} ${element.id || ''} ${element.getAttribute('data-test') || ''} ${element.getAttribute('data-testid') || ''}`;
    let score = Math.min(120, text.length / 8);
    if (/bio|about|profile|essay|summary/i.test(marker)) score += 90;
    if (/nav|menu|footer|header|cookie|modal|comment|article-body/i.test(marker)) score -= 80;
    if (text.split(/\s+/).length < 8) score -= 35;
    return { element, text, score };
  };
  const findBestBioCandidate = () => {
    const selectors = [
      '[data-test*="bio" i]',
      '[data-testid*="bio" i]',
      '[class*="bio" i]',
      '[class*="about" i]',
      '[id*="bio" i]',
      '[id*="about" i]',
      '[class*="profile" i]',
      'article',
      'section',
      'main'
    ];
    const seen = new Set();
    const candidates = [];
    for (const selector of selectors) {
      for (const element of document.querySelectorAll(selector)) {
        if (seen.has(element) || !isVisible(element)) continue;
        seen.add(element);
        const candidate = scoreBioElement(element);
        if (candidate) candidates.push(candidate);
      }
    }
    return candidates.sort((a, b) => b.score - a.score)[0] || null;
  };
  const localExtractAge = (text) => {
    const direct = localFirstMatch(/\b(?:age[:\s]*)?([1-9][0-9])\b/i, text, '');
    const age = Number(direct);
    return age >= 18 && age <= 90 ? age : 27;
  };
  const bestBio = findBestBioCandidate();
  const root = bestBio?.element?.closest('article, section, main, [class*="profile" i], [class*="card" i], [class*="user" i]') || document.body;
  const fallbackText = localCleanText(document.body?.innerText || '');
  const scopedText = localCleanText(root?.innerText || '');
  const text = localCleanText(selectedText.length >= 20 ? selectedText : (bestBio?.text || '')).slice(0, 2400);
  const fullText = localCleanText(`${document.title || ''} ${scopedText || fallbackText}`);
  const scopedImages = root ? Array.from(root.querySelectorAll('img')) : [];
  const imageElements = scopedImages.length ? scopedImages : Array.from(document.images || []);
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
    extraction_mode: selectedText.length >= 20 ? 'selected_text' : (bestBio ? 'best_profile_block' : 'no_clean_profile_text'),
    extraction_note: text
      ? 'Page Audit used selected text or the best profile-like block, then read metadata and images from the nearest profile container.'
      : 'No clean profile text was found. Select a bio paragraph on the page or paste text into the popup.',
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

function profileFromManualInput(mode = 'bio_only') {
  const text = cleanText(manualBioEl?.value || '');
  if (!text) return null;
  return {
    text,
    analysis_mode: mode,
    extraction_mode: mode === 'bio_only' ? 'bio_only_manual_text' : 'manual_popup_text',
    extraction_note: mode === 'bio_only'
      ? 'Bio Only mode used only the text pasted into the popup. Metadata and image checks were disabled.'
      : 'The audit used text pasted directly into the extension popup.',
    tabular: {
      age: 27,
      height: 170,
      income: -1,
      drinks: 'socially',
      drugs: 'never',
      smokes: 'no',
      status: 'single',
      sex: 'm',
      orientation: 'straight',
      body_type: 'average',
      diet: 'mostly anything',
      education: 'graduated from college/university',
      ethnicity: 'not specified',
      job: 'other',
      offspring: "doesn't have kids",
      pets: 'likes dogs',
      religion: 'agnosticism',
      sign: 'leo',
      missingFieldCount: 1
    },
    images: [],
    title: 'Manual popup input',
    url: 'extension://manual-input'
  };
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
      <p><b>Audit:</b> ${escapeHtml(profile.audit_mode || 'unknown')}</p>
      <p><b>Mode:</b> ${escapeHtml(profile.extraction_mode || 'unknown')}</p>
      <p class="muted">${escapeHtml(profile.extraction_note || '')}</p>
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
    analyzeBtn.disabled = false;
    updateModeUi();
  } catch (error) {
    statusEl.textContent = 'Model load failed. Check extension/models/nlp_model.json.';
    moduleEl.innerHTML = `<div class="module-pill high">${escapeHtml(error.message)}</div>`;
    console.error(error);
  }
}

analyzeBtn.addEventListener('click', async () => {
  analyzeBtn.disabled = true;
  const mode = currentMode();
  statusEl.textContent = mode === 'bio_only'
    ? 'Running pasted-bio audit...'
    : 'Reading page DOM and running full page audit...';
  resultsEl.innerHTML = '';

  try {
    const profileData = mode === 'bio_only'
      ? profileFromManualInput(mode)
      : await getProfileFromActiveTab();
    if (!profileData) {
      throw new Error(mode === 'bio_only'
        ? 'Paste a bio into the Bio text box before running Bio Only mode.'
        : 'No profile data was extracted from this page.');
    }
    if (!cleanText(profileData.text)) {
      throw new Error(profileData.extraction_note || 'No clean profile text found. Select a bio paragraph or paste a bio into the popup.');
    }
    const audit = analyzer.aggregate(profileData, { mode });
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

function updateModeUi() {
  const mode = currentMode();
  analyzeBtn.textContent = mode === 'bio_only' ? 'Run Bio Audit' : 'Run Page Audit';
  manualBioEl.placeholder = mode === 'bio_only'
    ? 'Paste one dating bio here. This mode ignores page metadata and images.'
    : 'Optional notes only. Page Audit reads bio, metadata, and image from the current page.';
  statusEl.textContent = mode === 'bio_only'
    ? 'Bio Only mode: paste a bio, then run the audit.'
    : 'Page Audit mode: open a profile-like page, then run the audit.';
}

auditModeInputs.forEach((input) => input.addEventListener('change', updateModeUi));
boot();
