function cleanText(value) {
  return String(value || '').replace(/\s+/g, ' ').trim();
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

function extractImages() {
  const imageElements = Array.from(document.images || []);
  const metaImage = document.querySelector('meta[property="og:image"], meta[name="twitter:image"]')?.content;
  const images = imageElements
    .map((img) => ({
      src: img.currentSrc || img.src || '',
      alt: cleanText(img.alt || ''),
      className: cleanText(img.className || ''),
      width: Number(img.naturalWidth || img.width || 0),
      height: Number(img.naturalHeight || img.height || 0)
    }))
    .filter((img) => img.src)
    .sort((a, b) => (b.width * b.height) - (a.width * a.height));

  if (metaImage && !images.some((img) => img.src === metaImage)) {
    images.unshift({ src: metaImage, alt: 'open graph image', className: 'meta', width: 0, height: 0 });
  }
  return images.slice(0, 8);
}

function extractProfileData() {
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
  const profileText = prioritySelectors.flatMap(visibleTextOf).join(' ');
  const fallbackText = cleanText(document.body?.innerText || '');
  const text = cleanText(profileText || fallbackText).slice(0, 6000);
  const fullText = cleanText(`${document.title || ''} ${fallbackText}`);

  const tabular = {
    age: extractAge(fullText),
    height: 170,
    income: -1,
    drinks: inferChoice(fullText, {
      'very often': [/drinks?\s+very often/, /heavy drink/],
      often: [/drinks?\s+often/, /part(y|ies) often/],
      socially: [/drinks?\s+socially/, /social drink/],
      rarely: [/drinks?\s+rarely/],
      'not at all': [/does not drink/, /no alcohol/, /sober/]
    }, 'socially'),
    drugs: inferChoice(fullText, {
      often: [/drugs?\s+often/, /420 friendly.*often/],
      sometimes: [/drugs?\s+sometimes/, /420 friendly/, /weed/, /cannabis/],
      never: [/no drugs/, /drug free/, /never.*drugs/]
    }, 'never'),
    smokes: inferChoice(fullText, {
      yes: [/smokes?\s+yes/, /smoker/],
      sometimes: [/smokes?\s+sometimes/, /trying to quit/],
      no: [/non smoker/, /does not smoke/, /no smoking/]
    }, 'no'),
    status: inferChoice(fullText, {
      married: [/married/],
      'seeing someone': [/seeing someone/, /in a relationship/],
      single: [/single/]
    }, 'single'),
    sex: inferChoice(fullText, { f: [/\bwoman\b/, /\bfemale\b/], m: [/\bman\b/, /\bmale\b/] }, 'm'),
    orientation: inferChoice(fullText, {
      bisexual: [/bisexual/, /\bbi\b/],
      gay: [/\bgay\b/, /lesbian/],
      straight: [/straight/]
    }, 'straight'),
    body_type: inferChoice(fullText, {
      athletic: [/athletic/],
      fit: [/\bfit\b/],
      thin: [/\bthin\b/],
      curvy: [/curvy/],
      average: [/average/]
    }, 'average'),
    diet: inferChoice(fullText, {
      vegan: [/vegan/],
      vegetarian: [/vegetarian/],
      halal: [/halal/],
      kosher: [/kosher/],
      'mostly anything': [/anything/, /foodie/]
    }, 'mostly anything'),
    education: inferChoice(fullText, {
      'graduated from masters program': [/masters/, /master's/],
      'graduated from college/university': [/college/, /university/, /degree/],
      'graduated from high school': [/high school/]
    }, 'graduated from college/university'),
    ethnicity: firstMatch(/\b(asian|white|black|hispanic|latino|middle eastern|indian|pacific islander)\b/i, fullText, 'not specified'),
    job: firstMatch(/\b(?:job|work|works as|career)[:\s]+([a-zA-Z /-]{2,40})/i, fullText, 'other'),
    offspring: inferChoice(fullText, {
      'has kids': [/has kids/, /has children/, /\bdad\b/, /\bmom\b/],
      "doesn't want kids": [/doesn't want kids/, /does not want kids/, /no kids/],
      "doesn't have kids": [/no children/, /no kids yet/]
    }, "doesn't have kids"),
    pets: inferChoice(fullText, {
      'likes dogs and likes cats': [/dogs?.*cats?|cats?.*dogs?/],
      'likes dogs': [/dog/],
      'likes cats': [/cat/]
    }, 'likes dogs'),
    religion: firstMatch(/\b(agnostic|atheist|christian|muslim|islam|hindu|buddhist|jewish|catholic)\b/i, fullText, 'agnosticism'),
    sign: firstMatch(/\b(aries|taurus|gemini|cancer|leo|virgo|libra|scorpio|sagittarius|capricorn|aquarius|pisces)\b/i, fullText, 'leo')
  };

  const expectedFields = Object.keys(tabular);
  const missingFieldCount = expectedFields.filter((field) => tabular[field] === 'not specified' || tabular[field] === '').length;

  return {
    text,
    tabular: { ...tabular, missingFieldCount },
    images: extractImages(),
    title: document.title || '',
    url: location.href
  };
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action !== 'GET_PROFILE_DATA') return false;
  sendResponse(extractProfileData());
  return true;
});
