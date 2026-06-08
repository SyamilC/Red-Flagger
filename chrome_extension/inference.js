import * as tf from '@tensorflow/tfjs';

const KEYWORD_RULES = {
  aggressive_tone: [
    /don't message me/i,
    /do not message me/i,
    /don't waste my time/i,
    /swipe left/i,
    /you better/i,
    /prove me wrong/i,
    /idiot|stupid|get lost|shut up/i
  ],
  hookup_focus: [
    /hook\s?up/i,
    /one[-\s]?night/i,
    /friends with benefits/i,
    /\bfwb\b/i,
    /nothing serious/i,
    /no strings attached/i,
    /\bdtf\b/i,
    /situationship/i
  ],
  negativity: [
    /tired of/i,
    /sick of/i,
    /people suck/i,
    /i hate/i,
    /waste of time/i,
    /everyone is fake/i,
    /dating is trash/i,
    /i don't trust people/i
  ],
  sarcasm_cynicism: [
    /sarcasm/i,
    /sarcastic/i,
    /fluent in sarcasm/i,
    /dark humor/i,
    /cynical/i,
    /dead inside/i
  ],
  controlling_behavior: [
    /always right/i,
    /must obey/i,
    /my rules/i,
    /i decide/i,
    /do what i say/i,
    /need permission/i
  ],
  emotional_manipulation: [
    /silent treatment/i,
    /test you/i,
    /prove you care/i,
    /make you jealous/i,
    /guilt trip/i
  ],
  entitlement_superiority: [
    /impress me/i,
    /high standards/i,
    /princess treatment/i,
    /alpha/i,
    /i deserve/i,
    /don't settle/i
  ],
  poor_conflict_resolution: [
    /cut people off/i,
    /block you/i,
    /revenge/i,
    /never apologize/i,
    /hold grudges/i
  ],
  main_character_syndrome: [
    /main character/i,
    /center of attention/i,
    /obsessed with me/i,
    /treat me like/i
  ],
  dismissive: [
    /too sensitive/i,
    /not my problem/i,
    /cry about it/i,
    /get over it/i,
    /feelings are/i
  ]
};

const LABEL_DESCRIPTIONS = {
  aggressive_tone: 'Harsh, hostile, demanding, or insulting wording.',
  hookup_focus: 'Strong focus on casual hookups or non-committal intent.',
  negativity: 'Pessimistic, bitter, cynical, or emotionally negative statements.',
  sarcasm_cynicism: 'Sarcastic, dismissive, or cynical tone.',
  substance_risk: 'Heavy drinking, smoking, drugs, or party-heavy lifestyle.',
  incomplete_profile: 'Low-effort profile with very little useful information.',
  controlling_behavior: 'Language suggesting control, obedience, punishment, or dominance over a partner.',
  emotional_manipulation: 'Silent treatment, guilt, emotional punishment, testing, or conditional affection.',
  entitlement_superiority: 'Self-important, superior, dismissive, or impress-me framing.',
  poor_conflict_resolution: 'Cutting people off, refusing communication, revenge, blame, or inability to resolve conflict maturely.',
  main_character_syndrome: 'Self-centred profile framing that expects special attention.',
  dismissive: 'Invalidating, belittling, or contemptuous wording toward feelings, interests, or boundaries.'
};

function clamp(value, min = 0, max = 1) {
  return Math.max(min, Math.min(max, Number(value) || 0));
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function tokenize(text) {
  return String(text || '').toLowerCase().match(/[a-z0-9']+/g) || [];
}

function scoreToRisk(score) {
  if (score >= 70) return 'High';
  if (score >= 40) return 'Moderate';
  if (score >= 20) return 'Low-to-Moderate';
  return 'Low';
}

function emptyLabelScores(labels) {
  const scores = {};
  labels.forEach((label) => {
    scores[label] = {
      probability: 0,
      predicted: 0,
      sources: [],
      evidence: [],
      description: LABEL_DESCRIPTIONS[label] || ''
    };
  });
  return scores;
}

export class ProfileAnalyzer {
  constructor() {
    this.nlpConfig = null;
    this.labels = [];
    this.status = {
      tfjsReady: false,
      nlpReady: false,
      tabularMode: 'rules',
      visualMode: 'dom_heuristic',
      notes: []
    };
  }

  async loadModels() {
    await tf.ready();
    this.status.tfjsReady = true;

    const nlpUrl = chrome.runtime.getURL('models/nlp_model.json');
    const response = await fetch(nlpUrl);
    if (!response.ok) {
      throw new Error(`Missing NLP model metadata at ${nlpUrl}`);
    }
    this.nlpConfig = await response.json();
    this.labels = this.nlpConfig.labels || Object.keys(LABEL_DESCRIPTIONS);
    this.status.nlpReady = true;
    this.status.notes.push('TF-IDF logistic regression metadata loaded from models/nlp_model.json.');
    this.status.notes.push('Tabular and visual modules run as browser-safe client-side heuristics unless TFJS assets are exported later.');
  }

  getStatus() {
    return { ...this.status };
  }

  analyzeNLP(text) {
    const labels = this.labels;
    const tokens = tokenize(text);
    const docLength = tokens.length || 1;
    const counts = new Map();
    tokens.forEach((token) => counts.set(token, (counts.get(token) || 0) + 1));

    const scores = {};
    labels.forEach((label, labelIndex) => {
      let logit = Number(this.nlpConfig.intercepts?.[labelIndex] || 0);
      const values = [];
      const weights = [];

      for (const [word, count] of counts.entries()) {
        const featureIndex = this.nlpConfig.vocab?.[word];
        if (featureIndex === undefined) continue;
        values.push((count / docLength) * Number(this.nlpConfig.idf?.[featureIndex] || 1));
        weights.push(Number(this.nlpConfig.coefs?.[labelIndex]?.[featureIndex] || 0));
      }

      if (values.length) {
        logit += tf.tidy(() => {
          const featureTensor = tf.tensor1d(values);
          const weightTensor = tf.tensor1d(weights);
          return featureTensor.mul(weightTensor).sum().arraySync();
        });
      }

      const probability = clamp(sigmoid(logit));
      const threshold = Number(this.nlpConfig.thresholds?.[label] ?? 0.5);
      scores[label] = {
        probability,
        threshold,
        predicted: probability >= threshold ? 1 : 0,
        source: 'tfjs_tfidf_logreg',
        description: this.nlpConfig.descriptions?.[label] || LABEL_DESCRIPTIONS[label] || ''
      };
    });

    return {
      available: true,
      module: 'TensorFlow.js TF-IDF logistic regression scorer',
      detected_flags: Object.entries(scores).filter(([, info]) => info.predicted).map(([label]) => label),
      label_scores: scores
    };
  }

  analyzeKeywordRules(text) {
    const scores = emptyLabelScores(this.labels);
    for (const [label, rules] of Object.entries(KEYWORD_RULES)) {
      if (!scores[label]) continue;
      const matched = rules.filter((rule) => rule.test(String(text || ''))).map((rule) => rule.source);
      if (matched.length) {
        scores[label] = {
          ...scores[label],
          probability: clamp(0.62 + matched.length * 0.08),
          threshold: 0.55,
          predicted: 1,
          source: 'keyword_rules',
          evidence: matched.slice(0, 4)
        };
      }
    }

    return {
      available: true,
      module: 'Transparent browser keyword rules',
      detected_flags: Object.entries(scores).filter(([, info]) => info.predicted).map(([label]) => label),
      label_scores: scores
    };
  }

  analyzeTabular(profileData) {
    const data = profileData || {};
    const scores = emptyLabelScores(this.labels);
    const textLength = Number(data.bioLength || 0);
    const missingCount = Number(data.missingFieldCount || 0);
    const drinks = String(data.drinks || '').toLowerCase();
    const drugs = String(data.drugs || '').toLowerCase();
    const smokes = String(data.smokes || '').toLowerCase();
    const status = String(data.status || '').toLowerCase();

    const substanceSignals = [
      /often|very often|desperately/.test(drinks),
      /sometimes|often/.test(drugs),
      /yes|sometimes|trying to quit/.test(smokes)
    ].filter(Boolean).length;

    if (substanceSignals) {
      scores.substance_risk = {
        ...scores.substance_risk,
        probability: clamp(0.55 + substanceSignals * 0.12),
        threshold: 0.5,
        predicted: 1,
        source: 'tabular_rules',
        evidence: ['drinks/drugs/smokes metadata']
      };
    }

    if (textLength > 0 && textLength < 20) {
      scores.incomplete_profile = {
        ...scores.incomplete_profile,
        probability: 0.72,
        threshold: 0.5,
        predicted: 1,
        source: 'tabular_rules',
        evidence: [`bio length ${textLength} words`]
      };
    } else if (missingCount >= 8) {
      scores.incomplete_profile = {
        ...scores.incomplete_profile,
        probability: 0.58,
        threshold: 0.5,
        predicted: 1,
        source: 'tabular_rules',
        evidence: [`${missingCount} metadata fields unavailable`]
      };
    }

    if (/married|seeing someone/.test(status)) {
      scores.hookup_focus = {
        ...scores.hookup_focus,
        probability: 0.56,
        threshold: 0.5,
        predicted: 1,
        source: 'tabular_rules',
        evidence: [`status: ${status}`]
      };
    }

    return {
      available: true,
      module: 'Client-side tabular metadata scorer',
      mode: 'heuristic_json_scorer',
      metadata_used: data,
      detected_flags: Object.entries(scores).filter(([, info]) => info.predicted).map(([label]) => label),
      label_scores: scores
    };
  }

  analyzeVisual(profileData) {
    const images = Array.isArray(profileData?.images) ? profileData.images : [];
    const visibleImages = images.filter((image) => Number(image.width || 0) >= 80 && Number(image.height || 0) >= 80);
    const flags = [];
    const evidence = [];

    if (!visibleImages.length) {
      flags.push('No_Face_Present');
      evidence.push('No large profile image was available from the DOM.');
    }

    const groupHint = visibleImages.some((image) => /group|friends|party|crew|team/i.test(`${image.alt || ''} ${image.className || ''} ${image.src || ''}`));
    if (groupHint) {
      flags.push('Group_Photo_Ambiguity');
      evidence.push('Image metadata suggests a group/party photo.');
    }

    const tinyOrAvatarOnly = images.length > 0 && visibleImages.length === 0;
    if (tinyOrAvatarOnly) {
      flags.push('Face_Obscured');
      evidence.push('Only small avatar-sized images were found.');
    }

    const detected = flags.length ? [...new Set(flags)] : ['clean'];
    const visualRisk = detected.includes('No_Face_Present') ? 0.55
      : detected.includes('Face_Obscured') ? 0.5
        : detected.includes('Group_Photo_Ambiguity') ? 0.35
          : 0;

    return {
      available: true,
      module: 'Browser DOM visual presentation scorer',
      mode: 'dom_heuristic_yolo_fallback',
      detected_visual_flags: detected,
      visual_risk_score: visualRisk,
      images_checked: images.slice(0, 4),
      evidence,
      note: 'The extension is fully client-side. Real YOLO TFJS inference can replace this fallback when models/yolo_model_tfjs/model.json is exported.'
    };
  }

  aggregate(profileData) {
    const text = profileData?.text || '';
    const nlp = this.analyzeNLP(text);
    const keyword = this.analyzeKeywordRules(text);
    const tabular = this.analyzeTabular({ ...(profileData?.tabular || {}), bioLength: tokenize(text).length });
    const visual = this.analyzeVisual(profileData);
    const combined = emptyLabelScores(this.labels);

    [nlp, keyword, tabular].forEach((moduleResult) => {
      for (const [label, info] of Object.entries(moduleResult.label_scores || {})) {
        if (!combined[label]) continue;
        if (Number(info.probability || 0) > combined[label].probability) {
          combined[label] = {
            ...combined[label],
            probability: Number(info.probability || 0),
            threshold: Number(info.threshold || 0.5),
            predicted: Number(info.predicted || 0),
            source: info.source || moduleResult.module,
            evidence: info.evidence || [],
            description: info.description || combined[label].description
          };
        }
      }
    });

    const detectedTextFlags = Object.entries(combined)
      .filter(([, info]) => info.predicted || info.probability >= 0.5)
      .sort((a, b) => b[1].probability - a[1].probability)
      .map(([label]) => label);
    const topScores = Object.values(combined).map((info) => info.probability).sort((a, b) => b - a).slice(0, 3);
    const textRisk = topScores.length ? topScores.reduce((sum, value) => sum + value, 0) / topScores.length : 0;
    const visualRisk = Number(visual.visual_risk_score || 0);
    const overallRiskScore = Math.round(clamp((0.78 * textRisk) + (0.22 * visualRisk)) * 1000) / 10;

    return {
      profile_summary: {
        source_url: profileData?.url || '',
        title: profileData?.title || '',
        bio_word_count: tokenize(text).length,
        images_found: profileData?.images?.length || 0
      },
      overall_risk_score: overallRiskScore,
      overall_risk_level: scoreToRisk(overallRiskScore),
      detected_text_tabular_flags: detectedTextFlags,
      detected_visual_flags: visual.detected_visual_flags,
      combined_label_scores: combined,
      module_outputs: {
        nlp_tfidf_tfjs: nlp,
        keyword_rules: keyword,
        tabular_metadata: tabular,
        visual_dom: visual
      },
      runtime: this.getStatus(),
      bio_excerpt: String(text || '').slice(0, 420)
    };
  }
}
