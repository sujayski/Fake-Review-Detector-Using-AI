
/* ============================================================
   SAMPLE REVIEWS
   ============================================================ */
const FAKE_REVIEWS = [
  "AMAZING!!! BEST PRODUCT EVER!!! BUY NOW DONT MISS!!!",
  "Perfect. Love it. No complaints. 5 stars!",
  "LIFE CHANGING!!! MUST BUY!!! HIGHLY RECOMMEND!!!",
  "Best purchase ever. Zero issues. Flawless. Worth every penny.",
  "Amazing quality. Perfect in every way!"
];

const REAL_REVIEWS = [
  "Bought 3 weeks ago. Battery drains faster than expected. Camera decent but not sharp. Build quality solid though.",
  "Decent product. However charging cable feels cheap. Would prefer longer warranty.",
  "After a month, works fine but size bigger than expected. Good value vs alternatives.",
  "Average product. Does what it says. Had packaging issue, support helped fast.",
  "Decent but seen better. Color different from image. Works as described though."
];

/* ============================================================
   KEYWORD LISTS
   ============================================================ */
const SPAM_WORDS    = ["amazing","best ever","must buy","buy now","perfect","life changing","zero issues","flawless","love it","no complaints","highly recommend","changed my life","worth every penny","dont miss","outstanding","fantastic","unbelievable","superb"];
const HONEST_WORDS  = ["however","but","although","issue","problem","could be better","average","decent","personally","tested","weeks later","one downside","compared","expected","disappointed","after using","not perfect","noticed","minor","drawback","unfortunately","lacks"];
const URGENCY_WORDS = ["limited time","order now","only today","act fast","selling out","last chance","hurry","must have"];
const SPECIFIC_WORDS= ["battery","camera","screen","size","color","price","paid","bought","model","delivery","packaging","weight"];

const IDF_FAKE = {"amazing":0.1,"perfect":0.1,"best":0.15,"love":0.2,"great":0.2,"excellent":0.15,"superb":0.1,"flawless":0.1,"outstanding":0.12,"recommended":0.18};
const IDF_REAL = {"however":2.1,"although":2.3,"issue":2.0,"problem":1.9,"but":1.5,"disappointed":2.4,"compared":2.1,"noticed":2.2,"unfortunately":2.5,"downside":2.6,"expected":1.8,"decent":1.7,"average":1.6};

const STOP_WORDS = new Set(["the","a","an","is","it","this","that","and","or","but","in","on","at","to","for","of","with","be"]);

const TRAINING_DATA = {
  fake: [
    "AMAZING best product ever must buy",
    "perfect love it no complaints five stars",
    "life changing highly recommend buy now",
    "outstanding quality flawless worth every penny",
    "excellent superb amazing fantastic wonderful",
    "best purchase ever zero issues perfect",
    "incredible product love everything about it",
    "amazing fast delivery perfect condition",
    "must buy dont miss outstanding quality",
    "best ever five stars highly recommended"
  ],
  real: [
    "decent product however battery drains fast",
    "works fine but size bigger than expected",
    "bought three weeks ago noticed screen dim",
    "average quality disappointed with packaging",
    "good value but charging slow compared others",
    "expected better quality for this price paid",
    "minor issue with button but support helped",
    "not perfect but does what it says overall",
    "one downside is weight heavier than shown",
    "after using one month some features limited"
  ]
};

const SPECIFIC_PATTERNS = [
  /\d+\s*(weeks?|months?|days?)/i,
  /[₹$€£]\s*[\d,]+/,
  /\d+(\.\d+)?%/,
  /(battery|camera|screen|display|processor)/i,
  /(shipped|delivered|arrived)/i,
  /(model|version|gen|series)/i,
  /(compared to|better than|worse than)/i,
  /(bought|purchased|ordered)/i
];

/* ============================================================
   NAIVE BAYES
   ============================================================ */
function trainNaiveBayes(data) {
  let fakeWords = {}, realWords = {}, fakeTotal = 0, realTotal = 0;
  data.fake.forEach(t => t.split(' ').forEach(w => { fakeWords[w]=(fakeWords[w]||0)+1; fakeTotal++; }));
  data.real.forEach(t => t.split(' ').forEach(w => { realWords[w]=(realWords[w]||0)+1; realTotal++; }));
  return { fakeWords, realWords, fakeTotal, realTotal };
}

function classifyNB(text, model) {
  const words = text.toLowerCase().split(/\s+/);
  let fakeScore = Math.log(0.5), realScore = Math.log(0.5);
  const V = 500;
  words.forEach(w => {
    fakeScore += Math.log(((model.fakeWords[w]||0)+1)/(model.fakeTotal+V));
    realScore += Math.log(((model.realWords[w]||0)+1)/(model.realTotal+V));
  });
  const fp = Math.exp(fakeScore), rp = Math.exp(realScore), tot = fp+rp;
  return { realProbability:(rp/tot)*100, fakeProbability:(fp/tot)*100 };
}

const nbModel = trainNaiveBayes(TRAINING_DATA);

/* ============================================================
   TENSORFLOW.JS MODEL
   ============================================================ */
let tfModel = null;
let tfReady  = false;

async function buildAndTrainModel() {
  if (typeof tf === 'undefined') return null;
  try {
    const TRAIN_X = tf.tensor2d([
      [5,4,0,0.4,3,0.8,0.0,2.1],[8,3,0,0.3,2,0.7,0.1,1.8],
      [6,5,0,0.5,4,0.9,0.0,2.5],[4,3,0,0.6,3,0.8,0.0,2.0],
      [7,4,1,0.2,2,0.6,0.1,1.5],
      [45,0,3,0.0,0,0.0,0.6,0.2],[38,1,2,0.0,0,0.1,0.5,0.1],
      [62,0,4,0.0,0,0.0,0.8,0.0],[29,0,2,0.0,1,0.0,0.4,0.3],
      [55,1,3,0.0,0,0.1,0.6,0.1]
    ]);
    const TRAIN_Y = tf.tensor2d([[1,0],[1,0],[1,0],[1,0],[1,0],[0,1],[0,1],[0,1],[0,1],[0,1]]);
    const model = tf.sequential();
    model.add(tf.layers.dense({units:16,activation:'relu',inputShape:[8]}));
    model.add(tf.layers.dropout({rate:0.2}));
    model.add(tf.layers.dense({units:8,activation:'relu'}));
    model.add(tf.layers.dense({units:2,activation:'softmax'}));
    model.compile({optimizer:tf.train.adam(0.01),loss:'categoricalCrossentropy',metrics:['accuracy']});
    await model.fit(TRAIN_X,TRAIN_Y,{epochs:100,verbose:0});
    TRAIN_X.dispose(); TRAIN_Y.dispose();
    return model;
  } catch(e) { return null; }
}

async function predictWithTF(features, model) {
  if (!model) return { fakePct:50, realPct:50 };
  try {
    const input = tf.tensor2d([features]);
    const pred  = model.predict(input);
    const vals  = await pred.data();
    input.dispose(); pred.dispose();
    return { fakePct:Math.round(vals[0]*100), realPct:Math.round(vals[1]*100) };
  } catch(e) { return { fakePct:50, realPct:50 }; }
}

// Train on load
buildAndTrainModel().then(m => {
  tfModel = m;
  tfReady  = true;
  const badge = document.getElementById('aiReadyBadge');
  if (badge) badge.classList.add('visible');
});

/* ============================================================
   SENTIMENT (built-in simple lexicon as fallback)
   ============================================================ */
const POS_WORDS = ["good","great","excellent","nice","happy","love","best","amazing","fantastic","wonderful","superb","perfect","outstanding","brilliant","awesome"];
const NEG_WORDS = ["bad","poor","terrible","awful","horrible","hate","worst","disappointed","disappointing","useless","broken","cheap","defective","slow","issue","problem","fault","wrong"];

function simpleSentiment(text) {
  const lower = text.toLowerCase();
  let score = 0, pos = [], neg = [];
  POS_WORDS.forEach(w => { if (lower.includes(w)) { score++; pos.push(w); } });
  NEG_WORDS.forEach(w => { if (lower.includes(w)) { score--; neg.push(w); } });
  const words = lower.split(/\s+/).filter(w=>w.length>0);
  const comparative = words.length ? score / words.length : 0;
  return { score, comparative, positive: pos, negative: neg };
}

function getSentimentObj(text) {
  if (typeof Sentiment !== 'undefined') {
    try { return new Sentiment().analyze(text); } catch(e) {}
  }
  return simpleSentiment(text);
}

/* ============================================================
   NLP ENGINE
   ============================================================ */
function preprocessText(text) {
  const lower    = text.toLowerCase().replace(/\s+/,' ').trim();
  const sentences= text.split(/[.!?]+/).map(s=>s.trim()).filter(s=>s.length>2);
  const allWords = lower.split(/\s+/).filter(w=>w.replace(/[^a-z]/g,'').length>0);
  const tokens   = allWords.filter(w=> !STOP_WORDS.has(w.replace(/[^a-z]/g,'')));
  return { lower, sentences, allWords, tokens };
}

function getTFIDFScore(text) {
  const { allWords } = preprocessText(text);
  const total = allWords.length || 1;
  const freq  = {};
  allWords.forEach(w => { const k=w.replace(/[^a-z]/g,''); freq[k]=(freq[k]||0)+1; });
  let fakeTFIDF=0, realTFIDF=0;
  Object.entries(freq).forEach(([w,cnt])=>{
    const tf=cnt/total;
    if (IDF_FAKE[w]) fakeTFIDF += tf*IDF_FAKE[w];
    if (IDF_REAL[w]) realTFIDF += tf*IDF_REAL[w];
  });
  const realScore = Math.min(100, Math.max(0, 50 + realTFIDF*200 - fakeTFIDF*300));
  return { fakeTFIDF, realTFIDF, realScore };
}

function getSentimentScore(text) {
  const result = getSentimentObj(text);
  const comp   = result.comparative;
  let score    = 50;
  if (comp > 1.5)  score -= 25;
  else if (comp < -0.5) score += 20;
  else if (Math.abs(comp) < 0.3) score += 10;
  if (result.positive.length > 5 && result.negative.length === 0) score -= 20;
  score = Math.min(100, Math.max(0, score));
  return { score, comparative: comp, positive: result.positive, negative: result.negative };
}

function getLinguisticScore(text) {
  const { lower, sentences, allWords } = preprocessText(text);
  let score = 50;
  const wordCount      = allWords.length;
  const sentenceCount  = Math.max(sentences.length, 1);
  const avgWPS         = wordCount / sentenceCount;
  if (avgWPS < 3) score -= 20;
  const allExclaim     = sentences.every(s => s.endsWith('!'));
  if (allExclaim && sentenceCount > 1) score -= 15;
  const uniqueWords    = new Set(allWords.map(w=>w.replace(/[^a-z]/g,'')));
  const TTR            = uniqueWords.size / Math.max(wordCount, 1);
  if (TTR < 0.4) score -= 15;
  else if (TTR > 0.7) score += 10;
  const capsWords   = allWords.filter(w => /^[A-Z]{2,}$/.test(w)).length;
  const capsRatio   = capsWords / Math.max(wordCount,1);
  if (capsRatio > 0.15) score -= 20;
  const exclamCount = (text.match(/!/g)||[]).length;
  const exclRatio   = exclamCount / sentenceCount;
  if (exclRatio > 0.8) score -= 15;
  // Length bonus
  if (wordCount < 10) score -= 30;
  else if (wordCount < 20) score -= 10;
  else if (wordCount >= 80) score += 10;
  if (wordCount > 150) score += 10;
  // Specificity
  const specMatches = SPECIFIC_PATTERNS.filter(p=>p.test(text)).length;
  score += specMatches * 5;
  // TTR vocabulary richness
  const adjRatio = 0; // simplified
  return {
    score: Math.min(100, Math.max(0, score)),
    TTR: +(TTR*100).toFixed(1),
    capsRatio: +(capsRatio*100).toFixed(1),
    exclamCount,
    wordCount,
    sentenceCount,
    specMatches,
    avgWPS: +avgWPS.toFixed(1)
  };
}

function getRuleBasedScore(text) {
  const lower = text.toLowerCase();
  let score   = 50;
  let spamCnt = 0, honestCnt = 0, urgencyCnt = 0;
  SPAM_WORDS.forEach(w    => { if (lower.includes(w)) { score -= 6; spamCnt++; } });
  HONEST_WORDS.forEach(w  => { if (lower.includes(w)) { score += 7; honestCnt++; } });
  URGENCY_WORDS.forEach(w => { if (lower.includes(w)) { score -= 12; urgencyCnt++; } });
  let foundSpecific = false;
  SPECIFIC_WORDS.forEach(w => { if (lower.includes(w)) foundSpecific = true; });
  if (foundSpecific) score += 10;
  const words  = text.split(/\s+/).filter(w=>w.trim().length>0);
  const wCount = words.length;
  if (wCount < 10) score -= 30;
  else if (wCount <= 30) score -= 10;
  else if (wCount >= 80 && wCount <= 150) score += 10;
  else if (wCount > 150) score += 20;
  const capsWords  = words.filter(w=>/^[A-Z]{2,}$/.test(w)).length;
  if (capsWords > 3) score -= 15;
  const exclamCnt  = (text.match(/!/g)||[]).length;
  if (exclamCnt > 2) score -= (exclamCnt - 2) * 5;
  const hasPos     = spamCnt > 0;
  const hasNeg     = honestCnt > 0;
  if (hasPos && hasNeg) score += 8;
  return {
    score:     Math.min(100, Math.max(0, score)),
    spamCnt, honestCnt, urgencyCnt,
    capsWords, exclamCnt, wordCount: wCount, foundSpecific
  };
}

function extractFeatures(text, ruleResult, tfidfResult, sentResult) {
  const r   = ruleResult   || getRuleBasedScore(text);
  const tfi = tfidfResult  || getTFIDFScore(text);
  const snt = sentResult   || getSentimentScore(text);
  const wc  = r.wordCount;
  const caps= r.capsWords / Math.max(wc, 1);
  return [
    Math.min(wc/100, 1)*100,
    r.spamCnt,
    r.honestCnt,
    caps,
    r.exclamCnt,
    tfi.fakeTFIDF,
    tfi.realTFIDF,
    Math.abs(snt.comparative)
  ];
}

function getVerdict(score) {
  if (score >= 70) return 'REAL';
  if (score >= 40) return 'SUSPICIOUS';
  return 'FAKE';
}

function getVerdictColor(verdict) {
  if (verdict === 'REAL') return 'var(--green)';
  if (verdict === 'SUSPICIOUS') return 'var(--orange)';
  return 'var(--red)';
}

/* ============================================================
   ENSEMBLE ANALYSIS
   ============================================================ */
async function runEnsemble(text) {
  const rule = getRuleBasedScore(text);
  const tfi  = getTFIDFScore(text);
  const nb   = classifyNB(text, nbModel);
  const sent = getSentimentScore(text);
  const ling = getLinguisticScore(text);
  const feats= extractFeatures(text, rule, tfi, sent);
  const tfr  = await predictWithTF(feats, tfModel);

  // If tfModel available, use full ensemble; else 4-model
  let final;
  if (tfReady && tfModel) {
    final = rule.score*0.25 + tfi.realScore*0.20 + nb.realProbability*0.20 + sent.score*0.10 + ling.score*0.15 + tfr.realPct*0.10;
  } else {
    final = rule.score*0.35 + tfi.realScore*0.25 + nb.realProbability*0.25 + sent.score*0.15;
  }
  final = Math.round(Math.min(100, Math.max(0, final)));

  return {
    finalScore: final,
    verdict:    getVerdict(final),
    breakdown: {
      ruleEngine:  Math.round(rule.score),
      tfidf:       Math.round(tfi.realScore),
      naiveBayes:  Math.round(nb.realProbability),
      sentiment:   Math.round(sent.score),
      linguistic:  Math.round(ling.score),
      neuralNet:   tfReady && tfModel ? tfr.realPct : null
    },
    details: { rule, tfi, nb, sent, ling, tfr }
  };
}

/* ============================================================
   CHART INSTANCES
   ============================================================ */
let signalChartInst  = null;
let qualityChartInst = null;
let reportBarInst    = null;

// Initialize default Chart colors based on theme
const initTheme = document.documentElement.getAttribute('data-theme') || 'dark';
Chart.defaults.color = initTheme === 'light' ? '#4b5563' : '#9ca3af';
Chart.defaults.borderColor = initTheme === 'light' ? 'rgba(0, 0, 0, 0.06)' : 'rgba(255,255,255,0.06)';

/* ============================================================
   THEME CONTROLLER (Day/Night Mode)
   ============================================================ */
function toggleTheme() {
  const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
  const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
  setTheme(newTheme);
}

function setTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('setting_theme', theme);

  const toggleBtn = document.getElementById('themeToggle');
  if (toggleBtn) {
    if (theme === 'light') {
      toggleBtn.innerHTML = '<i class="fa-solid fa-moon"></i>';
      toggleBtn.title = 'Switch to Dark Mode';
    } else {
      toggleBtn.innerHTML = '<i class="fa-solid fa-sun"></i>';
      toggleBtn.title = 'Switch to Light Mode';
    }
  }
  updateChartThemes(theme);
}

function updateChartThemes(theme) {
  const isLight = theme === 'light';
  const textColor = isLight ? '#4b5563' : '#9ca3af';
  const borderColor = isLight ? 'rgba(0, 0, 0, 0.06)' : 'rgba(255,255,255,0.06)';
  const gridColor = isLight ? 'rgba(0, 0, 0, 0.04)' : 'rgba(255,255,255,0.04)';

  Chart.defaults.color = textColor;
  Chart.defaults.borderColor = borderColor;

  if (signalChartInst) {
    signalChartInst.update();
  }

  if (qualityChartInst) {
    const emptyColor = isLight ? 'rgba(0, 0, 0, 0.06)' : 'rgba(255,255,255,0.06)';
    qualityChartInst.data.datasets[0].backgroundColor[1] = emptyColor;
    qualityChartInst.update();
  }

  if (reportBarInst) {
    if (reportBarInst.options.scales) {
      if (reportBarInst.options.scales.x && reportBarInst.options.scales.x.grid) {
        reportBarInst.options.scales.x.grid.color = gridColor;
      }
      if (reportBarInst.options.scales.y && reportBarInst.options.scales.y.grid) {
        reportBarInst.options.scales.y.grid.color = gridColor;
      }
    }
    reportBarInst.update();
  }
}

function initDashboardCharts(score = 71) {
  // Signal Breakdown Donut
  const sCtx = document.getElementById('signalChart');
  if (sCtx) {
    if (signalChartInst) { signalChartInst.destroy(); signalChartInst=null; }
    signalChartInst = new Chart(sCtx, {
      type: 'doughnut',
      data: {
        labels:   ['Spam','Urgency','CAPS','Honest','Specific'],
        datasets: [{
          data:            [0.001, 0.001, 0.001, 17.6, 82.4],
          backgroundColor:['#ef4444','#f59e0b','#eab308','#10b981','#6366f1'],
          borderWidth:     0,
          hoverOffset:     6
        }]
      },
      options: {
        cutout:  '65%',
        plugins: { legend:{ display:false } },
        animation:{ animateScale:true, animateRotate:true }
      }
    });
  }

  // Quality Donut
  const qCtx = document.getElementById('qualityChart');
  if (qCtx) {
    if (qualityChartInst) { qualityChartInst.destroy(); qualityChartInst=null; }
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
    const emptyColor = currentTheme === 'light' ? 'rgba(0, 0, 0, 0.06)' : 'rgba(255,255,255,0.06)';
    qualityChartInst = new Chart(qCtx, {
      type: 'doughnut',
      data: {
        datasets: [{
          data: [score, 100-score],
          backgroundColor: ['#10b981', emptyColor],
          borderWidth: 0
        }]
      },
      options: {
        cutout: '72%',
        plugins:{ legend:{ display:false } },
        animation:{ animateScale:true, animateRotate:true }
      }
    });
  }
  document.getElementById('dashQualityScore').textContent = score;
}

function initReportChart() {
  const ctx = document.getElementById('reportBarChart');
  if (!ctx) return;
  if (reportBarInst) { reportBarInst.destroy(); reportBarInst=null; }

  const history = getHistory();
  const last7   = getLast7DaysData(history);
  const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
  const gridColor = currentTheme === 'light' ? 'rgba(0, 0, 0, 0.04)' : 'rgba(255,255,255,0.04)';

  reportBarInst = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: last7.labels,
      datasets: [{
        label: 'Analyses',
        data:  last7.counts,
        backgroundColor: 'rgba(99,102,241,0.5)',
        borderColor:     '#6366f1',
        borderWidth:     2,
        borderRadius:    6
      }]
    },
    options: {
      responsive: true,
      plugins:{ legend:{ display:false } },
      scales: {
        x: { grid:{ color: gridColor } },
        y: { grid:{ color: gridColor }, beginAtZero:true, ticks:{ stepSize:1 } }
      }
    }
  });
}

/* ============================================================
   PAGE NAVIGATION
   ============================================================ */
const PAGE_TITLES = {
  dashboard: ['Dashboard','Real-time AI analysis of reviews and fraud detection'],
  analyze:   ['Analyze Review','Run ML + NLP ensemble on any product review'],
  history:   ['History','All your previous review analyses'],
  reports:   ['Reports','Aggregated data and downloadable CSV reports'],
  settings:  ['Settings','API keys and detection configuration']
};

function showPage(name) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));

  const page = document.getElementById('page-'+name);
  const nav  = document.getElementById('nav-'+name);
  if (page) page.classList.add('active');
  if (nav)  nav.classList.add('active');

  const [title, sub] = PAGE_TITLES[name] || ['',''];
  document.getElementById('headerTitle').textContent = title;
  document.getElementById('headerSub').textContent   = sub;

  // Page-specific init
  if (name === 'dashboard') initDashboardCharts();
  if (name === 'history')   renderHistory();
  if (name === 'reports') {
    updateReportStats();
    setTimeout(initReportChart, 50);
  }
  if (name === 'settings') loadSettings();

  closeSidebar();
}

/* ============================================================
   REVIEW ANALYSIS
   ============================================================ */
function onReviewInput() {
  const val = document.getElementById('reviewInput').value;
  const auto= localStorage.getItem('setting_autoAnalyze') === 'true';
  if (auto && val.trim().length > 10) runAnalysis();
}

async function runAnalysis() {
  const text = document.getElementById('reviewInput').value.trim();
  if (!text) { showToast('Please paste a review first.','error'); return; }

  const btn = document.getElementById('analyzeBtn');
  btn.disabled = true;
  btn.textContent = 'Analyzing...';

  // Show loading steps
  const stepsDiv = document.getElementById('loadingSteps');
  stepsDiv.style.display = 'flex';
  const steps = document.querySelectorAll('.loading-step');
  steps.forEach(s => { s.className='loading-step'; s.querySelector('.step-icon').textContent='⟳'; });

  const showStep = (i, delay) => new Promise(res => setTimeout(() => {
    if (i > 0) {
      const prev = steps[i-1];
      prev.classList.add('done');
      prev.querySelector('.step-icon').textContent = '✓';
    }
    if (steps[i]) steps[i].classList.add('active');
    res();
  }, delay));

  await showStep(0, 0);
  await showStep(1, 200);
  await showStep(2, 350);
  await showStep(3, 500);
  await showStep(4, 650);
  await showStep(5, 800);
  const result = await runEnsemble(text);
  await new Promise(res => setTimeout(res, 200));
  await showStep(6, 0);
  await new Promise(res => setTimeout(res, 250));

  // Mark all done
  steps.forEach(s => { s.classList.remove('active'); s.classList.add('done'); s.querySelector('.step-icon').textContent='✓'; });

  stepsDiv.style.display = 'none';
  btn.disabled = false;
  btn.textContent = 'Analyze Review →';

  renderResults(text, result);
  updateDashboard(result, text);
  saveToHistory(text, result);
}

function renderResults(text, result) {
  const { finalScore, verdict, breakdown, details } = result;
  const col = document.getElementById('resultsCol');

  const verdictClass = verdict.toLowerCase();
  const verdictColor = getVerdictColor(verdict);
  const barColor     = verdict === 'REAL' ? 'var(--green)' : verdict === 'SUSPICIOUS' ? 'var(--orange)' : 'var(--red)';

  // Highlight text
  const showHighlights = localStorage.getItem('setting_showHighlights') !== 'false';
  let highlightedText  = escHtml(text);
  if (showHighlights) {
    SPAM_WORDS.forEach(w => {
      highlightedText = highlightedText.replace(new RegExp('\\b'+w+'\\b','gi'),
        m => `<span class="highlight-fake">${m}</span>`);
    });
    HONEST_WORDS.forEach(w => {
      highlightedText = highlightedText.replace(new RegExp('\\b'+w+'\\b','gi'),
        m => `<span class="highlight-real">${m}</span>`);
    });
  }

  // Reasons
  const reasons = buildReasons(details, verdict);
  const reasonsHTML = reasons.map(r => `
    <div class="reason-item">
      <span class="reason-icon" style="color:${r.color}"><i class="fa-solid fa-${r.icon}"></i></span>
      <span>${r.text}</span>
    </div>`).join('');

  // ML breakdown rows
  const breakdown_items = [
    { label:'Rule Engine',   val: breakdown.ruleEngine },
    { label:'TF-IDF',        val: breakdown.tfidf },
    { label:'Naive Bayes',   val: breakdown.naiveBayes },
    { label:'Sentiment',     val: breakdown.sentiment },
    { label:'Linguistic',    val: breakdown.linguistic },
    ...(breakdown.neuralNet != null ? [{label:'Neural Network', val: breakdown.neuralNet}] : [])
  ];

  const mlRowsHTML = breakdown_items.map(item => {
    const c = item.val >= 70 ? 'var(--green)' : item.val >= 40 ? 'var(--orange)' : 'var(--red)';
    return `<div class="ml-row">
      <span class="ml-row-label">${item.label}</span>
      <div class="ml-bar-bg">
        <div class="ml-bar-fill" style="width:0%;background:${c}" data-width="${item.val}%"></div>
      </div>
      <span class="ml-row-pct">${item.val}%</span>
    </div>`;
  }).join('');

  const ensembleColor = finalScore >= 70 ? 'var(--green)' : finalScore >= 40 ? 'var(--orange)' : 'var(--red)';

  // NLP Insights
  const ling   = details.ling;
  const sentObj= details.sent;
  const sentPol= sentObj.comparative > 0.5 ? 'Positive' : sentObj.comparative < -0.2 ? 'Negative' : 'Neutral';
  const complexity = ling.avgWPS < 5 ? 'Simple' : ling.avgWPS < 12 ? 'Moderate' : 'Complex';
  const specificity= ling.specMatches === 0 ? 'Low' : ling.specMatches < 3 ? 'Medium' : 'High';

  col.innerHTML = `
    <div class="card animate-in" style="margin-bottom:1rem;">
      <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1rem;flex-wrap:wrap;">
        <span class="verdict-badge ${verdictClass}">
          <i class="fa-solid fa-${verdict==='REAL'?'circle-check':verdict==='SUSPICIOUS'?'triangle-exclamation':'circle-xmark'}"></i>
          ${verdict}
        </span>
        <span style="font-size:1.8rem;font-weight:900;color:${verdictColor}">${finalScore}<span style="font-size:1rem;font-weight:500;color:var(--txt-muted)">/100</span></span>
      </div>
      <div class="score-bar-wrap">
        <div style="font-size:0.75rem;color:var(--txt-muted);margin-bottom:4px;">Authenticity Score</div>
        <div class="score-bar-bg">
          <div class="score-bar-fill" id="mainScoreBar" style="background:${barColor}"></div>
        </div>
      </div>
    </div>

    <div class="card animate-in" style="margin-bottom:1rem;animation-delay:0.1s">
      <div class="card-title" style="display:flex;align-items:center;gap:8px;">
        <span style="background:rgba(99,102,241,0.15);border-radius:6px;padding:4px 8px;font-size:0.8rem;color:var(--indigo)"><i class="fa-solid fa-brain"></i></span>
        ML Engine Breakdown
      </div>
      <div class="ml-breakdown" id="mlBreakdownRows">${mlRowsHTML}</div>
      <div class="ml-row ensemble-row">
        <span class="ml-row-label" style="font-weight:700;color:var(--txt-primary)">Ensemble Score</span>
        <div class="ml-bar-bg">
          <div class="ml-bar-fill" style="width:0%;background:${ensembleColor}" data-width="${finalScore}%"></div>
        </div>
        <span class="ml-row-pct" style="color:${ensembleColor}">${finalScore}%</span>
        <span class="verdict-badge ${verdictClass}" style="margin-left:8px;font-size:0.7rem;padding:2px 8px;">
          <i class="fa-solid fa-${verdict==='REAL'?'circle-check':verdict==='SUSPICIOUS'?'triangle-exclamation':'circle-xmark'}"></i>
          ${verdict}
        </span>
      </div>
    </div>

    <div class="card animate-in" style="margin-bottom:1rem;animation-delay:0.15s">
      <div class="card-title">🔬 NLP Insights</div>
      <div class="nlp-insights-grid">
        <div class="nlp-insight-item">
          <div class="nlp-insight-label">Vocabulary Richness (TTR)</div>
          <div class="nlp-insight-value" style="color:var(--cyan)">${ling.TTR}%</div>
        </div>
        <div class="nlp-insight-item">
          <div class="nlp-insight-label">Sentiment Polarity</div>
          <div class="nlp-insight-value" style="color:${sentPol==='Positive'?'var(--green)':sentPol==='Negative'?'var(--orange)':'var(--txt-secondary)'}">${sentPol}</div>
        </div>
        <div class="nlp-insight-item">
          <div class="nlp-insight-label">Sentence Complexity</div>
          <div class="nlp-insight-value">${complexity}</div>
        </div>
        <div class="nlp-insight-item">
          <div class="nlp-insight-label">Specificity Level</div>
          <div class="nlp-insight-value" style="color:${specificity==='High'?'var(--green)':specificity==='Medium'?'var(--orange)':'var(--red)'}">${specificity}</div>
        </div>
        <div class="nlp-insight-item">
          <div class="nlp-insight-label">CAPS Ratio</div>
          <div class="nlp-insight-value" style="color:${ling.capsRatio>15?'var(--red)':'var(--txt-secondary)'}">${ling.capsRatio}%</div>
        </div>
        <div class="nlp-insight-item">
          <div class="nlp-insight-label">Avg Words/Sentence</div>
          <div class="nlp-insight-value">${ling.avgWPS}</div>
        </div>
      </div>
    </div>

    <div class="card animate-in" style="margin-bottom:1rem;animation-delay:0.2s">
      <div class="card-title">Review Text Analysis</div>
      <div class="highlighted-text">${highlightedText}</div>
      <div style="margin-top:0.5rem;font-size:0.72rem;color:var(--txt-muted);">
        <span style="background:rgba(239,68,68,0.15);border-radius:3px;padding:2px 6px;margin-right:8px;">Spam indicators</span>
        <span style="background:rgba(16,185,129,0.15);border-radius:3px;padding:2px 6px;">Honest indicators</span>
      </div>
    </div>

    <div class="card animate-in" style="animation-delay:0.25s">
      <div class="card-title">Detection Reasons</div>
      <div class="reasons-grid">${reasonsHTML || '<p style="color:var(--txt-muted);font-size:0.82rem;">No specific signals detected.</p>'}</div>
    </div>
  `;

  // Animate score bars
  requestAnimationFrame(() => {
    const mainBar = document.getElementById('mainScoreBar');
    if (mainBar) mainBar.style.width = finalScore + '%';
    document.querySelectorAll('.ml-bar-fill[data-width]').forEach(el => {
      setTimeout(() => { el.style.width = el.getAttribute('data-width'); }, 100);
    });
  });
}

function buildReasons(details, verdict) {
  const reasons = [];
  const rule    = details.rule;
  const ling    = details.ling;
  const sent    = details.sent;

  if (rule.spamCnt > 0)
    reasons.push({ icon:'circle-exclamation', color:'var(--red)', text:`${rule.spamCnt} spam word(s) detected` });
  if (rule.honestCnt > 0)
    reasons.push({ icon:'circle-check', color:'var(--green)', text:`${rule.honestCnt} honest signal(s) detected` });
  if (rule.urgencyCnt > 0)
    reasons.push({ icon:'bolt', color:'var(--orange)', text:`${rule.urgencyCnt} urgency phrase(s) found` });
  if (rule.foundSpecific)
    reasons.push({ icon:'magnifying-glass', color:'var(--cyan)', text:'Specific product details mentioned' });
  if (rule.capsWords > 3)
    reasons.push({ icon:'a', color:'var(--orange)', text:`${rule.capsWords} ALL CAPS words detected` });
  if (rule.exclamCnt > 2)
    reasons.push({ icon:'exclamation', color:'var(--orange)', text:`${rule.exclamCnt} exclamation marks found` });
  if (rule.wordCount < 20)
    reasons.push({ icon:'file-lines', color:'var(--txt-muted)', text:`Very short review (${rule.wordCount} words)` });
  if (ling.TTR > 70)
    reasons.push({ icon:'book-open', color:'var(--green)', text:'Rich and varied vocabulary' });
  if (sent.comparative > 1.5)
    reasons.push({ icon:'face-laugh', color:'var(--red)', text:'Overly positive sentiment detected' });
  if (sent.negative.length > 0)
    reasons.push({ icon:'thumbs-down', color:'var(--green)', text:'Balanced negative sentiment (authentic)' });

  return reasons.slice(0, 8);
}

function updateDashboard(result, text) {
  const rule   = result.details.rule;
  const score  = result.finalScore;
  const verdict= result.verdict;

  animateCounter('dash-spam',    rule.spamCnt);
  animateCounter('dash-honest',  rule.honestCnt);
  animateCounter('dash-caps',    rule.capsWords);
  animateCounter('dash-exclaim', rule.exclamCnt);
  animateCounter('dash-words',   rule.wordCount);
  animateCounter('dashQualityScore', score);
  animateCounter('dashMlPct',    score, '%');

  const dashBadge   = document.getElementById('dashVerdictBadge');
  const verdictClass= verdict.toLowerCase();
  dashBadge.innerHTML = `<span class="verdict-badge ${verdictClass}">
    <i class="fa-solid fa-${verdict==='REAL'?'circle-check':verdict==='SUSPICIOUS'?'triangle-exclamation':'circle-xmark'}"></i>
    ${verdict}
  </span>`;

  const bullets = document.getElementById('dashMlBullets');
  const bcolor  = verdict === 'REAL' ? 'var(--green)' : verdict === 'SUSPICIOUS' ? 'var(--orange)' : 'var(--red)';
  const bulletItems = buildReasons(result.details, verdict).slice(0,3);
  bullets.innerHTML = bulletItems.map(r =>
    `<div class="ml-bullet"><div class="ml-bullet-dot" style="background:${bcolor}"></div><span>${r.text}</span></div>`
  ).join('');

  document.getElementById('dashInsightText').innerHTML = generateInsight(verdict, result);

  // Update signal chart
  const spamPct   = rule.spamCnt   * 5;
  const urgPct    = rule.urgencyCnt* 8;
  const capsPct   = rule.capsWords > 0 ? Math.min(rule.capsWords*5,30) : 0;
  const honestPct = rule.honestCnt * 5;
  const rest      = Math.max(0, 100 - spamPct - urgPct - capsPct - honestPct);

  if (signalChartInst) {
    signalChartInst.data.datasets[0].data = [
      spamPct||0.001, urgPct||0.001, capsPct||0.001, honestPct||1, rest||1
    ];
    signalChartInst.update();
  }

  // Update quality chart
  if (qualityChartInst) {
    qualityChartInst.data.datasets[0].data = [score, 100-score];
    qualityChartInst.data.datasets[0].backgroundColor[0] =
      score >= 70 ? '#10b981' : score >= 40 ? '#f59e0b' : '#ef4444';
    qualityChartInst.update();
  }

  // Quality label
  const qlabel = document.getElementById('dashQualityLabel');
  qlabel.textContent = score >= 70 ? 'Good Quality' : score >= 40 ? 'Suspicious' : 'Poor Quality';
  qlabel.style.color = score >= 70 ? 'var(--green)' : score >= 40 ? 'var(--orange)' : 'var(--red)';
}

function generateInsight(verdict, result) {
  const score = result.finalScore;
  const rule  = result.details.rule;
  if (verdict === 'REAL') {
    return `This review appears to be <strong style="color:var(--green)">REAL</strong> with ${score}% confidence. It ${rule.foundSpecific ? 'contains specific product details and ' : ''}uses natural language patterns${rule.honestCnt > 0 ? ' with ' + rule.honestCnt + ' honest signal(s)' : ''} typical of authentic reviews.`;
  }
  if (verdict === 'SUSPICIOUS') {
    return `This review is <strong style="color:var(--orange)">SUSPICIOUS</strong> with ${score}% authenticity score. It shows some genuine signals but also contains ${rule.spamCnt > 0 ? rule.spamCnt + ' potential spam indicator(s)' : 'unusual patterns'} that warrant further review.`;
  }
  return `This review is likely <strong style="color:var(--red)">FAKE</strong> with only ${score}% authenticity. It ${rule.spamCnt > 0 ? 'contains ' + rule.spamCnt + ' spam signal(s)' : ''}${rule.capsWords > 3 ? ' and excessive capitalization' : ''}${rule.exclamCnt > 2 ? ' and ' + rule.exclamCnt + ' exclamation marks' : ''}, typical of fraudulent reviews.`;
}

/* ============================================================
   COUNTER ANIMATION
   ============================================================ */
function animateCounter(id, target, suffix='') {
  const el = document.getElementById(id);
  if (!el) return;
  const start    = 0;
  const duration = 600;
  const step     = Math.max(1, Math.round(target / (duration / 16)));
  let current    = start;
  const timer = setInterval(() => {
    current = Math.min(current + step, target);
    el.textContent = current + suffix;
    if (current >= target) clearInterval(timer);
  }, 16);
}

/* ============================================================
   HISTORY
   ============================================================ */
function getHistory() {
  try { return JSON.parse(localStorage.getItem('rinneganHistory') || localStorage.getItem('trueReviewHistory') || '[]'); } catch { return []; }
}

function saveToHistory(text, result) {
  const history = getHistory();
  history.unshift({
    id:      Date.now(),
    date:    new Date().toISOString(),
    text,
    score:   result.finalScore,
    verdict: result.verdict
  });
  localStorage.setItem('rinneganHistory', JSON.stringify(history.slice(0, 200)));
}

function renderHistory() {
  const tbody   = document.getElementById('historyBody');
  const history = getHistory();

  if (!history.length) {
    tbody.innerHTML = `<tr><td colspan="6"><div class="empty-state">
      <i class="fa-solid fa-clock-rotate-left"></i>
      <p>No analyses yet.<br>Start by analyzing a review.</p>
    </div></td></tr>`;
    return;
  }

  tbody.innerHTML = history.map((h, i) => {
    const d    = new Date(h.date);
    const dt   = d.toLocaleDateString('en-IN',{day:'2-digit',month:'short',year:'numeric'}) + ' ' +
                 d.toLocaleTimeString('en-IN',{hour:'2-digit',minute:'2-digit'});
    const prev = h.text.length > 60 ? h.text.slice(0,60)+'…' : h.text;
    const vc   = h.verdict ? h.verdict.toLowerCase() : 'suspicious';
    return `<tr>
      <td style="color:var(--txt-muted)">${i+1}</td>
      <td style="white-space:nowrap;font-size:0.78rem;">${dt}</td>
      <td style="max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escHtml(prev)}</td>
      <td><span style="font-weight:700;color:${getVerdictColor(h.verdict||'SUSPICIOUS')}">${h.score}</span></td>
      <td><span class="verdict-tag ${vc}">${h.verdict||'SUSPICIOUS'}</span></td>
      <td><button class="btn-small-del" onclick="deleteHistoryItem(${h.id})">Delete</button></td>
    </tr>`;
  }).join('');
}

function deleteHistoryItem(id) {
  const history = getHistory().filter(h => h.id !== id);
  localStorage.setItem('rinneganHistory', JSON.stringify(history));
  renderHistory();
}

function clearHistory() {
  if (confirm('Clear all analysis history?')) {
    localStorage.removeItem('rinneganHistory');
    localStorage.removeItem('trueReviewHistory');
    renderHistory();
    showToast('History cleared.','success');
  }
}

/* ============================================================
   REPORTS
   ============================================================ */
function updateReportStats() {
  const history = getHistory();
  const real    = history.filter(h=>h.verdict==='REAL').length;
  const susp    = history.filter(h=>h.verdict==='SUSPICIOUS').length;
  const fake    = history.filter(h=>h.verdict==='FAKE').length;

  document.getElementById('rpt-total').textContent = history.length;
  document.getElementById('rpt-real').textContent  = real;
  document.getElementById('rpt-susp').textContent  = susp;
  document.getElementById('rpt-fake').textContent  = fake;
}

function getLast7DaysData(history) {
  const labels=[], counts=[];
  for (let i=6; i>=0; i--) {
    const d = new Date();
    d.setDate(d.getDate()-i);
    const key = d.toISOString().slice(0,10);
    labels.push(d.toLocaleDateString('en-IN',{weekday:'short',day:'numeric'}));
    counts.push(history.filter(h => h.date && h.date.startsWith(key)).length);
  }
  return { labels, counts };
}

function downloadCSV() {
  const history = getHistory();
  if (!history.length) { showToast('No history to export.','error'); return; }

  const rows = [['#','Date','Review','Score','Verdict']];
  history.forEach((h,i) => {
    rows.push([i+1, new Date(h.date).toLocaleString(), '"'+h.text.replace(/"/g,'""')+'"', h.score, h.verdict]);
  });
  const csv  = rows.map(r=>r.join(',')).join('\n');
  const blob = new Blob([csv], {type:'text/csv'});
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = `Rinnegan_Report_${new Date().toISOString().slice(0,10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
  showToast('Report downloaded!','success');
}

/* ============================================================
   SETTINGS
   ============================================================ */
function loadSettings() {
  const gKey = localStorage.getItem('setting_geminiKey');
  const oKey = localStorage.getItem('setting_openaiKey');
  if (gKey) document.getElementById('geminiKey').value = gKey;
  if (oKey) document.getElementById('openaiKey').value = oKey;

  const auto = localStorage.getItem('setting_autoAnalyze') === 'true';
  const high = localStorage.getItem('setting_showHighlights') !== 'false';
  document.getElementById('autoAnalyze').checked    = auto;
  document.getElementById('showHighlights').checked = high;

  const sens = localStorage.getItem('setting_sensitivity') || '5';
  document.getElementById('sensitivitySlider').value = sens;
  document.getElementById('sensitivityVal').textContent = sens;
}

function saveKey(type) {
  const id  = type === 'gemini' ? 'geminiKey' : 'openaiKey';
  const val = document.getElementById(id).value.trim();
  localStorage.setItem('setting_'+type+'Key', val);
  showToast(`${type === 'gemini' ? 'Gemini' : 'OpenAI'} API key saved!`,'success');
}

function saveSetting(key, value) {
  localStorage.setItem('setting_'+key, value);
}

function updateSlider() {
  const val = document.getElementById('sensitivitySlider').value;
  document.getElementById('sensitivityVal').textContent = val;
  saveSetting('sensitivity', val);
}

/* ============================================================
   TOAST
   ============================================================ */
function showToast(msg, type='success') {
  const wrap = document.getElementById('toastWrap');
  const t    = document.createElement('div');
  t.className = 'toast ' + (type==='error'?'error':type==='info'?'info':'success');
  t.innerHTML = `<i class="fa-solid fa-${type==='error'?'circle-xmark':type==='info'?'circle-info':'circle-check'}"></i> ${msg}`;
  wrap.appendChild(t);
  setTimeout(() => t.remove(), 3000);
}

/* ============================================================
   SIDEBAR MOBILE
   ============================================================ */
function toggleSidebar() {
  const sb  = document.getElementById('sidebar');
  const ov  = document.getElementById('sidebarOverlay');
  const open= sb.classList.toggle('open');
  ov.classList.toggle('visible', open);
}

function closeSidebar() {
  document.getElementById('sidebar').classList.remove('open');
  document.getElementById('sidebarOverlay').classList.remove('visible');
}

/* ============================================================
   CSV UPLOAD
   ============================================================ */
function handleCSV(e) {
  const file   = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = ev => {
    const rows = ev.target.result.split('\n').filter(r=>r.trim());
    if (rows.length > 0) {
      const review = rows[rows.length > 1 ? 1 : 0].replace(/^"|"$/g,'');
      document.getElementById('reviewInput').value = review;
      showToast(`CSV loaded. ${rows.length} row(s) found.`,'success');
    }
  };
  reader.readAsText(file);
  e.target.value = '';
}

/* ============================================================
   HELPERS
   ============================================================ */
function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

/* ============================================================
   INIT SAMPLE PILLS
   ============================================================ */
function initSamplePills() {
  const fakeDiv = document.getElementById('fakeSamples');
  const realDiv = document.getElementById('realSamples');

  FAKE_REVIEWS.forEach(r => {
    const b = document.createElement('button');
    b.className = 'sample-pill fake';
    b.title = r;
    b.textContent = r.length > 35 ? r.slice(0,35)+'…' : r;
    b.onclick = () => { document.getElementById('reviewInput').value = r; };
    fakeDiv.appendChild(b);
  });

  REAL_REVIEWS.forEach(r => {
    const b = document.createElement('button');
    b.className = 'sample-pill real';
    b.title = r;
    b.textContent = r.length > 35 ? r.slice(0,35)+'…' : r;
    b.onclick = () => { document.getElementById('reviewInput').value = r; };
    realDiv.appendChild(b);
  });
}

/* ============================================================
   DASHBOARD COUNTER ANIMATION ON LOAD
   ============================================================ */
function animateDashboardOnLoad() {
  setTimeout(() => animateCounter('dash-honest', 3), 400);
  setTimeout(() => animateCounter('dash-words',  17), 500);
  setTimeout(() => animateCounter('dashQualityScore', 71), 600);
}

/* ============================================================
   BOOT
   ============================================================ */
document.addEventListener('DOMContentLoaded', () => {
  initSamplePills();
  initDashboardCharts(71);
  animateDashboardOnLoad();
  loadSettings();
});