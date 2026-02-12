const siteData = window.showcaseSite;

if (!siteData) {
  throw new Error('showcaseSite data is missing. Make sure content.js is loaded before app.js.');
}

const langButtons = document.querySelectorAll('[data-lang]');
const navMarkEl = document.getElementById('nav-mark');
const badgeEl = document.getElementById('badge');
const heroKickerEl = document.getElementById('hero-kicker');
const titleEl = document.getElementById('title');
const descEl = document.getElementById('description');
const githubBtnEl = document.getElementById('github-btn');
const demoBtnEl = document.getElementById('demo-btn');
const tagListEl = document.getElementById('tag-list');
const metricListEl = document.getElementById('metric-list');
const sectionTitleEl = document.getElementById('section-title');
const highlightListEl = document.getElementById('highlight-list');
const screenTitleEl = document.getElementById('screen-title');
const screenListEl = document.getElementById('screen-list');
const bilingualTitleEl = document.getElementById('bilingual-title');
const zhLabelEl = document.getElementById('zh-label');
const enLabelEl = document.getElementById('en-label');
const zhCopyEl = document.getElementById('zh-copy');
const enCopyEl = document.getElementById('en-copy');
const currentYearEl = document.getElementById('year');

function render(lang) {
  const text = siteData[lang];
  const metrics = siteData.metrics[lang];
  const features = siteData.features[lang];
  const screenshots = siteData.screenshots[lang];

  navMarkEl.textContent = text.navMark;
  badgeEl.textContent = text.badge;
  heroKickerEl.textContent = text.kicker;
  titleEl.textContent = text.title;
  descEl.textContent = text.description;

  githubBtnEl.textContent = siteData.githubText[lang];
  githubBtnEl.href = siteData.githubUrl;

  demoBtnEl.textContent = siteData.demoText[lang];
  demoBtnEl.href = siteData.demoUrl;
  if (siteData.demoUrl.startsWith('#')) {
    demoBtnEl.removeAttribute('target');
    demoBtnEl.removeAttribute('rel');
  } else {
    demoBtnEl.target = '_blank';
    demoBtnEl.rel = 'noreferrer';
  }

  tagListEl.innerHTML = '';
  for (const tag of text.tags) {
    const li = document.createElement('li');
    li.textContent = tag;
    tagListEl.appendChild(li);
  }

  metricListEl.innerHTML = '';
  for (const item of metrics) {
    const li = document.createElement('li');
    li.innerHTML = `<div class="metric-value">${item.value}</div><div class="metric-label">${item.label}</div>`;
    metricListEl.appendChild(li);
  }

  sectionTitleEl.textContent = text.sectionTitle;
  highlightListEl.innerHTML = '';
  for (const feature of features) {
    const li = document.createElement('li');
    li.textContent = feature;
    highlightListEl.appendChild(li);
  }

  screenTitleEl.textContent = text.screenTitle;
  screenListEl.innerHTML = '';
  for (const shot of screenshots) {
    const li = document.createElement('li');
    li.className = 'screen-card';
    li.innerHTML = `
      <figure class="screen-frame">
        <img src="${shot.image}" alt="${shot.title}" loading="lazy" decoding="async" />
      </figure>
      <div class="screen-copy">
        <h3>${shot.title}</h3>
        <p>${shot.description}</p>
      </div>
    `;
    screenListEl.appendChild(li);
  }

  bilingualTitleEl.textContent = text.bilingualTitle;
  zhLabelEl.textContent = text.zhLabel;
  enLabelEl.textContent = text.enLabel;
  zhCopyEl.textContent = siteData.projectIntro?.zh || siteData.zh.description;
  enCopyEl.textContent = siteData.projectIntro?.en || siteData.en.description;

  for (const btn of langButtons) {
    const active = btn.dataset.lang === lang;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-pressed', active ? 'true' : 'false');
  }

  document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
}

for (const btn of langButtons) {
  btn.addEventListener('click', () => render(btn.dataset.lang));
}

currentYearEl.textContent = String(new Date().getFullYear());
render('zh');
