// ============================================================
// frontend/js/app.js - 前端核心交互逻辑
// 功能：API调用、图表渲染、深色模式、弹窗预警、报告导出
// ============================================================

// ==================== 全局状态 ====================
let currentData = null;
let sentimentData = null;
let doughnutChart = null;
let trendChart = null;
let darkMode = localStorage.getItem('darkMode') === 'true';

// ==================== 初始化 ====================
document.addEventListener('DOMContentLoaded', () => {
  if (darkMode) document.documentElement.classList.add('dark');
  updateThemeIcon();
  initCharts();
  initWorkflowViz();
});

// ==================== 深浅色切换 ====================
function toggleTheme() {
  darkMode = !darkMode;
  document.documentElement.classList.toggle('dark', darkMode);
  localStorage.setItem('darkMode', darkMode);
  updateThemeIcon();
  if (currentData) updateAllPanels(currentData);
}

function updateThemeIcon() {
  const icon = document.getElementById('themeIcon');
  icon.className = darkMode ? 'fas fa-sun text-lg' : 'fas fa-moon text-lg';
}

// ==================== 图表初始化 ====================
function initCharts() {
  const isDark = darkMode;

  // 环形图 - 情感分布
  const dCtx = document.getElementById('sentimentDoughnut').getContext('2d');
  doughnutChart = new Chart(dCtx, {
    type: 'doughnut',
    data: {
      labels: ['负面', '中性', '正向'],
      datasets: [{
        data: [0, 0, 1],
        backgroundColor: ['#ef4444', '#94a3b8', '#22c55e'],
        borderColor: isDark ? '#1e293b' : '#ffffff',
        borderWidth: 3,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: true, cutout: '65%',
      plugins: { legend: { display: false } }
    }
  });

  // 折线图 - 热度走势
  const tCtx = document.getElementById('trendLine').getContext('2d');
  trendChart = new Chart(tCtx, {
    type: 'line',
    data: {
      labels: ['当前', '2小时后', '12小时后', '24小时后'],
      datasets: [{
        label: '风险指数走势',
        data: [0, 0, 0, 0],
        borderColor: '#3b82f6',
        backgroundColor: 'rgba(59,130,246,0.1)',
        fill: true, tension: 0.4,
        pointRadius: 5, pointBackgroundColor: '#3b82f6',
        borderWidth: 2.5,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: true,
      plugins: { legend: { display: false } },
      scales: {
        y: { min: 0, max: 100, grid: { color: isDark ? '#334155' : '#e2e8f0' } },
        x: { grid: { display: false } }
      }
    }
  });
}

// ==================== 工作流可视化 ====================
function initWorkflowViz() {
  const container = document.getElementById('workflowViz');
  const steps = [
    { id: 'crawler', icon: 'fa-magnifying-glass', label: '爬虫Agent', desc: '全网舆情抓取' },
    { id: 'sentiment', icon: 'fa-face-smile', label: '情感分析Agent', desc: 'NLP情感评分' },
    { id: 'parallel', icon: 'fa-code-branch', label: '并行节点', desc: 'RAG检索 + MySQL查询', isParallel: true },
    { id: 'competitor', icon: 'fa-chess', label: '竞品分析Agent', desc: '竞品动作推演' },
    { id: 'risk', icon: 'fa-calculator', label: '风险计算', desc: '风险指数+走势预测' },
    { id: 'pr', icon: 'fa-bullhorn', label: '公关策略Agent', desc: '分级应对方案' },
  ];

  container.innerHTML = steps.map((s, i) => `
    <div class="flex items-start gap-3 relative">
      <div class="flex flex-col items-center">
        <div id="wf-${s.id}" class="w-10 h-10 rounded-full flex items-center justify-center text-white text-sm shadow-lg transition-all duration-500"
             style="background:#94a3b8;">
          <i class="fas ${s.icon}"></i>
        </div>
        ${i < steps.length - 1 ? `<div class="w-0.5 h-8 my-1 rounded transition-all duration-500" id="line-${s.id}" style="background:#cbd5e1;"></div>` : ''}
      </div>
      <div class="flex-1 pb-2">
        <p class="text-sm font-medium">${s.label}</p>
        <p class="text-xs opacity-50">${s.desc}</p>
        ${s.isParallel ? '<p class="text-[10px] text-purple-500 mt-0.5"><i class="fas fa-bolt"></i> 并行执行</p>' : ''}
      </div>
    </div>
  `).join('');
}

function highlightWorkflowNodes() {
  ['crawler', 'sentiment', 'parallel', 'competitor', 'risk', 'pr'].forEach((id, i) => {
    setTimeout(() => {
      const node = document.getElementById(`wf-${id}`);
      const line = document.getElementById(`line-${id}`);
      if (node) { node.style.background = 'linear-gradient(135deg, #3b82f6, #8b5cf6)'; node.classList.add('flow-animate'); }
      if (line) line.style.background = 'linear-gradient(180deg, #3b82f6, #8b5cf6)';
    }, i * 300);
  });
}

// ==================== 核心：运行分析 ====================
async function runAnalysis() {
  const query = document.getElementById('searchQuery').value.trim();
  const brand = document.getElementById('brandName').value.trim();
  const count = parseInt(document.getElementById('simCount').value);

  if (!brand) { showToast('请输入品牌名称', 'warn'); return; }
  // query 可以为空，系统会用 LLM 自动分析品牌

  showLoading(true);
  const btn = document.getElementById('analyzeBtn');
  btn.disabled = true;
  btn.innerHTML = '<i class="fas fa-spinner spinner mr-2"></i>分析中...';
  highlightWorkflowNodes();

  try {
    const resp = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, brand_name: brand, simulate_count: count })
    });
    if (!resp.ok) throw new Error((await resp.json()).detail || '分析失败');

    const result = await resp.json();
    if (result.success && result.data) {
      currentData = result.data;
      prepareSentimentData();
      updateAllPanels(currentData);
      updateWorkflowStatus('success');

      const rl = currentData.summary.risk_level;
      if (rl === '特级' || rl === '高') showAlert(rl, currentData.summary.risk_index);
      showToast(`分析完成：风险等级 ${rl}，风险指数 ${currentData.summary.risk_index}`, 'success');
    }
  } catch (err) {
    updateWorkflowStatus('error', err.message);
    showToast('分析失败：' + err.message, 'error');
  } finally {
    showLoading(false);
    btn.disabled = false;
    btn.innerHTML = '<i class="fas fa-magnifying-glass-chart mr-2"></i>开始分析';
  }
}

// ==================== 数据整理 ====================
function prepareSentimentData() {
  if (!currentData) return;
  const sentMap = {};
  (currentData.sentiment_analysis || []).forEach(s => { sentMap[s.original_id] = s; });
  sentimentData = (currentData.sentiments || []).map(item => ({
    ...item, analysis: sentMap[item.id] || null,
  }));
}

function updateAllPanels(data) {
  updateKPICards(data);
  updateCharts(data);
  updateSentimentList();
  updateSimilarCases(data);
  updateCompetitorList(data);
  updateStrategyList(data);
}

// ==================== KPI卡片 ====================
function updateKPICards(data) {
  const s = data.summary;
  document.getElementById('kpiRiskIndex').textContent = s.risk_index;
  document.getElementById('kpiNegative').textContent = s.negative_count;
  document.getElementById('kpiTotalSent').textContent = `共 ${s.total_sentiments} 条`;
  document.getElementById('kpiHotProb').textContent = s.hot_search_probability + '%';

  const levelEl = document.getElementById('kpiRiskLevel');
  levelEl.textContent = s.risk_level;

  const styles = {
    '特级': { cls: 'risk-critical', desc: '⚠️ 全员应急响应', bg: '#ef4444' },
    '高': { cls: 'risk-high', desc: '🔥 立即启动公关', bg: '#f97316' },
    '中': { cls: 'risk-medium', desc: '📋 密切关注态势', bg: '#eab308' },
    '低': { cls: 'risk-low', desc: '✅ 日常监控即可', bg: '#22c55e' },
  };
  const ls = styles[s.risk_level] || styles['低'];
  levelEl.className = `text-3xl font-bold px-3 py-1 rounded-lg inline-block ${ls.cls}`;
  document.getElementById('kpiRiskDesc').textContent = ls.desc;
  document.getElementById('riskIndexBg').style.background = ls.bg;

  const kpiRisk = document.getElementById('kpiRiskIndex');
  kpiRisk.style.color = s.risk_index >= 75 ? '#ef4444' : s.risk_index >= 50 ? '#f97316' : s.risk_index >= 25 ? '#eab308' : '#22c55e';
}

// ==================== 图表更新 ====================
function updateCharts(data) {
  const s = data.summary;
  const neuCount = s.total_sentiments - s.negative_count - s.positive_count;
  const isDark = darkMode;

  doughnutChart.data.datasets[0].data = [s.negative_count, Math.max(0, neuCount), s.positive_count];
  doughnutChart.data.datasets[0].borderColor = isDark ? '#1e293b' : '#ffffff';
  doughnutChart.update();
  document.getElementById('dNeg').textContent = s.negative_count;
  document.getElementById('dNeu').textContent = Math.max(0, neuCount);
  document.getElementById('dPos').textContent = s.positive_count;

  const trend = data.trend_forecast;
  trendChart.data.datasets[0].data = [s.risk_index, trend['2h'] || s.risk_index, trend['12h'] || s.risk_index, trend['24h'] || s.risk_index];
  trendChart.options.scales.y.grid.color = isDark ? '#334155' : '#e2e8f0';
  trendChart.update();
}

// ==================== 舆情列表 ====================
function updateSentimentList() {
  const container = document.getElementById('sentimentList');
  if (!sentimentData || sentimentData.length === 0) {
    container.innerHTML = '<p class="text-sm opacity-40 text-center py-12">暂无舆情数据</p>';
    return;
  }

  const fp = document.getElementById('filterPlatform').value;
  const fs = document.getElementById('filterSentiment').value;
  let filtered = sentimentData;
  if (fp !== 'all') filtered = filtered.filter(s => s.platform === fp);
  if (fs !== 'all') filtered = filtered.filter(s => s.analysis && s.analysis.label === fs);

  container.innerHTML = filtered.map(s => {
    const a = s.analysis || {};
    const label = a.label || '中性';
    const rowCls = label === '负面' ? 'negative-row' : label === '正向' ? 'positive-row' : 'neutral-row';
    const labelClr = label === '负面' ? 'text-red-500' : label === '正向' ? 'text-green-500' : 'text-gray-400';
    const tags = (a.risk_tags || []).map(t => `<span class="px-1.5 py-0.5 rounded text-[10px] bg-red-50 text-red-600 dark:bg-red-900 dark:text-red-300">${t}</span>`).join(' ');
    const time = s.publish_time ? new Date(s.publish_time).toLocaleString('zh-CN') : '';

    return `<div class="p-3 rounded-lg text-sm ${rowCls}" style="background:var(--bg-primary);">
      <div class="flex items-center gap-2 mb-1">
        <span class="px-1.5 py-0.5 rounded text-[10px] font-medium bg-brand-100 text-brand-700 dark:bg-brand-900 dark:text-brand-300">${s.platform}</span>
        <span class="text-[10px] opacity-40">${s.author}</span>
        <span class="text-[10px] opacity-40">${time}</span>
        <span class="ml-auto text-[10px] font-medium ${labelClr}">${label} (${a.score || 0})</span>
      </div>
      <p class="text-xs leading-relaxed opacity-80">${s.content}</p>
      <div class="flex items-center gap-3 mt-1.5 text-[10px] opacity-40">
        <span><i class="far fa-heart mr-0.5"></i>${s.likes}</span>
        <span><i class="fas fa-retweet mr-0.5"></i>${s.shares}</span>
        <span><i class="far fa-comment mr-0.5"></i>${s.comments}</span>
        <span class="ml-auto">${tags}</span>
      </div>
    </div>`;
  }).join('');
}

function filterSentiments() { updateSentimentList(); }

// ==================== 相似案例 ====================
function updateSimilarCases(data) {
  const container = document.getElementById('similarCasesGrid');
  const cases = data.similar_cases || [];
  if (cases.length === 0) {
    container.innerHTML = '<p class="text-sm opacity-40 col-span-full text-center py-8">未找到相似历史案例</p>';
    return;
  }
  const riskColors = { '特级': 'bg-red-500', '高': 'bg-orange-500', '中': 'bg-yellow-500', '低': 'bg-green-500' };

  container.innerHTML = cases.map(c => `
    <div class="p-4 rounded-lg border transition-all hover:shadow-md" style="background:var(--bg-primary);border-color:var(--border);">
      <div class="flex items-start justify-between mb-2">
        <h4 class="text-sm font-semibold flex-1 mr-2">${c.title}</h4>
        <span class="px-2 py-0.5 rounded-full text-[10px] font-bold text-white ${riskColors[c.risk_level] || 'bg-gray-400'} whitespace-nowrap">${c.risk_level}</span>
      </div>
      <div class="flex items-center gap-2 mb-2">
        <span class="px-1.5 py-0.5 rounded text-[10px] bg-purple-50 text-purple-600 dark:bg-purple-900 dark:text-purple-300">${c.category}</span>
        <span class="text-[11px] opacity-50">相似度 <b class="text-brand-600">${Math.round(c.score * 100)}%</b></span>
      </div>
      <p class="text-xs opacity-60 mb-2 line-clamp-3">${c.description}</p>
      <details class="text-xs">
        <summary class="cursor-pointer text-brand-600 hover:text-brand-700 font-medium">查看处理方案</summary>
        <p class="mt-2 p-2 rounded opacity-70" style="background:var(--bg-card);">${c.resolution}</p>
      </details>
    </div>
  `).join('');
}

// ==================== 竞品列表 ====================
function updateCompetitorList(data) {
  const container = document.getElementById('competitorList');
  const competitors = data.competitor_analysis || [];
  if (competitors.length === 0) {
    container.innerHTML = '<p class="text-sm opacity-40 text-center py-8">暂无竞品推演数据</p>';
    return;
  }
  container.innerHTML = competitors.map(c => {
    const probClr = c.probability >= 0.7 ? 'text-red-500' : c.probability >= 0.4 ? 'text-orange-500' : 'text-gray-400';
    const barClr = c.probability >= 0.7 ? '#ef4444' : c.probability >= 0.4 ? '#f97316' : '#94a3b8';
    return `<div class="p-4 rounded-lg" style="background:var(--bg-primary);">
      <div class="flex items-center justify-between mb-2">
        <span class="text-sm font-semibold">${c.competitor_name}</span>
        <span class="px-2 py-0.5 rounded-full text-[10px] font-bold bg-brand-100 text-brand-700 dark:bg-brand-900 dark:text-brand-300">${c.action_type}</span>
      </div>
      <div class="flex items-center gap-2 mb-2">
        <div class="flex-1 h-1.5 rounded-full bg-gray-200 dark:bg-gray-700 overflow-hidden">
          <div class="h-full rounded-full transition-all duration-500" style="width:${c.probability*100}%;background:${barClr};"></div>
        </div>
        <span class="text-xs font-bold ${probClr} w-12 text-right">${Math.round(c.probability*100)}%</span>
      </div>
      <p class="text-xs opacity-60 mb-1">${c.description}</p>
      <p class="text-[10px] opacity-40"><i class="fas fa-history mr-1"></i>${c.historical_case}</p>
    </div>`;
  }).join('');
}

// ==================== 公关策略 ====================
function updateStrategyList(data) {
  const container = document.getElementById('strategyList');
  const strategies = data.pr_strategies || [];
  if (strategies.length === 0) {
    container.innerHTML = '<p class="text-sm opacity-40 text-center py-8">暂无公关策略</p>';
    return;
  }
  container.innerHTML = strategies.map((s, i) => {
    const isUrgent = s.urgency.includes('紧急');
    const borderCls = isUrgent ? 'border-red-400 bg-red-50 dark:bg-red-950' :
                      s.urgency.includes('长期') ? 'border-blue-400 bg-blue-50 dark:bg-blue-950' :
                      'border-yellow-400 bg-yellow-50 dark:bg-yellow-950';
    const stmt = s.suggested_statement || '';

    return `<div class="p-4 rounded-lg border-l-4 ${borderCls}" style="background:var(--bg-primary);border-left-width:4px;">
      <div class="flex items-center justify-between mb-2">
        <span class="text-sm font-bold">${i+1}. ${s.urgency}</span>
        <span class="text-[10px] opacity-50">${s.risk_level}风险</span>
      </div>
      <div class="mb-2">
        <p class="text-xs font-medium opacity-70 mb-1">📌 话术要点：</p>
        <ul class="space-y-0.5">${(s.talking_points || []).map(t => `<li class="text-xs opacity-60 flex items-start gap-1"><span class="text-brand-500 mt-0.5">•</span> ${t}</li>`).join('')}</ul>
      </div>
      <div class="mb-2">
        <p class="text-xs font-medium opacity-70 mb-1">📋 行动步骤：</p>
        <ol class="space-y-0.5">${(s.action_steps || []).map(a => `<li class="text-xs opacity-60">${a}</li>`).join('')}</ol>
      </div>
      ${stmt ? `<div class="mt-3 p-3 rounded-lg relative" style="background:var(--bg-card);">
        <p class="text-xs font-medium opacity-70 mb-1">📝 建议声明文案：</p>
        <p class="text-xs leading-relaxed opacity-80">${stmt}</p>
        <button onclick="copyText('${stmt.replace(/'/g, "\\'")}')" class="absolute top-2 right-2 px-2 py-1 rounded text-[10px] bg-brand-600 text-white hover:bg-brand-700 transition-colors"><i class="far fa-copy mr-1"></i>复制</button>
      </div>` : ''}
    </div>`;
  }).join('');
}

// ==================== 工作流状态 ====================
function updateWorkflowStatus(status, errorMsg = '') {
  const el = document.getElementById('workflowStatus');
  if (status === 'success') {
    el.innerHTML = `<p class="text-green-600 dark:text-green-400 font-medium"><i class="fas fa-check-circle mr-1"></i>所有Agent执行完成</p>
                    <p class="text-xs opacity-50 mt-1">爬虫 → 情感分析 → 并行(RAG+MySQL) → 竞品 → 风险计算 → 公关策略</p>`;
  } else if (status === 'error') {
    el.innerHTML = `<p class="text-red-500"><i class="fas fa-times-circle mr-1"></i>执行异常：${errorMsg}</p>`;
  }
}

// ==================== 弹窗预警 ====================
function showAlert(level, index) {
  const modal = document.getElementById('alertModal');
  document.getElementById('alertMsg').innerHTML = `系统检测到<b>${level}风险</b>！综合风险指数 <b>${index}/100</b>，请立即查看公关应对方案并启动应急响应流程。`;
  modal.classList.remove('hidden');
}

function closeAlert() { document.getElementById('alertModal').classList.add('hidden'); }

// ==================== 导出报告 ====================
async function exportReport() {
  if (!currentData) { showToast('请先运行分析', 'warn'); return; }
  showLoading(true);
  try {
    const resp = await fetch('/api/export-report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ brand_name: currentData.brand, query: currentData.query, include_cases: true, include_strategies: true })
    });
    const result = await resp.json();
    if (result.success && result.report) {
      const blob = new Blob([result.report], { type: 'text/plain;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `舆情公关报告_${currentData.brand}_${new Date().toISOString().slice(0,10)}.txt`;
      a.click();
      URL.revokeObjectURL(url);
      showToast('报告导出成功！', 'success');
    }
  } catch (err) {
    showToast('导出失败：' + err.message, 'error');
  } finally { showLoading(false); }
}

// ==================== 工具函数 ====================
function copyText(text) {
  navigator.clipboard.writeText(text).then(() => showToast('已复制到剪贴板', 'success')).catch(() => showToast('复制失败', 'error'));
}

function showToast(message, type = 'info') {
  const toast = document.getElementById('toast');
  const colors = { success: 'border-green-400 text-green-700', error: 'border-red-400 text-red-700', warn: 'border-yellow-400 text-yellow-700', info: 'border-blue-400 text-blue-700' };
  const icons = { success: 'fa-check-circle', error: 'fa-times-circle', warn: 'fa-exclamation-circle', info: 'fa-info-circle' };
  toast.innerHTML = `<div class="card px-5 py-3 flex items-center gap-2 shadow-xl text-sm ${colors[type]}"><i class="fas ${icons[type]}"></i> ${message}</div>`;
  toast.classList.remove('hidden');
  setTimeout(() => toast.classList.add('hidden'), 3000);
}

function showLoading(show) {
  document.getElementById('loadingOverlay').classList.toggle('hidden', !show);
}

function clearAll() {
  currentData = null;
  sentimentData = null;
  ['sentimentList','similarCasesGrid','competitorList','strategyList'].forEach(id => {
    document.getElementById(id).innerHTML = '<p class="text-sm opacity-40 text-center py-8">点击"开始分析"获取数据</p>';
  });
  document.getElementById('similarCasesGrid').classList.add('col-span-full');
  ['kpiRiskIndex','kpiRiskLevel','kpiNegative','kpiHotProb'].forEach(id => document.getElementById(id).textContent = '--');
  document.getElementById('kpiRiskLevel').className = 'text-3xl font-bold';
  document.getElementById('kpiRiskDesc').textContent = '等待分析';
  document.getElementById('kpiTotalSent').textContent = '共 0 条';
  document.getElementById('kpiRiskIndex').style.color = '';
  document.getElementById('workflowStatus').innerHTML = '<p class="opacity-50 text-xs">等待启动分析...</p>';

  doughnutChart.data.datasets[0].data = [0, 0, 1];
  doughnutChart.update();
  trendChart.data.datasets[0].data = [0, 0, 0, 0];
  trendChart.update();

  ['crawler','sentiment','parallel','competitor','risk','pr'].forEach(id => {
    const node = document.getElementById(`wf-${id}`);
    const line = document.getElementById(`line-${id}`);
    if (node) { node.style.background = '#94a3b8'; node.classList.remove('flow-animate'); }
    if (line) line.style.background = '#cbd5e1';
  });
  showToast('已清空所有数据', 'info');
}

// ==================== 键盘快捷键 ====================
document.addEventListener('keydown', e => {
  if (e.ctrlKey && e.key === 'Enter') { e.preventDefault(); runAnalysis(); }
  if (e.key === 'Escape') closeAlert();
});
