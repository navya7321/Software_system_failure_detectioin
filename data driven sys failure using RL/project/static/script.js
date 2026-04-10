let cpuChart, memChart;

async function fetchMetrics() {
  const res = await fetch('/metrics');
  return res.json();
}

async function fetchActions() {
  const res = await fetch('/actions');
  return res.json();
}

async function fetchSummary() {
  const res = await fetch('/summary');
  return res.json();
}

function fmtTime(ts) {
  const d = new Date(ts);
  return d.toLocaleTimeString();
}

function ensureCharts(ctxCpu, ctxMem) {
  if (!cpuChart) {
    cpuChart = new Chart(ctxCpu, {
      type: 'line',
      data: { labels: [], datasets: [{ label: 'CPU %', data: [], borderColor: '#00e5ff', backgroundColor: 'rgba(0,229,255,0.08)', fill: true, tension: 0.3 }] },
      options: { responsive: true, plugins: { legend: { labels: { color: '#cbd5e1' } } }, scales: { x: { ticks: { color: '#94a3b8' } }, y: { min: 0, max: 100, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(148,163,184,0.15)' } } } }
    });
  }
  if (!memChart) {
    memChart = new Chart(ctxMem, {
      type: 'line',
      data: { labels: [], datasets: [{ label: 'Memory %', data: [], borderColor: '#a3ff12', backgroundColor: 'rgba(163,255,18,0.08)', fill: true, tension: 0.3 }] },
      options: { responsive: true, plugins: { legend: { labels: { color: '#cbd5e1' } } }, scales: { x: { ticks: { color: '#94a3b8' } }, y: { min: 0, max: 100, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(148,163,184,0.15)' } } } }
    });
  }
}

async function refresh() {
  const metrics = await fetchMetrics();
  const statusEl = document.getElementById('status');
  statusEl.textContent = metrics.status;
  statusEl.className = metrics.status === 'Healthy' ? 'badge healthy' : 'badge failed';
  const healthDot = document.getElementById('health-dot');
  const healthText = document.getElementById('health-text');
  const healthCard = document.getElementById('health-card');
  healthDot.className = 'dot ' + (metrics.status === 'Healthy' ? 'healthy' : 'failed');
  healthText.textContent = metrics.status;
  healthCard.classList.remove('pulse');
  void healthCard.offsetWidth; // restart animation
  healthCard.classList.add('pulse');

  const labels = metrics.metrics.map(m => fmtTime(m.timestamp));
  const cpuData = metrics.metrics.map(m => m.cpu);
  const memData = metrics.metrics.map(m => m.memory);

  cpuChart.data.labels = labels;
  cpuChart.data.datasets[0].data = cpuData;
  cpuChart.update();

  memChart.data.labels = labels;
  memChart.data.datasets[0].data = memData;
  memChart.update();

  const actions = await fetchActions();
  const tbody = document.querySelector('#actions tbody');
  tbody.innerHTML = '';
  actions.forEach(a => {
    const tr = document.createElement('tr');
    const rtime = a.recovery_time != null ? `${a.recovery_time.toFixed ? a.recovery_time.toFixed(2) : a.recovery_time}s` : '-';
    tr.innerHTML = `<td>${fmtTime(a.timestamp)}</td><td>${a.action}</td><td>${a.result}</td><td>${a.reward}</td>`;
    tbody.appendChild(tr);
  });
}

async function call(path, options) {
  const res = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(options || {}) });
  return res.json();
}

function toast(message) {
  const t = document.getElementById('toast');
  t.textContent = message;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 1800);
}

window.addEventListener('DOMContentLoaded', async () => {
  const ctxCpu = document.getElementById('cpuChart').getContext('2d');
  const ctxMem = document.getElementById('memChart').getContext('2d');
  ensureCharts(ctxCpu, ctxMem);

  document.getElementById('btn-train').addEventListener('click', async () => {
    const btn = document.getElementById('btn-train');
    btn.disabled = true; btn.textContent = 'Training...';
    await call('/train', { episodes: 10 });
    btn.disabled = false; btn.textContent = 'Train AI';
    toast('Training complete');
    await refresh();
  });

  document.getElementById('btn-simulate').addEventListener('click', async () => {
    await call('/simulate_failure');
    toast('Failure simulated');
    await refresh();
  });

  document.getElementById('btn-recover').addEventListener('click', async () => {
    const r = await call('/recover');
    toast(`Recovery action: ${r.action} → ${r.result}`);
    await refresh();
  });

  await refresh();
  setInterval(refresh, 3000);
});
