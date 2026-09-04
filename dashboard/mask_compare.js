// mask_compare.js — Mask comparison tab for the direct-answer dashboard.
// Loads two masks from the same prompt, computes difference metrics,
// and renders interactive visualizations for comparing them.

// ══════════════════════════════════════════════════════════════
// State
// ══════════════════════════════════════════════════════════════
let cmpMaskA = null, cmpMaskB = null;
let cmpPathA = null, cmpPathB = null;
let cmpScoresA = null, cmpScoresB = null;  // flat arrays of learnable scores
let cmpMatA = null, cmpMatB = null;        // S×S matrices
let cmpFrozen = null;                       // S×S boolean frozen filter
let cmpLearnableEdges = [];                 // [{i, j, scoreA, scoreB}]
let cmpSentences = [];
let cmpGroups = [];                         // cached mask groups
let cmpSelectedPairIdx = null;
let cmpHeatmapCleanup = null;  // teardown function for canvas heatmap event listeners
let cmpHeatmapState = null;    // {offsetX, offsetY, gridSize, S, diffMat, maxAbs, hoverCell, selectedCell}

const CMP_COLORS = {
  maskA: '#2196f3',
  maskB: '#ff9800',
  diff_pos: '#ef4444',
  diff_neg: '#3b82f6',
};

// ══════════════════════════════════════════════════════════════
// Initialization
// ══════════════════════════════════════════════════════════════
function initCompare() {
  loadCompareGroups();
  document.getElementById('cmp-metric-select').addEventListener('change', renderCompareView);
  document.getElementById('cmp-topk-slider').addEventListener('input', onTopKSliderChange);
  document.getElementById('cmp-sort-select').addEventListener('change', renderCompareView);
  document.getElementById('cmp-load-btn').addEventListener('click', loadComparisonMasks);

  const selA = document.getElementById('cmp-group-a');
  const selB = document.getElementById('cmp-group-b');
  selA.addEventListener('change', () => populateSparsitySelect('cmp-sparsity-a', selA.value));
  selB.addEventListener('change', () => populateSparsitySelect('cmp-sparsity-b', selB.value));
}

function loadCompareGroups() {
  fetch('/api/mask_groups').then(r => r.json()).then(data => {
    cmpGroups = data.groups || [];
    const ungrouped = data.ungrouped || [];
    for (const selId of ['cmp-group-a', 'cmp-group-b']) {
      const sel = document.getElementById(selId);
      sel.innerHTML = '<option value="">-- select mask group --</option>';

      const byDir = {};
      cmpGroups.forEach((g, idx) => {
        const dir = g.dir ? g.dir.split('/').slice(0, -1).join('/') : 'other';
        if (!byDir[dir]) byDir[dir] = [];
        byDir[dir].push({ group: g, idx });
      });
      for (const [dir, entries] of Object.entries(byDir).sort()) {
        const shortDir = dir.replace(/^results\/snp_sweep\//, '').replace(/^results\/circuit_discovery\//, '');
        const og = document.createElement('optgroup');
        og.label = shortDir || 'masks';
        for (const { group: g, idx } of entries) {
          const o = document.createElement('option');
          o.value = 'g:' + idx;
          const tspRange = g.sparsities.length > 1
            ? `${g.sparsities[0].tsp}-${g.sparsities[g.sparsities.length - 1].tsp}%`
            : `${g.sparsities[0].tsp}%`;
          o.textContent = `${g.label} [${tspRange}]`;
          og.appendChild(o);
        }
        sel.appendChild(og);
      }
      if (ungrouped.length > 0) {
        const og = document.createElement('optgroup');
        og.label = 'Individual masks';
        for (const p of ungrouped) {
          const o = document.createElement('option');
          o.value = 'p:' + p;
          const si = p.indexOf('snp_sweep/');
          o.textContent = si >= 0 ? p.slice(si + 'snp_sweep/'.length) : p;
          og.appendChild(o);
        }
        sel.appendChild(og);
      }
    }
  }).catch(err => console.error('Failed to load compare groups:', err));
}

function populateSparsitySelect(selectId, groupVal) {
  const sel = document.getElementById(selectId);
  sel.innerHTML = '';
  if (!groupVal || !groupVal.startsWith('g:')) {
    sel.style.display = 'none';
    return;
  }
  const idx = parseInt(groupVal.slice(2));
  const group = cmpGroups[idx];
  if (!group) { sel.style.display = 'none'; return; }
  sel.style.display = '';
  for (const entry of group.sparsities) {
    const o = document.createElement('option');
    o.value = entry.path;
    o.textContent = `${entry.tsp}% sparsity`;
    sel.appendChild(o);
  }
  const midIdx = Math.floor(group.sparsities.length / 2);
  sel.value = group.sparsities[midIdx].path;
}

function resolveSelectedPath(groupSelId, sparsitySelId) {
  const groupVal = document.getElementById(groupSelId).value;
  if (!groupVal) return null;
  if (groupVal.startsWith('p:')) return groupVal.slice(2);
  if (groupVal.startsWith('g:')) {
    const spSel = document.getElementById(sparsitySelId);
    return spSel.value || null;
  }
  return null;
}

function loadComparisonMasks() {
  const pathA = resolveSelectedPath('cmp-group-a', 'cmp-sparsity-a');
  const pathB = resolveSelectedPath('cmp-group-b', 'cmp-sparsity-b');
  if (!pathA || !pathB) {
    showCompareStatus('Select both masks before loading.', 'err');
    return;
  }
  showCompareStatus('Loading masks...', '');

  Promise.all([
    fetch('/api/mask?path=' + encodeURIComponent(pathA)).then(r => r.json()),
    fetch('/api/mask?path=' + encodeURIComponent(pathB)).then(r => r.json()),
  ]).then(([a, b]) => {
    cmpMaskA = a; cmpMaskB = b;
    cmpPathA = pathA; cmpPathB = pathB;
    processComparison();
  }).catch(err => {
    showCompareStatus('Failed to load: ' + err.message, 'err');
  });
}

function showCompareStatus(msg, cls) {
  const el = document.getElementById('cmp-status');
  el.innerHTML = cls === 'err'
    ? `<span class="compat-err">${msg}</span>`
    : cls === 'ok'
      ? `<span class="compat-ok">${msg}</span>`
      : msg;
}

// ══════════════════════════════════════════════════════════════
// Process loaded masks into comparison structures
// ══════════════════════════════════════════════════════════════
function processComparison() {
  const sA = cmpMaskA.sentences || [];
  const sB = cmpMaskB.sentences || [];
  if (sA.length !== sB.length) {
    showCompareStatus(`Sentence count mismatch: A has ${sA.length}, B has ${sB.length}`, 'err');
    return;
  }
  const S = sA.length;
  cmpSentences = sA;

  const metaA = cmpMaskA.metadata || {};
  const metaB = cmpMaskB.metadata || {};
  const gapA = metaA.sentence_gap || 0;
  const gapB = metaB.sentence_gap || 0;
  const gap = Math.max(gapA, gapB);
  const numFrozenA = metaA.num_frozen_prompt_sentences || 0;
  const numFrozenB = metaB.num_frozen_prompt_sentences || 0;
  const numFrozen = Math.max(numFrozenA, numFrozenB);

  cmpFrozen = Array.from({ length: S }, () => Array(S).fill(false));
  for (let i = 0; i < S; i++) {
    for (let j = 0; j < S; j++) {
      if (gap > 0 && Math.abs(i - j) < gap) cmpFrozen[i][j] = true;
      if (i === j) cmpFrozen[i][j] = true;
      if (j > i) cmpFrozen[i][j] = true;
      if (numFrozen > 0 && (i < numFrozen || j < numFrozen)) cmpFrozen[i][j] = true;
    }
  }

  cmpMatA = extractPairMatrix(cmpMaskA);
  cmpMatB = extractPairMatrix(cmpMaskB);

  cmpLearnableEdges = [];
  for (let i = 0; i < S; i++) {
    for (let j = 0; j < S; j++) {
      if (cmpFrozen[i][j]) continue;
      cmpLearnableEdges.push({
        i, j,
        scoreA: cmpMatA[i][j],
        scoreB: cmpMatB[i][j],
      });
    }
  }

  cmpSelectedPairIdx = null;

  const slider = document.getElementById('cmp-topk-slider');
  slider.max = cmpLearnableEdges.length;
  slider.value = Math.min(50, cmpLearnableEdges.length);
  updateTopKLabel();

  const labelA = shortLabel(cmpPathA);
  const labelB = shortLabel(cmpPathB);
  showCompareStatus(
    `Loaded: <b>${labelA}</b> vs <b>${labelB}</b><br>` +
    `${cmpLearnableEdges.length} learnable edges, ${S} sentences`,
    'ok'
  );

  document.getElementById('cmp-empty').style.display = 'none';
  document.getElementById('cmp-content').style.display = '';
  renderCompareView();
}

function extractPairMatrix(mask) {
  const meta = mask.metadata || {};
  const g = meta.mask_granularity || 'head';
  const raw = mask.scores;
  const S = (mask.sentences || []).length;
  if (g === 'pair') {
    return raw.map(r => [...r]);
  }
  // For layer/head granularity, aggregate across layers (mean)
  const res = Array.from({ length: S }, () => Array(S).fill(0));
  if (g === 'layer') {
    const keys = Object.keys(raw);
    for (const l of keys) {
      const m = raw[l];
      for (let i = 0; i < S; i++) for (let j = 0; j < S; j++) res[i][j] += m[i][j];
    }
    const n = keys.length || 1;
    for (let i = 0; i < S; i++) for (let j = 0; j < S; j++) res[i][j] /= n;
  } else {
    let count = 0;
    for (const l in raw) for (const h in raw[l]) {
      const m = raw[l][h];
      for (let i = 0; i < S; i++) for (let j = 0; j < S; j++) res[i][j] += m[i][j];
      count++;
    }
    if (count > 0) for (let i = 0; i < S; i++) for (let j = 0; j < S; j++) res[i][j] /= count;
  }
  return res;
}

function shortLabel(path) {
  if (!path) return '?';
  const parts = path.split('/');
  return parts[parts.length - 1].replace('.json', '');
}

// ══════════════════════════════════════════════════════════════
// Ranking & statistics utilities
// ══════════════════════════════════════════════════════════════
function rankArray(arr) {
  const indexed = arr.map((v, i) => ({ v, i }));
  indexed.sort((a, b) => a.v - b.v);
  const ranks = new Array(arr.length);
  for (let k = 0; k < indexed.length; k++) ranks[indexed[k].i] = k;
  return ranks;
}

function spearmanRho(xs, ys) {
  const n = xs.length;
  if (n < 3) return NaN;
  const rx = rankArray(xs), ry = rankArray(ys);
  let sumD2 = 0;
  for (let i = 0; i < n; i++) sumD2 += (rx[i] - ry[i]) ** 2;
  return 1 - (6 * sumD2) / (n * (n * n - 1));
}

function pearsonR(xs, ys) {
  const n = xs.length;
  if (n < 3) return NaN;
  const mx = d3.mean(xs), my = d3.mean(ys);
  let num = 0, dx2 = 0, dy2 = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx, dy = ys[i] - my;
    num += dx * dy; dx2 += dx * dx; dy2 += dy * dy;
  }
  return num / Math.sqrt(dx2 * dy2);
}

function percentile(sortedArr, v) {
  let lo = 0, hi = sortedArr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sortedArr[mid] < v) lo = mid + 1; else hi = mid;
  }
  return lo / sortedArr.length * 100;
}

// ══════════════════════════════════════════════════════════════
// Top-K slider
// ══════════════════════════════════════════════════════════════
function onTopKSliderChange() {
  updateTopKLabel();
  const metric = document.getElementById('cmp-metric-select').value;
  if (metric === 'pair_table') renderCompareView();
}

function updateTopKLabel() {
  const k = parseInt(document.getElementById('cmp-topk-slider').value);
  document.getElementById('cmp-topk-label').textContent = k;
}

// ══════════════════════════════════════════════════════════════
// Main render dispatcher
// ══════════════════════════════════════════════════════════════
function renderCompareView() {
  if (!cmpLearnableEdges.length) return;
  const metric = document.getElementById('cmp-metric-select').value;
  const container = document.getElementById('cmp-chart-container');
  const detailPanel = document.getElementById('cmp-detail-panel');

  const showSlider = (metric === 'pair_table');
  const showDetail = showSlider || metric === 'diff_heatmap';
  document.getElementById('cmp-topk-row').style.display = showSlider ? '' : 'none';
  document.getElementById('cmp-sort-row').style.display = showSlider ? '' : 'none';
  detailPanel.style.display = (showDetail && cmpSelectedPairIdx !== null) ? 'block' : 'none';

  // Tear down previous canvas heatmap listeners
  if (cmpHeatmapCleanup) { cmpHeatmapCleanup(); cmpHeatmapCleanup = null; }
  // Clear previous Plotly content
  Plotly.purge(container);
  container.innerHTML = '';

  switch (metric) {
    case 'pair_table': renderPairTable(container); break;
    case 'diff_heatmap': renderDiffHeatmap(container); break;
    case 'scatter_raw': renderScatter(container, 'raw'); break;
    case 'scatter_rank': renderScatter(container, 'rank'); break;
    case 'marginals_col': renderMarginals(container, 'col'); break;
    case 'marginals_row': renderMarginals(container, 'row'); break;
    case 'overlap': renderOverlap(container); break;
  }
}

// ══════════════════════════════════════════════════════════════
// View 1: Ranked pair table
// ══════════════════════════════════════════════════════════════
function renderPairTable(container) {
  const K = parseInt(document.getElementById('cmp-topk-slider').value);
  const sortBy = document.getElementById('cmp-sort-select').value;

  const edges = cmpLearnableEdges.map((e, idx) => ({
    ...e, idx,
    diff: e.scoreA - e.scoreB,
    absDiff: Math.abs(e.scoreA - e.scoreB),
  }));

  const sortedScoresA = [...edges.map(e => e.scoreA)].sort(d3.ascending);
  const sortedScoresB = [...edges.map(e => e.scoreB)].sort(d3.ascending);

  for (const e of edges) {
    e.pctA = percentile(sortedScoresA, e.scoreA);
    e.pctB = percentile(sortedScoresB, e.scoreB);
  }

  switch (sortBy) {
    case 'disagreement': edges.sort((a, b) => b.absDiff - a.absDiff); break;
    case 'score_a': edges.sort((a, b) => b.scoreA - a.scoreA); break;
    case 'score_b': edges.sort((a, b) => b.scoreB - a.scoreB); break;
    case 'diff_a_minus_b': edges.sort((a, b) => b.diff - a.diff); break;
    case 'diff_b_minus_a': edges.sort((a, b) => a.diff - b.diff); break;
    case 'joint': edges.sort((a, b) => (b.scoreA + b.scoreB) - (a.scoreA + a.scoreB)); break;
  }

  const topK = edges.slice(0, K);
  const labelA = shortLabel(cmpPathA);
  const labelB = shortLabel(cmpPathB);

  let h = '<table class="cmp-table"><thead><tr>';
  h += '<th>#</th><th>Pair (i,j)</th>';
  h += `<th>Score A</th><th>Pct A</th>`;
  h += `<th>Score B</th><th>Pct B</th>`;
  h += '<th>|Diff|</th><th>Signed</th>';
  h += '</tr></thead><tbody>';

  for (let k = 0; k < topK.length; k++) {
    const e = topK[k];
    const cls = cmpSelectedPairIdx === e.idx ? ' class="selected"' : '';
    h += `<tr${cls} data-edge-idx="${e.idx}" onclick="selectComparePair(${e.idx})">`;
    h += `<td>${k + 1}</td>`;
    h += `<td class="pair-cell">(${e.i}, ${e.j})</td>`;
    h += `<td>${e.scoreA.toFixed(4)}</td>`;
    h += `<td>${e.pctA.toFixed(0)}%</td>`;
    h += `<td>${e.scoreB.toFixed(4)}</td>`;
    h += `<td>${e.pctB.toFixed(0)}%</td>`;
    h += `<td>${e.absDiff.toFixed(4)}</td>`;
    h += `<td class="${e.diff > 0 ? 'diff-pos' : 'diff-neg'}">${e.diff > 0 ? '+' : ''}${e.diff.toFixed(4)}</td>`;
    h += '</tr>';
  }
  h += '</tbody></table>';
  container.innerHTML = h;

  if (cmpSelectedPairIdx !== null) {
    renderPairDetail(cmpSelectedPairIdx);
  }
}

function selectComparePair(edgeIdx) {
  cmpSelectedPairIdx = edgeIdx;
  document.querySelectorAll('.cmp-table tbody tr').forEach(tr => {
    tr.classList.toggle('selected', parseInt(tr.dataset.edgeIdx) === edgeIdx);
  });
  document.getElementById('cmp-detail-panel').style.display = 'block';
  renderPairDetail(edgeIdx);
}

function renderPairDetail(edgeIdx) {
  const panel = document.getElementById('cmp-detail-panel');
  const e = cmpLearnableEdges[edgeIdx];
  if (!e) { panel.innerHTML = ''; return; }

  const sI = cmpSentences[e.i] || {};
  const sJ = cmpSentences[e.j] || {};
  const labelA = shortLabel(cmpPathA);
  const labelB = shortLabel(cmpPathB);

  const sortedA = cmpLearnableEdges.map(x => x.scoreA).sort(d3.ascending);
  const sortedB = cmpLearnableEdges.map(x => x.scoreB).sort(d3.ascending);
  const pctA = percentile(sortedA, e.scoreA);
  const pctB = percentile(sortedB, e.scoreB);

  let h = `<h3>Pair (${e.i}, ${e.j})</h3>`;
  h += '<div class="cmp-pair-sentences">';
  h += `<div class="cmp-sent"><span class="cmp-sent-idx">S${e.i}</span> ${escHtml(sI.text || '')}</div>`;
  h += `<div class="cmp-sent"><span class="cmp-sent-idx">S${e.j}</span> ${escHtml(sJ.text || '')}</div>`;
  h += '</div>';

  h += '<div class="cmp-pair-scores">';
  h += `<div class="cmp-score-row">`;
  h += `<span class="cmp-score-label" style="color:${CMP_COLORS.maskA}">A</span>`;
  h += `<div class="cmp-score-bar-bg"><div class="cmp-score-bar" style="width:${pctA}%;background:${CMP_COLORS.maskA}"></div></div>`;
  h += `<span class="cmp-score-val">${e.scoreA.toFixed(4)} (${pctA.toFixed(0)}%)</span>`;
  h += `</div>`;
  h += `<div class="cmp-score-row">`;
  h += `<span class="cmp-score-label" style="color:${CMP_COLORS.maskB}">B</span>`;
  h += `<div class="cmp-score-bar-bg"><div class="cmp-score-bar" style="width:${pctB}%;background:${CMP_COLORS.maskB}"></div></div>`;
  h += `<span class="cmp-score-val">${e.scoreB.toFixed(4)} (${pctB.toFixed(0)}%)</span>`;
  h += `</div>`;
  h += `<div class="cmp-score-diff">Difference: <b>${(e.scoreA - e.scoreB) > 0 ? '+' : ''}${(e.scoreA - e.scoreB).toFixed(4)}</b></div>`;
  h += '</div>';

  panel.innerHTML = h;
}

function escHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ══════════════════════════════════════════════════════════════
// View 2: Difference heatmap (S×S) — canvas-based, matching
// the aggregated-mask style from shared.js
// ══════════════════════════════════════════════════════════════
function renderDiffHeatmap(container) {
  const S = cmpSentences.length;
  const diffMat = Array.from({ length: S }, () => Array(S).fill(null));
  let maxAbs = 0;
  for (const e of cmpLearnableEdges) {
    const d = e.scoreA - e.scoreB;
    diffMat[e.i][e.j] = d;
    maxAbs = Math.max(maxAbs, Math.abs(d));
  }
  if (maxAbs === 0) maxAbs = 1;

  const cvs = document.createElement('canvas');
  cvs.style.display = 'block';
  container.appendChild(cvs);

  const pad = { top: 36, right: 60, bottom: 40, left: 54 };

  cmpHeatmapState = {
    offsetX: 0, offsetY: 0, gridSize: 0, S, diffMat, maxAbs,
    hoverCell: null, selectedCell: null,
  };

  function cellColor(val) {
    const norm = Math.abs(val) / maxAbs;
    if (val < 0) return d3.interpolateBlues(0.15 + norm * 0.7);
    return d3.interpolateReds(0.15 + norm * 0.7);
  }

  function relayout() {
    const dpr = window.devicePixelRatio || 1;
    const W = container.clientWidth || 700;
    const gridSize = Math.max(8, (W - pad.left - pad.right) / S);
    const gridW = gridSize * S, gridH = gridSize * S;
    const H = pad.top + gridH + pad.bottom;
    const offsetX = pad.left;
    const offsetY = pad.top;

    cvs.style.width = W + 'px';
    cvs.style.height = H + 'px';
    cvs.width = Math.round(W * dpr);
    cvs.height = Math.round(H * dpr);

    const st = cmpHeatmapState;
    st.offsetX = offsetX; st.offsetY = offsetY;
    st.gridSize = gridSize;
    return { W, H, gridSize, gridW, gridH, offsetX, offsetY, dpr };
  }

  function drawHeatmap() {
    const { W, H, gridSize, gridW, gridH, offsetX, offsetY, dpr } = relayout();
    const c = cvs.getContext('2d');
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, W, H);
    const st = cmpHeatmapState;
    const hlRows = new Set(), hlCols = new Set();
    if (st.selectedCell) { hlRows.add(st.selectedCell.i); hlCols.add(st.selectedCell.j); }
    const hasHighlight = hlRows.size > 0 || hlCols.size > 0;

    for (let i = 0; i < S; i++) {
      for (let j = 0; j < S; j++) {
        const x = offsetX + j * gridSize;
        const y = offsetY + i * gridSize;
        const isHl = hlRows.has(i) || hlCols.has(j);

        if (j > i || cmpFrozen[i][j]) {
          c.fillStyle = isHl ? '#e8f5e9' : '#f1f5f9';
          c.fillRect(x, y, gridSize, gridSize);
        } else {
          const val = diffMat[i][j];
          if (val === null) {
            c.fillStyle = isHl ? '#e8f5e9' : '#f1f5f9';
          } else {
            c.fillStyle = cellColor(val);
            if (isHl) {
              c.fillRect(x, y, gridSize, gridSize);
              c.fillStyle = 'rgba(76, 175, 80, 0.10)';
            }
          }
          c.fillRect(x, y, gridSize, gridSize);
        }

        if (hasHighlight && !isHl && j <= i) {
          c.fillStyle = 'rgba(255,255,255,0.45)';
          c.fillRect(x, y, gridSize, gridSize);
        }

        c.strokeStyle = '#fff'; c.lineWidth = 0.5;
        c.strokeRect(x, y, gridSize, gridSize);
      }
    }

    c.strokeStyle = '#4caf50'; c.lineWidth = 2;
    if (st.selectedCell) {
      c.strokeRect(offsetX, offsetY + st.selectedCell.i * gridSize, gridW, gridSize);
      if (st.selectedCell.j !== st.selectedCell.i) {
        c.strokeRect(offsetX + st.selectedCell.j * gridSize, offsetY, gridSize, gridH);
      }
    }

    if (st.hoverCell && st.hoverCell.i < S && st.hoverCell.j < S) {
      const hx = offsetX + st.hoverCell.j * gridSize;
      const hy = offsetY + st.hoverCell.i * gridSize;
      c.strokeStyle = '#1e293b'; c.lineWidth = 2.5;
      c.strokeRect(hx, hy, gridSize, gridSize);
    }

    const fontSize = Math.min(10, gridSize * 0.6);
    for (let s = 0; s < S; s++) {
      const isHlLabel = hlRows.has(s) || hlCols.has(s);
      const isHv = st.hoverCell && (st.hoverCell.i === s || st.hoverCell.j === s);
      c.fillStyle = isHlLabel ? '#2e7d32' : isHv ? '#1e293b' : '#64748b';
      c.font = (isHlLabel || isHv ? 'bold ' : '') + fontSize + 'px monospace';
      c.textAlign = 'center';
      c.fillText('S' + s, offsetX + s * gridSize + gridSize / 2, offsetY + gridH + 14);
      c.textAlign = 'right';
      c.fillText('S' + s, offsetX - 4, offsetY + s * gridSize + gridSize / 2 + 3);
    }

    const legendX = offsetX + gridW + 10;
    const legendY = offsetY;
    const legendH = Math.min(gridH, 140);
    const legendW = 12;
    const steps = 30;
    for (let k = 0; k < steps; k++) {
      const frac = k / steps;
      const val = maxAbs * (1 - 2 * frac);
      const ly = legendY + frac * legendH;
      c.fillStyle = cellColor(val);
      c.fillRect(legendX, ly, legendW, legendH / steps + 1);
    }
    c.strokeStyle = '#94a3b8'; c.lineWidth = 0.5;
    c.strokeRect(legendX, legendY, legendW, legendH);

    c.fillStyle = '#64748b'; c.font = '8px monospace'; c.textAlign = 'left';
    c.fillText('A>' + maxAbs.toExponential(1), legendX + legendW + 3, legendY + 6);
    c.fillText('0', legendX + legendW + 3, legendY + legendH / 2 + 3);
    c.fillText('B>' + maxAbs.toExponential(1), legendX + legendW + 3, legendY + legendH);

    c.fillStyle = '#1e293b'; c.font = 'bold 12px sans-serif'; c.textAlign = 'center';
    c.fillText(
      `Score difference: ${shortLabel(cmpPathA)} − ${shortLabel(cmpPathB)}`,
      offsetX + gridW / 2, 20
    );
  }

  function hitTest(clientX, clientY) {
    const rect = cvs.getBoundingClientRect();
    const st = cmpHeatmapState;
    // Map CSS pixels to our coordinate system (canvas buffer may differ)
    const scaleX = (cvs.width / (window.devicePixelRatio || 1)) / rect.width;
    const scaleY = (cvs.height / (window.devicePixelRatio || 1)) / rect.height;
    const x = (clientX - rect.left) * scaleX;
    const y = (clientY - rect.top) * scaleY;
    const j = Math.floor((x - st.offsetX) / st.gridSize);
    const i = Math.floor((y - st.offsetY) / st.gridSize);
    if (i < 0 || i >= S || j < 0 || j >= S) return null;
    if (j > i || cmpFrozen[i][j]) return null;
    return { i, j };
  }

  const tip = document.getElementById('tooltip');

  function onMouseMove(ev) {
    const cell = hitTest(ev.clientX, ev.clientY);
    const st = cmpHeatmapState;
    if (!cell) {
      if (st.hoverCell) { st.hoverCell = null; tip.style.display = 'none'; drawHeatmap(); }
      return;
    }
    if (st.hoverCell && st.hoverCell.i === cell.i && st.hoverCell.j === cell.j) {
      tip.style.left = (ev.clientX + 14) + 'px';
      tip.style.top = (ev.clientY - 8) + 'px';
      return;
    }
    st.hoverCell = cell;
    const sA = cmpMatA[cell.i][cell.j], sB = cmpMatB[cell.i][cell.j];
    const diff = diffMat[cell.i][cell.j];
    const qText = (cmpSentences[cell.i].text || '').slice(0, 80);
    const kText = (cmpSentences[cell.j].text || '').slice(0, 80);
    tip.innerHTML =
      `<div style="font-weight:700">S${cell.i} ← S${cell.j}</div>` +
      `<div style="opacity:0.8;margin:3px 0"><b>Query:</b> ${qText}</div>` +
      `<div style="opacity:0.8;margin:3px 0"><b>Key:</b> ${kText}</div>` +
      `<div style="margin-top:4px">` +
      `<span style="color:${CMP_COLORS.maskA}">A: ${sA.toFixed(4)}</span> &middot; ` +
      `<span style="color:${CMP_COLORS.maskB}">B: ${sB.toFixed(4)}</span><br>` +
      `Diff (A−B): <b>${diff != null ? (diff > 0 ? '+' : '') + diff.toFixed(4) : 'frozen'}</b></div>`;
    tip.style.display = 'block';
    tip.style.left = (ev.clientX + 14) + 'px';
    tip.style.top = (ev.clientY - 8) + 'px';
    drawHeatmap();
  }

  function onMouseOut() {
    const st = cmpHeatmapState;
    if (st.hoverCell) { st.hoverCell = null; tip.style.display = 'none'; drawHeatmap(); }
  }

  function onClick(ev) {
    const cell = hitTest(ev.clientX, ev.clientY);
    if (!cell) return;
    const st = cmpHeatmapState;
    if (st.selectedCell && st.selectedCell.i === cell.i && st.selectedCell.j === cell.j) {
      st.selectedCell = null;
    } else {
      st.selectedCell = cell;
    }
    drawHeatmap();
    const idx = cmpLearnableEdges.findIndex(e => e.i === cell.i && e.j === cell.j);
    if (idx >= 0) {
      cmpSelectedPairIdx = idx;
      document.getElementById('cmp-detail-panel').style.display = 'block';
      renderPairDetail(idx);
    }
  }

  cvs.style.cursor = 'crosshair';
  cvs.addEventListener('mousemove', onMouseMove);
  cvs.addEventListener('mouseout', onMouseOut);
  cvs.addEventListener('click', onClick);

  // Redraw on container resize (e.g. detail panel opening/closing)
  const resizeObs = new ResizeObserver(() => drawHeatmap());
  resizeObs.observe(container);

  cmpHeatmapCleanup = function() {
    resizeObs.disconnect();
    cvs.removeEventListener('mousemove', onMouseMove);
    cvs.removeEventListener('mouseout', onMouseOut);
    cvs.removeEventListener('click', onClick);
    tip.style.display = 'none';
    cmpHeatmapState = null;
  };

  drawHeatmap();
}

// ══════════════════════════════════════════════════════════════
// View 3: Scatter plots (raw scores and rank-based)
// ══════════════════════════════════════════════════════════════
function renderScatter(container, mode) {
  const labelA = shortLabel(cmpPathA);
  const labelB = shortLabel(cmpPathB);
  const edges = cmpLearnableEdges;
  const n = edges.length;

  let xs, ys, xLabel, yLabel, titleText;
  let rhoLabel, rhoVal;

  if (mode === 'raw') {
    xs = edges.map(e => e.scoreA);
    ys = edges.map(e => e.scoreB);
    xLabel = `Score in A (${labelA})`;
    yLabel = `Score in B (${labelB})`;
    titleText = 'Score scatter (raw values)';
    const r = pearsonR(xs, ys);
    const rho = spearmanRho(xs, ys);
    rhoLabel = `Pearson r = ${r.toFixed(4)}, Spearman rho = ${rho.toFixed(4)}`;
  } else {
    const ranksA = rankArray(edges.map(e => e.scoreA));
    const ranksB = rankArray(edges.map(e => e.scoreB));
    xs = ranksA;
    ys = ranksB;
    xLabel = `Rank in A (${labelA})`;
    yLabel = `Rank in B (${labelB})`;
    titleText = 'Rank-based scatter';
    const rho = spearmanRho(edges.map(e => e.scoreA), edges.map(e => e.scoreB));
    rhoLabel = `Spearman rho = ${rho.toFixed(4)}`;
  }

  const hoverTexts = edges.map(e => {
    const sI = cmpSentences[e.i], sJ = cmpSentences[e.j];
    return `(${e.i},${e.j})<br>A: ${e.scoreA.toFixed(4)}<br>B: ${e.scoreB.toFixed(4)}<br>` +
      `S${e.i}: ${(sI.text || '').slice(0, 40)}<br>S${e.j}: ${(sJ.text || '').slice(0, 40)}`;
  });

  const absDiffs = edges.map(e => Math.abs(e.scoreA - e.scoreB));
  const maxDiff = d3.max(absDiffs) || 1;
  const colors = absDiffs.map(d => d / maxDiff);

  const minV = Math.min(d3.min(xs), d3.min(ys));
  const maxV = Math.max(d3.max(xs), d3.max(ys));
  const pad = (maxV - minV) * 0.05;

  const traces = [
    {
      x: [minV - pad, maxV + pad],
      y: [minV - pad, maxV + pad],
      mode: 'lines',
      line: { color: '#94a3b8', dash: 'dash', width: 1 },
      name: 'y = x',
      showlegend: true,
      hoverinfo: 'skip',
    },
    {
      x: xs, y: ys,
      mode: 'markers',
      marker: {
        size: 5,
        color: colors,
        colorscale: [[0, '#94a3b8'], [0.5, '#f59e0b'], [1, '#ef4444']],
        cmin: 0, cmax: 1,
        opacity: 0.7,
        colorbar: { title: { text: '|diff|', side: 'right' }, tickfont: { size: 9 }, len: 0.6 },
      },
      text: hoverTexts,
      hovertemplate: '%{text}<extra></extra>',
      name: 'Edges',
    },
  ];

  Plotly.newPlot(container, traces, {
    ...PLOTLY_LAYOUT_BASE,
    height: 500,
    xaxis: { title: xLabel, gridcolor: '#e2e8f0' },
    yaxis: { title: yLabel, gridcolor: '#e2e8f0', scaleanchor: 'x', scaleratio: 1 },
    title: { text: `${titleText}<br><span style="font-size:10px;color:#94a3b8">${rhoLabel}</span>`, font: { size: 13 } },
  }, PLOTLY_CONFIG);
}

// ══════════════════════════════════════════════════════════════
// View 4: Column or row marginal differences (mode = 'col'|'row')
// ══════════════════════════════════════════════════════════════
function renderMarginals(container, mode) {
  const S = cmpSentences.length;
  const labelA = shortLabel(cmpPathA);
  const labelB = shortLabel(cmpPathB);

  const sumA = Array(S).fill(0), sumB = Array(S).fill(0);
  const counts = Array(S).fill(0);

  for (const e of cmpLearnableEdges) {
    const k = mode === 'col' ? e.j : e.i;
    sumA[k] += e.scoreA; sumB[k] += e.scoreB;
    counts[k]++;
  }

  const meanA = sumA.map((s, i) => counts[i] ? s / counts[i] : 0);
  const meanB = sumB.map((s, i) => counts[i] ? s / counts[i] : 0);
  const diff = meanA.map((a, i) => a - meanB[i]);

  const indices = d3.range(S);
  const sentLabels = cmpSentences.map((s, i) => `S${i}: ${(s.text || '').slice(0, 50)}`);

  const isCol = mode === 'col';
  const axisLabel = isCol ? 'Key sentence j' : 'Query sentence i';
  const titleBase = isCol ? 'Column marginal (mean score per key sentence)'
                          : 'Row marginal (mean score per query sentence)';
  const titleDiff = isCol ? 'Column marginal difference (A − B)'
                          : 'Row marginal difference (A − B)';
  const hoverKind = isCol ? 'col' : 'row';

  const abTraces = [
    {
      x: indices, y: meanA, type: 'bar', name: `A (${labelA})`,
      marker: { color: CMP_COLORS.maskA, opacity: 0.6 },
      text: sentLabels,
      hovertemplate: `%{text}<br>Mean ${hoverKind} score: %{y:.4f}<extra>A</extra>`,
    },
    {
      x: indices, y: meanB, type: 'bar', name: `B (${labelB})`,
      marker: { color: CMP_COLORS.maskB, opacity: 0.6 },
      text: sentLabels,
      hovertemplate: `%{text}<br>Mean ${hoverKind} score: %{y:.4f}<extra>B</extra>`,
    },
  ];

  const diffTraces = [{
    x: indices, y: diff, type: 'bar',
    marker: { color: diff.map(d => d > 0 ? CMP_COLORS.diff_pos : CMP_COLORS.diff_neg) },
    text: sentLabels,
    hovertemplate: `%{text}<br>${hoverKind} diff (A−B): %{y:.4f}<extra></extra>`,
    showlegend: false,
  }];

  container.innerHTML =
    '<div id="cmp-marginal-ab" style="min-height:300px;"></div>' +
    '<div id="cmp-marginal-diff" style="min-height:220px;"></div>';

  const layoutBase = { ...PLOTLY_LAYOUT_BASE, barmode: 'group' };

  Plotly.newPlot('cmp-marginal-ab', abTraces, {
    ...layoutBase, height: 300,
    title: { text: titleBase, font: { size: 12 } },
    xaxis: { title: axisLabel, dtick: 5 }, yaxis: { title: 'Mean score' },
  }, PLOTLY_CONFIG);

  Plotly.newPlot('cmp-marginal-diff', diffTraces, {
    ...layoutBase, height: 220,
    title: { text: titleDiff, font: { size: 11 } },
    xaxis: { title: axisLabel, dtick: 5 }, yaxis: { title: 'A − B' },
  }, PLOTLY_CONFIG);
}

// ══════════════════════════════════════════════════════════════
// View 5: Overlap at matched sparsity (Jaccard curve)
// ══════════════════════════════════════════════════════════════
function renderOverlap(container) {
  const n = cmpLearnableEdges.length;
  const labelA = shortLabel(cmpPathA);
  const labelB = shortLabel(cmpPathB);

  const rankedA = cmpLearnableEdges.map((e, idx) => ({ idx, score: e.scoreA }))
    .sort((a, b) => b.score - a.score);
  const rankedB = cmpLearnableEdges.map((e, idx) => ({ idx, score: e.scoreB }))
    .sort((a, b) => b.score - a.score);

  const sparsities = [];
  const jaccards = [];
  const intersectionSizes = [];
  const keepSizes = [];

  const steps = 50;
  for (let s = 1; s <= steps; s++) {
    const sparsity = s / steps;
    const keepK = Math.max(1, Math.round(n * (1 - sparsity)));

    const setA = new Set(rankedA.slice(0, keepK).map(e => e.idx));
    const setB = new Set(rankedB.slice(0, keepK).map(e => e.idx));

    let intersection = 0;
    for (const idx of setA) { if (setB.has(idx)) intersection++; }
    const union = setA.size + setB.size - intersection;
    const jaccard = union > 0 ? intersection / union : 1;

    sparsities.push(sparsity);
    jaccards.push(jaccard);
    intersectionSizes.push(intersection);
    keepSizes.push(keepK);
  }

  const traces = [
    {
      x: sparsities, y: jaccards,
      mode: 'lines+markers',
      marker: { size: 5, color: '#4caf50' },
      line: { color: '#4caf50', width: 2.5 },
      name: 'Jaccard index',
      hovertemplate: 'Sparsity: %{x:.0%}<br>Jaccard: %{y:.3f}<br>Keep: %{customdata[0]}<br>Shared: %{customdata[1]}<extra></extra>',
      customdata: keepSizes.map((k, i) => [k, intersectionSizes[i]]),
    },
    {
      x: sparsities,
      y: intersectionSizes.map((inter, i) => keepSizes[i] > 0 ? inter / keepSizes[i] : 0),
      mode: 'lines+markers',
      marker: { size: 4, color: '#2196f3', symbol: 'diamond' },
      line: { color: '#2196f3', width: 1.5, dash: 'dot' },
      name: 'Overlap fraction (|A∩B| / K)',
      hovertemplate: 'Sparsity: %{x:.0%}<br>Overlap: %{y:.3f}<extra></extra>',
    },
  ];

  Plotly.newPlot(container, traces, {
    ...PLOTLY_LAYOUT_BASE,
    height: 400,
    xaxis: { title: 'Sparsity (fraction of edges ablated)', tickformat: '.0%', gridcolor: '#e2e8f0' },
    yaxis: { title: 'Overlap', gridcolor: '#e2e8f0', range: [0, 1.05] },
    title: {
      text: `Top-K overlap: ${labelA} vs ${labelB}`,
      font: { size: 13 },
    },
  }, PLOTLY_CONFIG);
}
