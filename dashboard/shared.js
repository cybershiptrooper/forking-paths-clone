// shared.js — Circuit Tracer shared visualization code
// Used by both index.html (global outcomes) and direct_answer.html (direct-answer probes).
// Pages define pageInitEval() and pageOnReprocess() hooks that this code calls.

// ══════════════════════════════════════════════════════════════
// State
// ══════════════════════════════════════════════════════════════
let maskData = null;
let processed = null;
let selectedNode = null;
let highlightSentence = null;
let viewMode = 'arcs';
let aggState = null;
let aggHoverCell = null;
let aggSelectedCell = null;
let zoomTransform = d3.zoomIdentity;
let thresholdValues = [];

// DOM refs (initialized by initShared)
let $gc, $empty, $tip;
let canvas, ctx, svgRoot;
let arcCanvas, arcCtx;  // second canvas for arcs (drawn above nodes)

const MARGIN = { top: 36, right: 30, bottom: 70, left: 54 };
const REGION_COLORS = {
  prompt:     'rgba(59,130,246,0.07)',
  assistant:  'rgba(168,85,247,0.07)',
  generation: 'rgba(245,158,11,0.07)',
};
const REGION_BORDER_COLORS = {
  prompt:     'rgba(59,130,246,0.25)',
  assistant:  'rgba(168,85,247,0.25)',
  generation: 'rgba(245,158,11,0.25)',
};

const TAG_COLORS = {
  problem_setup:        '#3b82f6',
  plan_generation:      '#8b5cf6',
  fact_retrieval:       '#06b6d4',
  active_computation:   '#f59e0b',
  uncertainty_management:'#ef4444',
  result_consolidation: '#10b981',
  self_checking:        '#f97316',
  final_answer_emission:'#ec4899',
  unknown:              '#94a3b8',
};

const PLOTLY_LAYOUT_BASE = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: '#f8fafc',
  font: { family: '-apple-system, BlinkMacSystemFont, sans-serif', size: 11, color: '#64748b' },
  margin: { l: 56, r: 16, t: 8, b: 52 },
  legend: { orientation: 'h', y: 1.12, x: 0, font: { size: 10 } },
  hovermode: 'closest',
};
const PLOTLY_CONFIG = {
  responsive: true,
  displayModeBar: true,
  modeBarButtonsToRemove: ['sendDataToCloud', 'lasso2d', 'select2d'],
  displaylogo: false,
};

// ══════════════════════════════════════════════════════════════
// Color scales
// ══════════════════════════════════════════════════════════════
function edgeColor(norm, isNegative) {
  if (isNegative) return d3.interpolatePurples(0.3 + Math.abs(norm) * 0.55);
  return d3.interpolateGreens(0.25 + norm * 0.65);
}

function primaryTag(sentIdx) {
  const s = (maskData.sentences || [])[sentIdx];
  if (!s || !s.function_tags || !s.function_tags.length) return 'unknown';
  return s.function_tags[0];
}

function categoryEdgeColor(srcIdx, tgtIdx, norm) {
  const srcTag = primaryTag(srcIdx);
  const tgtTag = primaryTag(tgtIdx);
  let base;
  if (srcTag === tgtTag) {
    base = TAG_COLORS[srcTag] || TAG_COLORS.unknown;
  } else {
    const c1 = d3.color(TAG_COLORS[srcTag] || TAG_COLORS.unknown);
    const c2 = d3.color(TAG_COLORS[tgtTag] || TAG_COLORS.unknown);
    base = d3.interpolateRgb(c1, c2)(0.5);
  }
  return d3.interpolateRgb('#ffffff', base)(0.25 + norm * 0.65);
}

function colorByCategory() {
  const cb = document.getElementById('color-by-category');
  return cb && cb.checked;
}

// ══════════════════════════════════════════════════════════════
// Threshold controls
// ══════════════════════════════════════════════════════════════
function getThreshold() {
  if (thresholdValues.length === 0) return 0;
  const idx = parseInt(document.getElementById('threshold-slider').value) || 0;
  return thresholdValues[Math.min(Math.max(idx, 0), thresholdValues.length - 1)];
}
function nearestThresholdIdx(v) {
  let best = 0, bestD = Infinity;
  for (let i = 0; i < thresholdValues.length; i++) {
    const d = Math.abs(thresholdValues[i] - v);
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}
function syncThresholdFromSlider() {
  const t = getThreshold();
  document.getElementById('threshold-input').value = t.toExponential(3);
  const idx = parseInt(document.getElementById('threshold-slider').value) || 0;
  document.getElementById('threshold-idx-label').textContent = `${idx + 1}/${thresholdValues.length}`;
}
function setThresholdToValue(v) {
  if (thresholdValues.length === 0) return;
  const idx = nearestThresholdIdx(v);
  document.getElementById('threshold-slider').value = idx;
  syncThresholdFromSlider();
}

// ══════════════════════════════════════════════════════════════
// View toggle + reprocess
// ══════════════════════════════════════════════════════════════
function setView(mode) {
  viewMode = mode;
  aggSelectedCell = null;
  selectedNode = null;
  document.getElementById('view-arcs').classList.toggle('active', mode==='arcs');
  document.getElementById('view-flow').classList.toggle('active', mode==='flow');
  document.getElementById('view-agg').classList.toggle('active', mode==='agg');
  reprocess();
}

function reprocess() {
  if (!maskData) return;
  syncThresholdFromSlider();
  document.getElementById('influence-value').textContent =
    document.getElementById('influence-slider').value;
  processData(); render();
  if (typeof pageOnReprocess === 'function') pageOnReprocess();
}

// ══════════════════════════════════════════════════════════════
// Boot (shared part)
// ══════════════════════════════════════════════════════════════
function boot() {
  $empty.style.display = 'none';
  selectedNode = null; highlightSentence = null;

  const hasTags = (maskData.sentences || []).some(s => s.function_tags && s.function_tags.length);
  const catLabel = document.getElementById('color-by-cat-label');
  if (catLabel) catLabel.style.display = hasTags ? '' : 'none';

  const meta = maskData.metadata || {};
  const g = meta.mask_granularity || 'head';
  document.getElementById('meta-info').innerHTML =
    `<div>Model: <b>${maskData.model_name||'?'}</b></div>` +
    `<div>Algorithm: <b>${maskData.algorithm||'?'}</b></div>` +
    `<div>Granularity: <b>${g}</b> &middot; Layers: <b>${(maskData.layers||[]).length}</b> &middot; Sentences: <b>${(maskData.sentences||[]).length}</b></div>`;

  const layers = (maskData.layers||[]).sort((a,b)=>a-b);
  for (const id of ['layer-start','layer-end']) {
    const sel = document.getElementById(id); sel.innerHTML = '';
    layers.forEach(l => { const o=document.createElement('option'); o.value=l; o.textContent='L'+l; sel.appendChild(o); });
  }
  document.getElementById('layer-end').value = layers[layers.length-1];

  const leg = document.getElementById('sentence-legend'); leg.innerHTML = '';
  const sents = maskData.sentences || [];
  const legRegions = computeSentenceRegions(sents);
  const regionLabelColors = { prompt: '#3b82f6', assistant: '#a855f7', generation: '#f59e0b' };
  sents.forEach((s,i) => {
    const t = (s.text||'').trim() || `Sentence ${i}`;
    const cat = i < legRegions.assistantIdx ? 'prompt' : i < legRegions.genStart ? 'assistant' : 'generation';
    const d = document.createElement('div');
    d.className = 'sleg'; d.dataset.idx = i;
    d.innerHTML = `<span class="sleg-i" style="color:${regionLabelColors[cat]}">S${i}</span><span class="sleg-t" title="${t.replace(/"/g,'&quot;')}">${t}</span>`;
    d.addEventListener('click', () => {
      highlightSentence = (highlightSentence === i) ? null : i;
      aggSelectedCell = null;
      leg.querySelectorAll('.sleg').forEach(el => el.classList.toggle('active', parseInt(el.dataset.idx)===highlightSentence));
      render();
    });
    leg.appendChild(d);
  });

  // Build discrete threshold list
  const te = meta.threshold_evaluation;
  if (Array.isArray(te) && te.length > 0) {
    const tSet = new Set();
    for (const entry of te) {
      if (entry && typeof entry.threshold === 'number') tSet.add(entry.threshold);
    }
    thresholdValues = [...tSet].sort((a, b) => a - b);
  } else {
    const all = getAllScores();
    if (all.length) {
      all.sort(d3.ascending);
      const percentiles = d3.range(0, 1.01, 0.02);
      const pSet = new Set([0]);
      for (const p of percentiles) {
        const v = d3.quantile(all, p);
        if (v !== undefined) pSet.add(v);
      }
      thresholdValues = [...pSet].sort((a, b) => a - b);
    } else {
      thresholdValues = [0];
    }
  }

  const sl = document.getElementById('threshold-slider');
  sl.min = 0;
  sl.max = thresholdValues.length - 1;
  sl.step = 1;
  sl.disabled = thresholdValues.length <= 1;
  const defaultIdx = Math.round((thresholdValues.length - 1) * 0.3);
  sl.value = defaultIdx;
  syncThresholdFromSlider();

  const granularity = (maskData.metadata || {}).mask_granularity || 'head';
  const isPair = granularity === 'pair';
  document.getElementById('layer-start').disabled = isPair;
  document.getElementById('layer-end').disabled = isPair;
  if (isPair) {
    document.getElementById('layer-start').title = 'Disabled: pair granularity shares scores across all layers';
    document.getElementById('layer-end').title = 'Disabled: pair granularity shares scores across all layers';
  } else {
    document.getElementById('layer-start').title = '';
    document.getElementById('layer-end').title = '';
  }
  const pairNote = document.getElementById('pair-note');
  if (pairNote) pairNote.style.display = isPair ? '' : 'none';

  processData(); render();
  if (typeof pageInitEval === 'function') pageInitEval();
}

// ══════════════════════════════════════════════════════════════
// Score access
// ══════════════════════════════════════════════════════════════

// Build a 2D boolean array where true = frozen (should be excluded from scores).
// Mirrors the Python build_combined_filter: gap | causal | prompt.
function _buildFrozenFilter() {
  const meta = maskData.metadata || {};
  const S = (maskData.sentences || []).length;
  const gap = meta.sentence_gap || 0;
  const numFrozenPrompt = meta.num_frozen_prompt_sentences || 0;
  if (S === 0) return null;
  const frozen = Array.from({length: S}, () => Array(S).fill(false));
  for (let i = 0; i < S; i++) {
    for (let j = 0; j < S; j++) {
      if (gap > 0 && Math.abs(i - j) < gap) frozen[i][j] = true;
      if (i === j) frozen[i][j] = true;
      if (j > i) frozen[i][j] = true;  // causal
      if (numFrozenPrompt > 0 && (i < numFrozenPrompt || j < numFrozenPrompt)) frozen[i][j] = true;
    }
  }
  return frozen;
}

function _isValidEdge(src, tgt, sentenceGap) {
  if (src === tgt) return false;
  if (tgt > src) return false;
  if (Math.abs(src - tgt) <= sentenceGap) return false;
  return true;
}

function _isValidEdgeWithFilter(src, tgt, sentenceGap, frozenFilter) {
  if (!_isValidEdge(src, tgt, sentenceGap)) return false;
  if (frozenFilter && frozenFilter[src] && frozenFilter[src][tgt]) return false;
  return true;
}

function getAllScores() {
  const meta = maskData.metadata || {};
  const g = meta.mask_granularity || 'head';
  const gap = meta.sentence_gap || 0;
  const raw = maskData.scores; const vs = [];
  const isSparse = meta.scores_format === 'sparse';
  const ff = _buildFrozenFilter();
  if (isSparse) {
    if (g === 'head') {
      for (const l in raw) for (const h in raw[l]) {
        for (const t of raw[l][h]) { if (Math.abs(t[2]) > 0 && _isValidEdgeWithFilter(t[0], t[1], gap, ff)) vs.push(t[2]); }
      }
    } else if (g === 'layer') {
      for (const l in raw) {
        for (const t of raw[l]) { if (Math.abs(t[2]) > 0 && _isValidEdgeWithFilter(t[0], t[1], gap, ff)) vs.push(t[2]); }
      }
    } else {
      for (const t of raw) { if (Math.abs(t[2]) > 0 && _isValidEdgeWithFilter(t[0], t[1], gap, ff)) vs.push(t[2]); }
    }
  } else {
    if (g==='head') {
      for (const l in raw) for (const h in raw[l]) {
        const m = raw[l][h];
        for (let i=0;i<m.length;i++) for (let j=0;j<m[i].length;j++) { if (Math.abs(m[i][j])>0 && _isValidEdgeWithFilter(i,j,gap,ff)) vs.push(m[i][j]); }
      }
    } else if (g==='layer') {
      for (const l in raw) {
        const m = raw[l];
        for (let i=0;i<m.length;i++) for (let j=0;j<m[i].length;j++) { if (Math.abs(m[i][j])>0 && _isValidEdgeWithFilter(i,j,gap,ff)) vs.push(m[i][j]); }
      }
    } else {
      for (let i=0;i<raw.length;i++) for (let j=0;j<raw[i].length;j++) { if (Math.abs(raw[i][j])>0 && _isValidEdgeWithFilter(i,j,gap,ff)) vs.push(raw[i][j]); }
    }
  }
  return vs;
}

function getLayerAgg(layer, agg) {
  const meta = maskData.metadata || {};
  const g = meta.mask_granularity || 'head';
  const raw = maskData.scores, ls = String(layer);
  const isSparse = meta.scores_format === 'sparse';
  const S = (maskData.sentences || []).length;

  if (isSparse) {
    const res = Array.from({length: S}, () => Array(S).fill(0));
    if (g === 'pair') {
      for (const t of raw) res[t[0]][t[1]] = t[2];
    } else if (g === 'layer') {
      const triples = raw[ls];
      if (triples) for (const t of triples) res[t[0]][t[1]] = t[2];
    } else {
      const heads = raw[ls]; if (!heads) return res;
      const hk = Object.keys(heads); if (!hk.length) return res;
      if (agg === 'max') {
        for (const h of hk) for (const t of heads[h]) res[t[0]][t[1]] = Math.max(res[t[0]][t[1]], t[2]);
      } else {
        for (const h of hk) for (const t of heads[h]) res[t[0]][t[1]] += t[2];
        if (agg === 'mean') { const n = hk.length; for (let i = 0; i < S; i++) for (let j = 0; j < S; j++) res[i][j] /= n; }
      }
    }
    return res;
  }

  if (g==='pair') return raw.map(r=>[...r]);
  if (g==='layer') return (raw[ls]||[]).map(r=>[...r]);
  const heads = raw[ls]; if (!heads) return [];
  const hk = Object.keys(heads); if (!hk.length) return [];
  const Sd = heads[hk[0]].length;
  const res = Array.from({length:Sd}, ()=>Array(Sd).fill(0));
  for (const h of hk) { const m=heads[h]; for(let i=0;i<Sd;i++) for(let j=0;j<Sd;j++) { if(agg==='max') res[i][j]=Math.max(res[i][j],m[i][j]); else res[i][j]+=m[i][j]; } }
  if (agg==='mean') { const n=hk.length; for(let i=0;i<Sd;i++) for(let j=0;j<Sd;j++) res[i][j]/=n; }
  return res;
}

// ══════════════════════════════════════════════════════════════
// Data processing
// ══════════════════════════════════════════════════════════════
function processData() {
  const threshold = getThreshold();
  const influencePct = parseInt(document.getElementById('influence-slider').value);
  const agg = document.getElementById('agg-select').value;
  const lStart = parseInt(document.getElementById('layer-start').value);
  const lEnd = parseInt(document.getElementById('layer-end').value);

  const allLayers = (maskData.layers||[]).sort((a,b)=>a-b);
  let layers = allLayers.filter(l=>l>=lStart && l<=lEnd);
  const sents = maskData.sentences || [];
  const S = sents.length;

  const granularity = (maskData.metadata || {}).mask_granularity || 'head';
  const isPair = granularity === 'pair';
  if (isPair && (viewMode === 'arcs' || viewMode === 'flow') && layers.length > 1) {
    layers = [layers[0]];
  }

  const sentenceGap = (maskData.metadata || {}).sentence_gap || 0;
  const frozenFilter = _buildFrozenFilter();

  let allEdges = [];
  const nodeImp = {};
  for (const layer of layers) {
    const mat = getLayerAgg(layer, agg);
    nodeImp[layer] = Array(S).fill(0);
    for (let src=0; src<S; src++) {
      for (let tgt=0; tgt<S; tgt++) {
        if (!_isValidEdgeWithFilter(src, tgt, sentenceGap, frozenFilter)) continue;
        const score = (mat[src]&&mat[src][tgt]!==undefined) ? mat[src][tgt] : 0;
        if (Math.abs(score) < 1e-12) continue;
        allEdges.push({layer,src,tgt,score});
      }
    }
  }

  allEdges.sort((a,b) => Math.abs(b.score)-Math.abs(a.score));
  const totalInfluence = allEdges.reduce((s,e)=>s+Math.abs(e.score),0);
  let cumulative = 0;
  let cutoffIdx = allEdges.length;
  for (let i=0; i<allEdges.length; i++) {
    cumulative += Math.abs(allEdges[i].score);
    if (cumulative >= totalInfluence * influencePct/100) { cutoffIdx = i+1; break; }
  }
  allEdges = allEdges.slice(0, cutoffIdx);

  const edges = allEdges.filter(e => e.score >= threshold);

  let globalMax = 0;
  for (const e of edges) {
    globalMax = Math.max(globalMax, Math.abs(e.score));
    nodeImp[e.layer][e.src] += Math.abs(e.score);
    nodeImp[e.layer][e.tgt] += Math.abs(e.score);
  }
  let maxImp = 0;
  for (const l of layers) maxImp = Math.max(maxImp, ...nodeImp[l]);

  const regions = computeSentenceRegions(sents);

  processed = { layers, sents, edges, nodeImp, globalMax, maxImp, S, regions };
  document.getElementById('edge-badge').textContent = edges.length + ' edges';
  document.getElementById('node-badge').textContent = (layers.length*S) + ' nodes';
}

const ASSISTANT_TOKENS = ['<｜assistant｜>', '<|im_start|>assistant', '<|start_header_id|>assistant', '<|assistant|>', 'assistant\n'];
function computeSentenceRegions(sents) {
  let assistantIdx = sents.length;
  for (let i = 0; i < sents.length; i++) {
    const text = (sents[i].text || '').toLowerCase();
    if (ASSISTANT_TOKENS.some(tok => text.includes(tok.toLowerCase()))) {
      assistantIdx = i;
      break;
    }
  }
  const meta = maskData.metadata || {};
  const numPrefix = meta.num_prefix_sentences;
  const genStart = (typeof numPrefix === 'number' && numPrefix < sents.length) ? numPrefix : sents.length;
  return { assistantIdx, genStart };
}

// ══════════════════════════════════════════════════════════════
// Rendering
// ══════════════════════════════════════════════════════════════
function _makeCanvas(W, H) {
  const c = document.createElement('canvas');
  c.width = W * devicePixelRatio; c.height = H * devicePixelRatio;
  c.style.width = W + 'px'; c.style.height = H + 'px';
  const cx = c.getContext('2d'); cx.scale(devicePixelRatio, devicePixelRatio);
  return { el: c, ctx: cx };
}

function render() {
  if (!processed) return;
  const W = $gc.clientWidth, H = $gc.clientHeight;

  // Remove old layers
  $gc.querySelectorAll('canvas,svg').forEach(el => el.remove());

  // Layer stack (bottom to top):
  //   1. canvas  — background: regions, grid
  //   2. svg     — nodes, axes, labels (+ invisible hit areas for interaction)
  //   3. arcCanvas — arcs/flow lines (drawn ABOVE nodes so connections are visible)
  const bg = _makeCanvas(W, H);
  canvas = bg.el; ctx = bg.ctx;
  $gc.appendChild(canvas);

  svgRoot = d3.select($gc).append('svg').attr('width', W).attr('height', H);
  const gAll = svgRoot.append('g');

  const arc = _makeCanvas(W, H);
  arcCanvas = arc.el; arcCtx = arc.ctx;
  arcCanvas.style.pointerEvents = 'none';
  $gc.appendChild(arcCanvas);

  const { layers, sents, edges, nodeImp, globalMax, maxImp, S } = processed;
  if (!layers.length || !S) return;

  const xScale = d3.scalePoint().domain(d3.range(S)).range([MARGIN.left, W - MARGIN.right]).padding(0.5);
  const yScale = d3.scalePoint().domain(layers).range([H - MARGIN.bottom, MARGIN.top]).padding(0.5);
  const showLabels = document.getElementById('show-labels').checked;

  const { regions } = processed;
  if (viewMode === 'agg') {
    aggHoverCell = null;
    // Agg heatmap uses only the bottom canvas
    drawAggregatedMask(ctx, W, H);
    setupAggMouseEvents(canvas, ctx, W, H);
    updateAggDetail();
    return;
  }

  // Background canvas: regions + grid
  drawRegions(xScale, yScale, layers, S, regions, W, H);
  drawGrid(xScale, yScale, layers, S, W, H);

  // SVG: nodes, axes, labels
  drawNodes(gAll, xScale, yScale, layers, S, edges, nodeImp, maxImp, showLabels, sents);
  drawAxes(gAll, xScale, yScale, layers, S, sents, W, H);
  drawRegionLabels(gAll, xScale, regions, S);

  // Arc canvas (on top): arcs or flow lines
  if (viewMode === 'arcs') drawArcs(xScale, yScale, edges, globalMax, W, H, arcCtx);
  else drawFlow(xScale, yScale, edges, layers, globalMax, S, W, H, arcCtx);

  // Zoom: SVG captures events, transforms all three layers
  const zoom = d3.zoom().scaleExtent([0.3, 6]).on('zoom', ev => {
    zoomTransform = ev.transform;
    gAll.attr('transform', ev.transform);
    // Redraw background canvas
    ctx.clearRect(0, 0, W, H); ctx.save();
    ctx.translate(zoomTransform.x, zoomTransform.y);
    ctx.scale(zoomTransform.k, zoomTransform.k);
    drawRegions(xScale, yScale, layers, S, regions, W, H);
    drawGrid(xScale, yScale, layers, S, W, H);
    ctx.restore();
    // Redraw arc canvas
    arcCtx.clearRect(0, 0, W, H); arcCtx.save();
    arcCtx.translate(zoomTransform.x, zoomTransform.y);
    arcCtx.scale(zoomTransform.k, zoomTransform.k);
    if (viewMode === 'arcs') drawArcs(xScale, yScale, edges, globalMax, W, H, arcCtx);
    else drawFlow(xScale, yScale, edges, layers, globalMax, S, W, H, arcCtx);
    arcCtx.restore();
  });
  svgRoot.call(zoom).style('pointer-events', 'all');
}

// ══════════════════════════════════════════════════════════════
// Drawing helpers
// ══════════════════════════════════════════════════════════════
function drawRegions(xScale, yScale, layers, S, regions, W, H) {
  if (!regions || S === 0) return;
  ctx.save();
  const step = xScale.step ? xScale.step() : (W - MARGIN.left - MARGIN.right) / S;
  const halfStep = step / 2;
  const top = MARGIN.top - 14;
  const bot = H - MARGIN.bottom + 14;

  const spans = [];
  if (regions.assistantIdx > 0)
    spans.push({ start: 0, end: regions.assistantIdx, fill: REGION_COLORS.prompt, border: REGION_BORDER_COLORS.prompt });
  if (regions.assistantIdx < regions.genStart)
    spans.push({ start: regions.assistantIdx, end: regions.genStart, fill: REGION_COLORS.assistant, border: REGION_BORDER_COLORS.assistant });
  if (regions.genStart < S)
    spans.push({ start: regions.genStart, end: S, fill: REGION_COLORS.generation, border: REGION_BORDER_COLORS.generation });

  for (const span of spans) {
    const x0 = xScale(span.start) - halfStep;
    const x1 = xScale(span.end - 1) + halfStep;
    ctx.fillStyle = span.fill;
    ctx.fillRect(x0, top, x1 - x0, bot - top);
    ctx.strokeStyle = span.border;
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(x0, top); ctx.lineTo(x0, bot); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x1, top); ctx.lineTo(x1, bot); ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.restore();
}

function drawRegionLabels(g, xScale, regions, S) {
  if (!regions || S === 0) return;
  const step = xScale.step ? xScale.step() : 50;
  const halfStep = step / 2;
  const y = MARGIN.top - 18;

  const labels = [];
  if (regions.assistantIdx > 0)
    labels.push({ start: 0, end: regions.assistantIdx, text: 'Prompt', color: 'rgba(59,130,246,0.7)' });
  if (regions.assistantIdx < regions.genStart)
    labels.push({ start: regions.assistantIdx, end: regions.genStart, text: 'Assistant Prefix', color: 'rgba(168,85,247,0.7)' });
  if (regions.genStart < S)
    labels.push({ start: regions.genStart, end: S, text: 'Generation', color: 'rgba(245,158,11,0.7)' });

  for (const lbl of labels) {
    const x0 = xScale(lbl.start) - halfStep;
    const x1 = xScale(lbl.end - 1) + halfStep;
    const cx = (x0 + x1) / 2;
    g.append('text')
      .attr('x', cx).attr('y', y)
      .attr('text-anchor', 'middle')
      .attr('font-size', 9).attr('font-weight', 600)
      .attr('fill', lbl.color)
      .text(lbl.text);
  }
}

function drawGrid(xScale, yScale, layers, S, W, H) {
  ctx.save();
  ctx.strokeStyle = '#fff'; ctx.lineWidth = 1;
  for (const l of layers) { const y=yScale(l); ctx.beginPath(); ctx.moveTo(MARGIN.left-8,y); ctx.lineTo(W-MARGIN.right+8,y); ctx.stroke(); }
  for (let s=0;s<S;s++) { const x=xScale(s); ctx.beginPath(); ctx.moveTo(x,MARGIN.top-8); ctx.lineTo(x,H-MARGIN.bottom+8); ctx.stroke(); }
  ctx.restore();
}

function isEdgeDimmed(edge) {
  if (selectedNode) {
    if (viewMode === 'arcs') {
      return !(edge.layer===selectedNode.layer && (edge.src===selectedNode.sentence || edge.tgt===selectedNode.sentence));
    } else {
      return !(edge.src===selectedNode.sentence || edge.tgt===selectedNode.sentence);
    }
  }
  if (highlightSentence !== null) {
    return !(edge.src===highlightSentence || edge.tgt===highlightSentence);
  }
  return false;
}

function drawArcs(xScale, yScale, edges, gMax, W, H, c) {
  if (!c) c = ctx;
  const useCat = colorByCategory();
  const sorted = [...edges].sort((a,b) => Math.abs(a.score)-Math.abs(b.score));
  for (const e of sorted) {
    const x1=xScale(e.tgt), x2=xScale(e.src), y=yScale(e.layer);
    const norm = gMax>0 ? Math.abs(e.score)/gMax : 0;
    const sqrtNorm = Math.sqrt(norm);
    const width = 0.5 + sqrtNorm*4;
    let opacity = 0.12 + sqrtNorm*0.7;
    const dimmed = isEdgeDimmed(e);
    if (dimmed) opacity *= 0.06;

    const dx = x2-x1;
    const bulge = -(18 + Math.abs(dx)*0.35);
    c.beginPath(); c.moveTo(x1,y);
    c.bezierCurveTo(x1+dx*0.3, y+bulge, x1+dx*0.7, y+bulge, x2, y);

    const color = useCat ? categoryEdgeColor(e.src, e.tgt, norm) : edgeColor(norm, e.score<0);
    c.strokeStyle = color; c.globalAlpha = opacity;
    c.lineWidth = width; c.stroke();

    const as = 3+width;
    const t=0.96;
    const bx1=x1+dx*0.3, bx2=x1+dx*0.7, by1=y+bulge, by2=y+bulge;
    const tx = 3*(1-t)*(1-t)*(bx1-x1)+6*(1-t)*t*(bx2-bx1)+3*t*t*(x2-bx2);
    const ty = 3*(1-t)*(1-t)*(by1-y)+6*(1-t)*t*(by2-by1)+3*t*t*(y-by2);
    const ang = Math.atan2(ty,tx);
    c.beginPath(); c.moveTo(x2,y);
    c.lineTo(x2-as*Math.cos(ang-0.4), y-as*Math.sin(ang-0.4));
    c.lineTo(x2-as*Math.cos(ang+0.4), y-as*Math.sin(ang+0.4));
    c.closePath(); c.fillStyle=color; c.fill();
  }
  c.globalAlpha=1;
}

function drawFlow(xScale, yScale, edges, layers, gMax, S, W, H, c) {
  if (!c) c = ctx;
  const useCat = colorByCategory();
  const pairMap = {};
  for (const e of edges) {
    const key = e.src+','+e.tgt;
    if (!pairMap[key]) pairMap[key] = [];
    pairMap[key].push(e);
  }

  for (const key in pairMap) {
    const group = pairMap[key].sort((a,b)=>a.layer-b.layer);
    const [src,tgt] = key.split(',').map(Number);

    for (let i=0; i<group.length-1; i++) {
      const e1 = group[i], e2 = group[i+1];
      const x1=xScale(tgt), y1=yScale(e1.layer);
      const x2=xScale(src), y2=yScale(e2.layer);

      const avgScore = (Math.abs(e1.score)+Math.abs(e2.score))/2;
      const norm = gMax>0 ? avgScore/gMax : 0;
      const sqrtNorm = Math.sqrt(norm);
      const width = 0.4 + sqrtNorm*3.5;
      let opacity = 0.1 + sqrtNorm*0.6;
      const dimmed = isEdgeDimmed(e1) && isEdgeDimmed(e2);
      if (dimmed) opacity *= 0.05;

      const color = useCat ? categoryEdgeColor(src, tgt, norm) : edgeColor(norm, avgScore<0);
      c.strokeStyle = color; c.globalAlpha = opacity; c.lineWidth = width;

      const midY = (y1+y2)/2;
      c.beginPath(); c.moveTo(x1,y1);
      c.bezierCurveTo(x1, midY, x2, midY, x2, y2);
      c.stroke();
    }

    for (const e of group) {
      const norm = gMax>0 ? Math.abs(e.score)/gMax : 0;
      const dimmed = isEdgeDimmed(e);
      const r = 2 + Math.sqrt(norm)*3;
      const opacity = dimmed ? 0.05 : 0.15+norm*0.6;
      c.globalAlpha = opacity;
      const dotColor = useCat ? categoryEdgeColor(e.src, e.tgt, norm) : edgeColor(norm, e.score<0);
      c.beginPath();
      c.arc(xScale(e.tgt), yScale(e.layer), r, 0, Math.PI*2);
      c.fillStyle = dotColor; c.fill();
      c.beginPath();
      c.arc(xScale(e.src), yScale(e.layer), r, 0, Math.PI*2);
      c.fill();
    }
  }
  c.globalAlpha = 1;
}

function drawNodes(g, xScale, yScale, layers, S, edges, nodeImp, maxImp, showLabels, sents) {
  for (const layer of layers) {
    for (let s=0; s<S; s++) {
      const x=xScale(s), y=yScale(layer);
      const imp = nodeImp[layer]?nodeImp[layer][s]:0;
      const normImp = maxImp>0 ? imp/maxImp : 0;

      const isSel = selectedNode && selectedNode.layer===layer && selectedNode.sentence===s;
      const isHighlight = highlightSentence===s;
      const isConn = (selectedNode || highlightSentence!==null) && edges.some(e =>
        (viewMode==='arcs' ? e.layer===layer : true) &&
        (e.src===s || e.tgt===s) &&
        (selectedNode ? (e.src===selectedNode.sentence || e.tgt===selectedNode.sentence) &&
          (viewMode==='arcs' ? e.layer===selectedNode.layer : true) :
          (e.src===highlightSentence || e.tgt===highlightSentence))
      );

      const ng = g.append('g').attr('transform', `translate(${x},${y})`).datum({layer,sentence:s,importance:imp});
      const r = normImp > 0.01 ? 5 + normImp*9 : 5;
      ng.append('circle').attr('r', r)
        .attr('fill', isSel ? 'rgba(240,0,255,0.12)' : isConn||isHighlight ? 'rgba(76,175,80,0.15)' : '#fff')
        .attr('stroke', isSel ? '#f0f' : isConn||isHighlight ? '#4caf50' : normImp>0.02 ? '#94a3b8' : '#cbd5e1')
        .attr('stroke-width', isSel ? 2 : isConn ? 1.5 : 0.8)
        .attr('stroke-dasharray', isSel ? '3 2' : 'none');

      if (normImp > 0.01) {
        const ir = 2 + normImp*4;
        ng.append('circle').attr('r', ir)
          .attr('fill', isSel ? '#f0f' : isConn||isHighlight ? '#4caf50' : '#64748b')
          .attr('fill-opacity', 0.25+normImp*0.75);
      }

      ng.append('circle').attr('class','node-hit').attr('r', Math.max(r+4, 10))
        .attr('fill','transparent')
        .on('mouseover', function(ev) {
          const d=d3.select(this.parentNode).datum();
          showTip(ev,d);
          document.getElementById('hover-info').textContent = `L${d.layer} S${d.sentence} imp=${d.importance.toFixed(4)}`;
        })
        .on('mousemove', function(ev) { moveTip(ev); })
        .on('mouseout', function() { hideTip(); document.getElementById('hover-info').textContent=''; })
        .on('click', function(ev) {
          const d=d3.select(this.parentNode).datum();
          if (selectedNode && selectedNode.layer===d.layer && selectedNode.sentence===d.sentence) selectedNode=null;
          else selectedNode=d;
          highlightSentence = null;
          document.querySelectorAll('.sleg').forEach(el=>el.classList.remove('active'));
          render(); updateDetail();
        });

      if (showLabels && normImp > 0.01) {
        ng.append('text').attr('text-anchor','middle').attr('dominant-baseline','middle')
          .attr('font-size', 8).attr('font-weight',600)
          .attr('fill', isSel||isConn ? '#1e293b' : '#94a3b8')
          .attr('dy', r+10).text('S'+s);
      }
    }
  }
}

function drawAxes(g, xScale, yScale, layers, S, sents, W, H) {
  for (const l of layers) {
    g.append('text').attr('x', MARGIN.left-12).attr('y', yScale(l))
      .attr('text-anchor','end').attr('dominant-baseline','middle')
      .attr('font-size',10).attr('fill','#64748b').attr('font-family','monospace')
      .text('L'+l);
  }
  for (let s=0; s<S; s++) {
    const t = (sents[s]&&sents[s].text) ? sents[s].text.trim() : `S${s}`;
    const tr = t.length>18 ? t.slice(0,16)+'…' : t;
    g.append('text')
      .attr('x', xScale(s)).attr('y', H-MARGIN.bottom+18)
      .attr('text-anchor','end').attr('dominant-baseline','hanging')
      .attr('font-size',10).attr('fill','#64748b')
      .attr('transform', `rotate(-40, ${xScale(s)}, ${H-MARGIN.bottom+18})`)
      .text(tr);
  }
}

// ══════════════════════════════════════════════════════════════
// Aggregated Mask Heatmap
// ══════════════════════════════════════════════════════════════
function drawAggregatedMask(c, W, H) {
  if (!processed) return;
  const { sents, S } = processed;
  const agg = document.getElementById('agg-select').value;
  const threshold = getThreshold();

  const lStart = parseInt(document.getElementById('layer-start').value);
  const lEnd = parseInt(document.getElementById('layer-end').value);
  const allLayers = (maskData.layers || []).sort((a, b) => a - b);
  const layers = allLayers.filter(l => l >= lStart && l <= lEnd);

  const aggMatrix = Array.from({ length: S }, () => Array(S).fill(0));
  const layerCount = layers.length;
  for (const layer of layers) {
    const mat = getLayerAgg(layer, agg);
    for (let i = 0; i < S; i++) {
      for (let j = 0; j < S; j++) {
        if (mat[i] && mat[i][j] !== undefined) {
          if (agg === 'max') aggMatrix[i][j] = Math.max(aggMatrix[i][j], mat[i][j]);
          else aggMatrix[i][j] += mat[i][j];
        }
      }
    }
  }
  if (agg === 'mean' && layerCount > 0) {
    for (let i = 0; i < S; i++) for (let j = 0; j < S; j++) aggMatrix[i][j] /= layerCount;
  }

  // Zero out frozen entries (gap, causal, prompt-freeze — all untrained scores)
  const frozenFilter = _buildFrozenFilter();
  for (let i = 0; i < S; i++) {
    for (let j = 0; j < S; j++) {
      if (frozenFilter && frozenFilter[i][j]) aggMatrix[i][j] = 0;
    }
  }

  let maxAbs = 0;
  for (let i = 0; i < S; i++) {
    for (let j = 0; j <= i; j++) {
      maxAbs = Math.max(maxAbs, Math.abs(aggMatrix[i][j]));
    }
  }
  if (maxAbs === 0) maxAbs = 1;

  const pad = { top: 36, right: 50, bottom: 36, left: 50 };
  const gridSize = Math.min((W - pad.left - pad.right) / S, (H - pad.top - pad.bottom) / S);
  const gridW = gridSize * S;
  const gridH = gridSize * S;
  const offsetX = pad.left + (W - pad.left - pad.right - gridW) / 2;
  const offsetY = pad.top + (H - pad.top - pad.bottom - gridH) / 2;

  aggState = { offsetX, offsetY, gridSize, S, aggMatrix, maxAbs };

  function cellColor(val) {
    const norm = Math.abs(val) / maxAbs;
    if (val < 0) return d3.interpolatePurples(0.15 + norm * 0.7);
    return d3.interpolateGreens(0.15 + norm * 0.7);
  }

  const hlRows = new Set();
  const hlCols = new Set();
  if (aggSelectedCell) {
    hlRows.add(aggSelectedCell.i);
    hlCols.add(aggSelectedCell.j);
  } else if (highlightSentence !== null) {
    hlRows.add(highlightSentence);
    hlCols.add(highlightSentence);
  }
  const hasHighlight = hlRows.size > 0 || hlCols.size > 0;

  for (let i = 0; i < S; i++) {
    for (let j = 0; j < S; j++) {
      const x = offsetX + j * gridSize;
      const y = offsetY + i * gridSize;
      const isHl = hlRows.has(i) || hlCols.has(j);

      if (j > i) {
        c.fillStyle = isHl ? '#e8f5e9' : '#f1f5f9';
        c.fillRect(x, y, gridSize, gridSize);
      } else {
        const val = aggMatrix[i][j];
        const belowThreshold = val < threshold;
        const useCat = colorByCategory();
        if (useCat && !belowThreshold) {
          const norm = maxAbs > 0 ? Math.abs(val) / maxAbs : 0;
          const keyColor = TAG_COLORS[primaryTag(j)] || TAG_COLORS.unknown;
          c.fillStyle = d3.interpolateRgb('#ffffff', keyColor)(0.15 + norm * 0.7);
          c.beginPath();
          c.moveTo(x, y); c.lineTo(x, y + gridSize); c.lineTo(x + gridSize, y + gridSize);
          c.closePath(); c.fill();
          const queryColor = TAG_COLORS[primaryTag(i)] || TAG_COLORS.unknown;
          c.fillStyle = d3.interpolateRgb('#ffffff', queryColor)(0.15 + norm * 0.7);
          c.beginPath();
          c.moveTo(x, y); c.lineTo(x + gridSize, y); c.lineTo(x + gridSize, y + gridSize);
          c.closePath(); c.fill();
          if (isHl) {
            c.fillStyle = 'rgba(76, 175, 80, 0.10)';
            c.fillRect(x, y, gridSize, gridSize);
          }
        } else {
          if (belowThreshold) {
            c.fillStyle = isHl ? '#cdd5dc' : '#e2e8f0';
          } else {
            c.fillStyle = cellColor(val);
            if (isHl) {
              c.fillRect(x, y, gridSize, gridSize);
              c.fillStyle = 'rgba(76, 175, 80, 0.10)';
            }
          }
          c.fillRect(x, y, gridSize, gridSize);
        }
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
  if (aggSelectedCell) {
    c.strokeRect(offsetX, offsetY + aggSelectedCell.i * gridSize, gridW, gridSize);
    if (aggSelectedCell.j !== aggSelectedCell.i) {
      c.strokeRect(offsetX + aggSelectedCell.j * gridSize, offsetY, gridSize, gridH);
    }
  } else if (highlightSentence !== null && highlightSentence < S) {
    c.strokeRect(offsetX, offsetY + highlightSentence * gridSize, gridW, gridSize);
    c.strokeRect(offsetX + highlightSentence * gridSize, offsetY, gridSize, gridH);
  }

  if (aggHoverCell && aggHoverCell.i < S && aggHoverCell.j < S) {
    const hx = offsetX + aggHoverCell.j * gridSize;
    const hy = offsetY + aggHoverCell.i * gridSize;
    c.strokeStyle = '#1e293b'; c.lineWidth = 2.5;
    c.strokeRect(hx, hy, gridSize, gridSize);
  }

  const fontSize = Math.min(10, gridSize * 0.6);
  for (let s = 0; s < S; s++) {
    const isHlLabel = hlRows.has(s) || hlCols.has(s);
    const isHv = aggHoverCell && (aggHoverCell.i === s || aggHoverCell.j === s);
    c.fillStyle = isHlLabel ? '#2e7d32' : isHv ? '#1e293b' : '#64748b';
    c.font = (isHlLabel || isHv ? 'bold ' : '') + fontSize + 'px monospace';
    c.textAlign = 'center';
    c.fillText('S' + s, offsetX + s * gridSize + gridSize / 2, offsetY + gridH + 14);
    c.textAlign = 'right';
    c.fillText('S' + s, offsetX - 4, offsetY + s * gridSize + gridSize / 2 + 3);
  }

  const legendX = offsetX + gridW + 10;
  const legendY = offsetY;
  const legendH = Math.min(gridH, 120);
  const legendW = 12;
  const steps = 20;
  for (let i = 0; i < steps; i++) {
    const frac = i / steps;
    const val = maxAbs * (1 - 2 * frac);
    const y = legendY + frac * legendH;
    c.fillStyle = val >= 0 ? d3.interpolateGreens(0.15 + (val / maxAbs) * 0.7)
                           : d3.interpolatePurples(0.15 + (Math.abs(val) / maxAbs) * 0.7);
    c.fillRect(legendX, y, legendW, legendH / steps + 1);
  }
  c.strokeStyle = '#94a3b8'; c.lineWidth = 0.5;
  c.strokeRect(legendX, legendY, legendW, legendH);

  c.fillStyle = '#64748b'; c.font = '8px monospace'; c.textAlign = 'left';
  c.fillText('+' + maxAbs.toExponential(1), legendX + legendW + 3, legendY + 6);
  c.fillText('0', legendX + legendW + 3, legendY + legendH / 2 + 3);
  c.fillText('-' + maxAbs.toExponential(1), legendX + legendW + 3, legendY + legendH);
}

// ══════════════════════════════════════════════════════════════
// Tooltip
// ══════════════════════════════════════════════════════════════
function showTip(ev, d) {
  const s = processed.sents;
  const text = (s[d.sentence]&&s[d.sentence].text) ? s[d.sentence].text.trim() : `Sentence ${d.sentence}`;
  const atTo = processed.edges.filter(e => (viewMode==='arcs' ? e.layer===d.layer : true) && e.src===d.sentence).length;
  const atBy = processed.edges.filter(e => (viewMode==='arcs' ? e.layer===d.layer : true) && e.tgt===d.sentence).length;
  $tip.innerHTML =
    `<div style="font-weight:700">S${d.sentence} @ Layer ${d.layer}</div>`+
    `<div style="opacity:0.7;margin:2px 0">${text.length>70?text.slice(0,68)+'…':text}</div>`+
    `<div>Importance: ${d.importance.toFixed(4)}</div>`+
    `<div>Attends to: ${atTo} &middot; Attended by: ${atBy}</div>`;
  $tip.style.display='block';
  moveTip(ev);
}
function moveTip(ev) { $tip.style.left=(ev.clientX+14)+'px'; $tip.style.top=(ev.clientY-8)+'px'; }
function hideTip() { $tip.style.display='none'; }

// ══════════════════════════════════════════════════════════════
// Aggregated mask mouse interactions
// ══════════════════════════════════════════════════════════════
function getAggCell(clientX, clientY, canvasEl) {
  if (!aggState) return null;
  const rect = canvasEl.getBoundingClientRect();
  const x = clientX - rect.left;
  const y = clientY - rect.top;
  const { offsetX, offsetY, gridSize, S } = aggState;
  const j = Math.floor((x - offsetX) / gridSize);
  const i = Math.floor((y - offsetY) / gridSize);
  if (i < 0 || i >= S || j < 0 || j >= S) return null;
  if (j > i) return null;
  return { i, j };
}

function showAggTip(ev, cell) {
  if (!aggState || !processed) return;
  const { aggMatrix } = aggState;
  const sents = processed.sents;
  const qText = (sents[cell.i] && sents[cell.i].text) ? sents[cell.i].text.trim() : `Sentence ${cell.i}`;
  const kText = (sents[cell.j] && sents[cell.j].text) ? sents[cell.j].text.trim() : `Sentence ${cell.j}`;
  const score = aggMatrix[cell.i][cell.j];
  const threshold = getThreshold();
  const belowT = score < threshold;
  $tip.innerHTML =
    `<div style="font-weight:700">S${cell.i} ← S${cell.j}</div>` +
    `<div style="opacity:0.8;margin:3px 0"><b>Query:</b> ${qText.length > 80 ? qText.slice(0, 78) + '…' : qText}</div>` +
    `<div style="opacity:0.8;margin:3px 0"><b>Key:</b> ${kText.length > 80 ? kText.slice(0, 78) + '…' : kText}</div>` +
    `<div style="margin-top:4px">Score: <b>${score.toExponential(3)}</b>${belowT ? ' <span style="opacity:0.6">(below threshold)</span>' : ''}</div>`;
  $tip.style.display = 'block';
  moveTip(ev);
}

function redrawAgg(ctxRef, W, H) {
  ctxRef.clearRect(0, 0, W, H);
  drawAggregatedMask(ctxRef, W, H);
}

function setupAggMouseEvents(canvasEl, ctxRef, W, H) {
  canvasEl.style.cursor = 'crosshair';

  canvasEl.addEventListener('mousemove', function(ev) {
    const cell = getAggCell(ev.clientX, ev.clientY, canvasEl);
    if (!cell) {
      if (aggHoverCell) { aggHoverCell = null; hideTip(); redrawAgg(ctxRef, W, H); }
      document.getElementById('hover-info').textContent = '';
      return;
    }
    if (aggHoverCell && aggHoverCell.i === cell.i && aggHoverCell.j === cell.j) { moveTip(ev); return; }
    aggHoverCell = cell;
    const score = aggState.aggMatrix[cell.i][cell.j];
    document.getElementById('hover-info').textContent = `S${cell.i}←S${cell.j} score=${score.toExponential(3)}`;
    showAggTip(ev, cell);
    redrawAgg(ctxRef, W, H);
  });

  canvasEl.addEventListener('mouseout', function() {
    if (aggHoverCell) {
      aggHoverCell = null; hideTip();
      document.getElementById('hover-info').textContent = '';
      redrawAgg(ctxRef, W, H);
    }
  });

  canvasEl.addEventListener('click', function(ev) {
    const cell = getAggCell(ev.clientX, ev.clientY, canvasEl);
    if (!cell) return;
    if (aggSelectedCell && aggSelectedCell.i === cell.i && aggSelectedCell.j === cell.j) {
      aggSelectedCell = null;
    } else {
      aggSelectedCell = cell;
    }
    highlightSentence = null;
    document.querySelectorAll('.sleg').forEach(function(el) { el.classList.remove('active'); });
    redrawAgg(ctxRef, W, H);
    updateAggDetail();
  });
}

// ══════════════════════════════════════════════════════════════
// Detail panel
// ══════════════════════════════════════════════════════════════
function classificationHtml(sent) {
  if (!sent || !sent.function_tags) return '';
  let h = '<div class="cls-tags">';
  for (const tag of sent.function_tags) {
    h += `<span class="cls-tag">${tag.replace(/_/g, ' ')}</span>`;
  }
  h += '</div>';
  if (sent.depends_on && sent.depends_on.length) {
    h += `<div class="cls-deps">Depends on: ${sent.depends_on.map(d => 'S' + d).join(', ')}</div>`;
  }
  return h;
}

function updateDetail() {
  const panel = document.getElementById('detail-content');
  if (!selectedNode) { panel.innerHTML='Click a node to inspect connections.'; return; }
  const {layer,sentence} = selectedNode;
  const s = processed.sents;
  const text = (s[sentence]&&s[sentence].text) ? s[sentence].text.trim() : `Sentence ${sentence}`;

  const filterLayer = viewMode==='arcs';
  const attendsTo = processed.edges
    .filter(e => (!filterLayer||e.layer===layer) && e.src===sentence)
    .sort((a,b)=>Math.abs(b.score)-Math.abs(a.score));
  const attendedBy = processed.edges
    .filter(e => (!filterLayer||e.layer===layer) && e.tgt===sentence)
    .sort((a,b)=>Math.abs(b.score)-Math.abs(a.score));

  let h = `<div style="font-weight:700;font-size:13px">S${sentence} @ Layer ${layer}</div>`;
  h += `<div style="font-size:10px;color:#64748b;margin:2px 0 4px">${text}</div>`;
  h += classificationHtml(s[sentence]);
  h += connSection('Attends to', attendsTo, 'tgt', s);
  h += connSection('Attended by', attendedBy, 'src', s);
  panel.innerHTML = h;
}

function connSection(title, list, field, sents) {
  let h = `<div style="font-weight:600;font-size:11px;margin:6px 0 3px">${title} (${list.length})</div>`;
  if (!list.length) return h;
  h += '<ul class="conn-list">';
  const gm = processed.globalMax||1;
  for (const e of list.slice(0,20)) {
    const idx = e[field];
    const t = (sents[idx]&&sents[idx].text)?sents[idx].text.trim():`S${idx}`;
    const norm = Math.abs(e.score)/gm;
    const barW = Math.round(50*norm);
    const col = e.score<0 ? '#9333ea' : '#4caf50';
    h += `<li><span>S${idx} L${e.layer}: ${t}</span>`;
    h += `<span class="mono">${e.score.toExponential(2)} <span class="bar" style="width:${barW}px;background:${col}"></span></span></li>`;
  }
  h += '</ul>';
  if (list.length>20) h += `<div style="font-size:10px;color:#94a3b8">…and ${list.length-20} more</div>`;
  return h;
}

function updateAggDetail() {
  const panel = document.getElementById('detail-content');
  if (!aggSelectedCell || !aggState || !processed) {
    panel.innerHTML = 'Click a cell to inspect sentence pair.';
    return;
  }
  const { i: qi, j: ki } = aggSelectedCell;
  const { aggMatrix, S } = aggState;
  const sents = processed.sents;
  const score = aggMatrix[qi][ki];
  const threshold = getThreshold();

  const qText = (sents[qi] && sents[qi].text) ? sents[qi].text.trim() : `Sentence ${qi}`;
  const kText = (sents[ki] && sents[ki].text) ? sents[ki].text.trim() : `Sentence ${ki}`;

  let h = `<div style="font-weight:700;font-size:13px">S${qi} ← S${ki}</div>`;
  h += `<div style="font-size:10px;color:#64748b;margin:2px 0">Score: <b>${score.toExponential(3)}</b></div>`;

  h += `<div style="font-weight:600;font-size:11px;margin:10px 0 3px;color:#2e7d32">Query: S${qi}</div>`;
  h += `<div style="font-size:10px;color:#64748b;margin:0 0 4px">${qText}</div>`;
  h += classificationHtml(sents[qi]);

  const qEdges = [];
  for (let j = 0; j <= qi && j < S; j++) {
    if (j === qi) continue;
    const s = aggMatrix[qi][j];
    if (s >= threshold && Math.abs(s) > 1e-12) qEdges.push({ idx: j, score: s });
  }
  qEdges.sort((a, b) => b.score - a.score);
  h += aggConnSection('Attends to', qEdges, sents, aggMatrix);

  h += `<div style="font-weight:600;font-size:11px;margin:10px 0 3px;color:#1565c0">Key: S${ki}</div>`;
  h += `<div style="font-size:10px;color:#64748b;margin:0 0 4px">${kText}</div>`;
  h += classificationHtml(sents[ki]);

  const kEdges = [];
  for (let i = ki; i < S; i++) {
    if (i === ki) continue;
    const s = aggMatrix[i][ki];
    if (s >= threshold && Math.abs(s) > 1e-12) kEdges.push({ idx: i, score: s });
  }
  kEdges.sort((a, b) => b.score - a.score);
  h += aggConnSection('Attended by', kEdges, sents, aggMatrix);

  panel.innerHTML = h;
}

function aggConnSection(title, edges, sents, aggMatrix) {
  let h = `<div style="font-weight:600;font-size:10px;margin:4px 0 2px">${title} (${edges.length})</div>`;
  if (!edges.length) return h;
  h += '<ul class="conn-list">';
  const maxScore = edges.length > 0 ? Math.max(...edges.map(e => Math.abs(e.score))) : 1;
  for (const e of edges.slice(0, 20)) {
    const t = (sents[e.idx] && sents[e.idx].text) ? sents[e.idx].text.trim() : `S${e.idx}`;
    const norm = maxScore > 0 ? Math.abs(e.score) / maxScore : 0;
    const barW = Math.round(50 * norm);
    const col = e.score < 0 ? '#9333ea' : '#4caf50';
    h += `<li><span>S${e.idx}: ${t}</span>`;
    h += `<span class="mono">${e.score.toExponential(2)} <span class="bar" style="width:${barW}px;background:${col}"></span></span></li>`;
  }
  h += '</ul>';
  if (edges.length > 20) h += `<div style="font-size:10px;color:#94a3b8">…and ${edges.length - 20} more</div>`;
  return h;
}

// ══════════════════════════════════════════════════════════════
// Initialization — called once by each page after DOM is ready
// ══════════════════════════════════════════════════════════════
function initShared() {
  $gc = document.getElementById('graph-container');
  $empty = document.getElementById('empty-state');
  $tip = document.getElementById('tooltip');

  // Shared control listeners
  for (const id of ['influence-slider','agg-select','layer-start','layer-end']) {
    const el = document.getElementById(id);
    if (el) el.addEventListener(id.includes('slider') ? 'input' : 'change', reprocess);
  }
  const showLabelsEl = document.getElementById('show-labels');
  if (showLabelsEl) showLabelsEl.addEventListener('change', render);

  const colorByCatEl = document.getElementById('color-by-category');
  if (colorByCatEl) colorByCatEl.addEventListener('change', () => {
    const legend = document.getElementById('cat-color-legend');
    if (colorByCategory()) {
      const tagSet = new Set();
      for (const s of (maskData ? maskData.sentences || [] : [])) {
        for (const t of (s.function_tags || [])) tagSet.add(t);
      }
      legend.innerHTML = [...tagSet].sort().map(t =>
        `<span class="cat-legend-item"><span class="cat-legend-swatch" style="background:${TAG_COLORS[t]||TAG_COLORS.unknown}"></span>${t.replace(/_/g,' ')}</span>`
      ).join('');
      legend.style.display = '';
    } else {
      legend.style.display = 'none';
    }
    render();
  });

  // Threshold slider
  document.getElementById('threshold-slider').addEventListener('input', () => {
    syncThresholdFromSlider();
    reprocess();
  });
  document.getElementById('threshold-input').addEventListener('change', () => {
    const v = parseFloat(document.getElementById('threshold-input').value);
    if (!isNaN(v) && thresholdValues.length > 0) {
      const idx = nearestThresholdIdx(v);
      document.getElementById('threshold-slider').value = idx;
      syncThresholdFromSlider();
      reprocess();
    }
  });

  // Detail panel drag-resize
  const handle = document.getElementById('detail-resize-handle');
  const panel = document.getElementById('detail-panel');
  if (handle && panel) {
    let dragging = false, startX = 0, startW = 0;
    handle.addEventListener('mousedown', e => {
      dragging = true; startX = e.clientX; startW = panel.offsetWidth;
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
      e.preventDefault();
    });
    document.addEventListener('mousemove', e => {
      if (!dragging) return;
      const delta = startX - e.clientX;
      const newW = Math.max(200, Math.min(800, startW + delta));
      panel.style.width = newW + 'px';
    });
    document.addEventListener('mouseup', () => {
      if (!dragging) return;
      dragging = false;
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      if (processed) { render(); if (typeof pageOnReprocess === 'function') pageOnReprocess(); }
    });
  }

  // Window resize
  window.addEventListener('resize', () => {
    if (processed) { render(); if (typeof pageOnReprocess === 'function') pageOnReprocess(); }
  });
}

// ══════════════════════════════════════════════════════════════
// Utility: populate server mask dropdown and return paths
// ══════════════════════════════════════════════════════════════
function loadServerMaskList(callback) {
  fetch('/api/masks').then(r=>r.json()).then(ps => {
    const sel = document.getElementById('server-masks');
    ps.forEach(p => {
      const o = document.createElement('option');
      o.value = p;
      const ci = p.indexOf('circuit_discovery/');
      const si = p.indexOf('snp_sweep/');
      if (ci >= 0) o.textContent = p.slice(ci + 'circuit_discovery/'.length);
      else if (si >= 0) o.textContent = p.slice(si + 'snp_sweep/'.length);
      else o.textContent = p;
      sel.appendChild(o);
    });
    if (typeof callback === 'function') callback(ps);
  }).catch(()=>{});
}
