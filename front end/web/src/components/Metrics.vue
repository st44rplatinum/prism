<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'

const API = 'http://localhost:8000'
const data = ref<any>(null)
const loading = ref(true)
const error = ref<string | null>(null)
const hover = ref<{ x: number; y: number; gene: string; zs: number; npt: number; n: number } | null>(null)

onMounted(async () => {
  try {
    const res = await fetch(`${API}/metrics`)
    if (!res.ok) {
      const body = await res.json().catch(() => ({}))
      throw new Error(body.detail || `API returned ${res.status}`)
    }
    data.value = await res.json()
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loading.value = false
  }
})

// --- headline ---------------------------------------------------------------
const MODEL_LABELS: Record<string, string> = {
  blosum62: 'BLOSUM62',
  esm2_wt: 'ESM-2 wt-marginal',
  esm2_masked: 'ESM-2 masked-marginal',
  protein_npt_inductive: 'ProteinNPT (inductive)',
  protein_npt_transductive: 'ProteinNPT (transductive)',
}
const rows = computed(() => {
  if (!data.value) return []
  const pg = data.value.headline.test_per_gene_auroc
  const pooled = data.value.headline.test_pooled_auroc
  return Object.keys(pg).map((k) => ({
    key: k,
    label: MODEL_LABELS[k] ?? k,
    perGene: pg[k],
    pooled: pooled[k],
    // The zero-shot masked score is the bar the supervised model had to clear.
    delta: k.startsWith('protein_npt') ? pg[k] - pg.esm2_masked : null,
  }))
})

// --- scatter: zero-shot vs NPT, one point per held-out gene ------------------
const S = { w: 380, h: 380, pad: 44, lo: 0.1, hi: 1.0 }
const sx = (v: number) => S.pad + ((v - S.lo) / (S.hi - S.lo)) * (S.w - S.pad - 12)
const sy = (v: number) => S.h - S.pad - ((v - S.lo) / (S.hi - S.lo)) * (S.h - S.pad - 12)
const ticks = [0.2, 0.4, 0.6, 0.8, 1.0]

function onPoint(e: MouseEvent, g: any) {
  hover.value = { x: e.clientX, y: e.clientY, gene: g.gene, zs: g.zeroshot, npt: g.npt_inductive, n: g.n }
}

// --- ROC --------------------------------------------------------------------
const R = { w: 300, h: 300, pad: 38 }
function rocPath(points: number[][]): string {
  const x = (v: number) => R.pad + v * (R.w - R.pad - 10)
  const y = (v: number) => R.h - R.pad - v * (R.h - R.pad - 10)
  return points.map((p, i) => `${i ? 'L' : 'M'}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join(' ')
}

// --- training curve ---------------------------------------------------------
const T = { w: 380, h: 190, pad: 40 }
const trainingPath = computed(() => {
  // Returns null rather than '' when there is no data: mixing a string and an
  // object in one computed makes every property access a type error.
  if (!data.value) return null
  const h = data.value.training
  const vals = h.map((d: any) => d.val_per_gene_auroc)
  const lo = Math.min(...vals) - 0.005
  const hi = Math.max(...vals) + 0.005
  const x = (i: number) => T.pad + (i / (h.length - 1)) * (T.w - T.pad - 10)
  const y = (v: number) => T.h - T.pad - ((v - lo) / (hi - lo)) * (T.h - T.pad - 12)
  return {
    d: h.map((p: any, i: number) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(p.val_per_gene_auroc).toFixed(1)}`).join(' '),
    best: { x: x(data.value.best_epoch), y: y(h[data.value.best_epoch].val_per_gene_auroc) },
    lo, hi,
  }
})

// --- calibration ------------------------------------------------------------
const C = { w: 300, h: 300, pad: 38 }
const cx = (v: number) => C.pad + v * (C.w - C.pad - 10)
const cy = (v: number) => C.h - C.pad - v * (C.h - C.pad - 10)
</script>

<template>
  <div class="wrap">
    <p v-if="loading" class="status">Loading metrics&hellip;</p>
    <p v-else-if="error" class="status error">
      {{ error }}<br />
      <small>Generate it with <code>python -m vep.eval.report</code></small>
    </p>

    <template v-else-if="data">
      <header>
        <h2>Held-out evaluation</h2>
        <span class="sub">
          {{ data.splits.test.genes }} test genes &middot;
          {{ data.splits.test.variants.toLocaleString() }} variants &middot;
          split by gene, never by variant
        </span>
      </header>

      <!-- headline table -->
      <table class="scores">
        <thead>
          <tr>
            <th>Model</th>
            <th>per-gene AUROC</th>
            <th>pooled AUROC</th>
            <th>vs zero-shot</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="r in rows" :key="r.key" :class="{ best: r.key === 'protein_npt_inductive' }">
            <td>{{ r.label }}</td>
            <td class="num">{{ r.perGene.toFixed(4) }}</td>
            <td class="num dim">{{ r.pooled.toFixed(4) }}</td>
            <td class="num">
              <span v-if="r.delta !== null" :class="r.delta > 0 ? 'up' : 'down'">
                {{ r.delta > 0 ? '+' : '' }}{{ r.delta.toFixed(4) }}
              </span>
            </td>
          </tr>
        </tbody>
      </table>
      <p class="note">
        Per-gene AUROC is the headline because pooled AUROC rewards a model for knowing
        which <em>genes</em> are constrained &mdash; information it will not have on a new gene.
        Inductive means the test gene contributed no labels of its own, so it is the only
        row directly comparable to zero-shot.
      </p>

      <div class="grid">
        <!-- scatter -->
        <figure>
          <figcaption>
            Per-gene: zero-shot vs ProteinNPT
            <span class="cap">
              {{ data.headline.n_improved }}/{{ data.headline.n_genes_evaluable }} genes improved
            </span>
          </figcaption>
          <svg :width="S.w" :height="S.h" @mouseleave="hover = null">
            <line :x1="sx(S.lo)" :y1="sy(S.lo)" :x2="sx(1)" :y2="sy(1)" class="diag" />
            <line :x1="sx(S.lo)" :y1="sy(0.5)" :x2="sx(1)" :y2="sy(0.5)" class="chance" />
            <line :x1="sx(0.5)" :y1="sy(S.lo)" :x2="sx(0.5)" :y2="sy(1)" class="chance" />
            <g v-for="t in ticks" :key="t">
              <text :x="sx(t)" :y="S.h - S.pad + 14" class="tick">{{ t }}</text>
              <text :x="S.pad - 8" :y="sy(t) + 3" class="tick end">{{ t }}</text>
            </g>
            <circle
              v-for="g in data.per_gene"
              :key="g.gene"
              :cx="sx(g.zeroshot)"
              :cy="sy(g.npt_inductive)"
              :r="Math.max(3, Math.min(9, Math.sqrt(g.n) / 3))"
              :class="['pt', g.gain > 0 ? 'gain' : 'loss']"
              @mousemove="onPoint($event, g)"
            />
            <text :x="S.w / 2" :y="S.h - 6" class="axis">zero-shot AUROC</text>
            <text :x="-S.h / 2" :y="12" class="axis" transform="rotate(-90)">ProteinNPT AUROC</text>
          </svg>
          <p class="cap">
            Points above the diagonal improved. Area &prop; variant count.
            The far-left point is SMAD4, which zero-shot ranks
            <em>below chance</em> (0.358) and the model lifts to 0.870.
          </p>
        </figure>

        <!-- ROC -->
        <figure>
          <figcaption>ROC, all held-out variants</figcaption>
          <svg :width="R.w" :height="R.h">
            <line :x1="R.pad" :y1="R.h - R.pad" :x2="R.w - 10" :y2="10" class="diag" />
            <path :d="rocPath(data.roc.zeroshot)" class="roc zs" />
            <path :d="rocPath(data.roc.npt)" class="roc npt" />
            <text :x="R.w / 2" :y="R.h - 6" class="axis">false positive rate</text>
            <text :x="-R.h / 2" :y="12" class="axis" transform="rotate(-90)">true positive rate</text>
          </svg>
          <p class="cap">
            <span class="key npt"></span> ProteinNPT
            <span class="key zs"></span> ESM-2 masked
          </p>
        </figure>

        <!-- calibration -->
        <figure>
          <figcaption>Calibration</figcaption>
          <svg :width="C.w" :height="C.h">
            <line :x1="cx(0)" :y1="cy(0)" :x2="cx(1)" :y2="cy(1)" class="diag" />
            <g v-for="b in data.calibration.reliability" :key="b.bin">
              <line :x1="cx(b.mean_predicted)" :y1="cy(b.mean_predicted)" :x2="cx(b.mean_predicted)" :y2="cy(b.observed)" class="err" />
              <circle :cx="cx(b.mean_predicted)" :cy="cy(b.observed)" :r="Math.max(3, Math.min(9, Math.sqrt(b.n) / 8))" class="pt gain" />
            </g>
            <text :x="C.w / 2" :y="C.h - 6" class="axis">predicted probability</text>
            <text :x="-C.h / 2" :y="12" class="axis" transform="rotate(-90)">observed frequency</text>
          </svg>
          <p class="cap">
            Platt scaling fitted on validation only.
            Test ECE {{ data.calibration.ece_before.toFixed(3) }} &rarr;
            <strong>{{ data.calibration.ece_after.toFixed(3) }}</strong>.
          </p>
        </figure>

        <!-- training -->
        <figure>
          <figcaption>Training</figcaption>
          <svg v-if="trainingPath" :width="T.w" :height="T.h">
            <path :d="trainingPath.d" class="roc npt" />
            <circle :cx="trainingPath.best.x" :cy="trainingPath.best.y" r="4" class="pt best" />
            <text :x="T.pad - 8" :y="16" class="tick end">{{ trainingPath.hi.toFixed(2) }}</text>
            <text :x="T.pad - 8" :y="T.h - T.pad" class="tick end">{{ trainingPath.lo.toFixed(2) }}</text>
            <text :x="T.w / 2" :y="T.h - 6" class="axis">epoch</text>
          </svg>
          <p class="cap">
            Validation per-gene AUROC. Selected epoch {{ data.best_epoch }}
            (early-stopped at {{ data.training.length - 1 }}).
          </p>
        </figure>
      </div>

      <!-- per-gene table -->
      <h3>Per-gene detail</h3>
      <table class="genes">
        <thead>
          <tr>
            <th>Gene</th><th class="num">n</th><th class="num">base rate</th>
            <th class="num">zero-shot</th><th class="num">NPT</th><th class="num">&Delta;</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="g in data.per_gene" :key="g.gene">
            <td>{{ g.gene }}</td>
            <td class="num dim">{{ g.n }}</td>
            <td class="num dim">{{ g.base_rate.toFixed(2) }}</td>
            <td class="num" :class="{ bad: g.zeroshot < 0.5 }">{{ g.zeroshot.toFixed(3) }}</td>
            <td class="num">{{ g.npt_inductive.toFixed(3) }}</td>
            <td class="num" :class="g.gain > 0 ? 'up' : 'down'">
              {{ g.gain > 0 ? '+' : '' }}{{ g.gain.toFixed(3) }}
            </td>
          </tr>
        </tbody>
      </table>

      <div v-if="hover" class="tooltip" :style="{ left: `${hover.x + 14}px`, top: `${hover.y + 14}px` }">
        <strong>{{ hover.gene }}</strong>
        <span>zero-shot {{ hover.zs.toFixed(3) }} &rarr; NPT {{ hover.npt.toFixed(3) }}</span>
        <span class="dim">{{ hover.n }} variants</span>
      </div>
    </template>
  </div>
</template>

<style scoped>
.wrap {
  font: 13px/1.5 system-ui, sans-serif;
  padding: 16px;
  background: #fff;
  color: #222;
  border-radius: 6px;
}
header { margin-bottom: 12px; }
h2 { font-size: 15px; font-weight: 600; margin: 0; color: #222; }
h3 { font-size: 13px; font-weight: 600; margin: 22px 0 8px; color: #222; }
.sub, .cap, .note { color: #666; font-size: 12px; }
.note { max-width: 760px; margin: 8px 0 0; }
.status { color: #666; }
.status.error { color: #b00020; }

table { border-collapse: collapse; font-size: 12px; color: #222; }
.scores { margin-top: 10px; }
.scores th, .scores td, .genes th, .genes td { padding: 4px 12px 4px 0; text-align: left; }
th { font-weight: 600; color: #555; border-bottom: 1px solid #ddd; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.dim { color: #888; }
tr.best td { font-weight: 600; background: #f4f8ff; }
.up { color: #0a7d38; }
.down { color: #b00020; }
.bad { color: #b00020; font-weight: 600; }

.grid {
  display: flex;
  flex-wrap: wrap;
  gap: 28px;
  margin-top: 20px;
}
figure { margin: 0; }
figcaption { font-size: 12px; font-weight: 600; margin-bottom: 6px; }
figcaption .cap { font-weight: 400; margin-left: 6px; }
figure .cap { max-width: 380px; margin: 4px 0 0; }

.diag { stroke: #bbb; stroke-dasharray: 3 3; }
.chance { stroke: #eee; }
.err { stroke: #d0d0d0; }
.pt { cursor: crosshair; }
.pt.gain { fill: rgba(10, 125, 56, 0.65); }
.pt.loss { fill: rgba(176, 0, 32, 0.65); }
.pt.best { fill: #b00020; }
.roc { fill: none; stroke-width: 2; }
.roc.npt { stroke: #0a58ca; }
.roc.zs { stroke: #d08a1a; }
.tick { font-size: 9px; fill: #999; text-anchor: middle; }
.tick.end { text-anchor: end; }
.axis { font-size: 10px; fill: #666; text-anchor: middle; }
.key { display: inline-block; width: 14px; height: 3px; margin: 0 4px 0 10px; vertical-align: middle; }
.key.npt { background: #0a58ca; }
.key.zs { background: #d08a1a; }

.tooltip {
  position: fixed; z-index: 10;
  display: flex; flex-direction: column;
  padding: 6px 9px; background: #1c1c1c; color: #f2f2f2;
  border-radius: 4px; font-size: 12px; line-height: 1.4;
  pointer-events: none; white-space: nowrap;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
}
.tooltip .dim { color: #aaa; }
</style>
