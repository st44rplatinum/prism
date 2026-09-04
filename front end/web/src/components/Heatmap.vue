<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'

// --- state -----------------------------------------------------------------
// ref() makes a value reactive: read/write it as .value here in the script,
// and as a bare name inside the <template>.
const cv = ref<HTMLCanvasElement | null>(null)
const matrix = ref<(number | null)[][]>([])
const sequence = ref('')
const alphabet = ref<string[]>([])
const loading = ref(true)
const error = ref<string | null>(null)
const jobProgress = ref<number | null>(null)

// Which quantity is being shown. These answer different questions:
//   llr          "would evolution tolerate this substitution?"  (zero-shot)
//   probability  "is this substitution pathogenic?"             (trained model)
// They come apart - SMAD4 is a held-out gene where the LLR ranks variants worse
// than chance while the model reaches 0.87 AUROC - so both are worth keeping.
type Mode = 'probability' | 'llr'
const mode = ref<Mode>('probability')

interface GeneSummary {
  symbol: string
  protein_name: string
  length: number
  n_variants: number
  n_pathogenic: number
  n_benign: number
  split: string | null
  saturation_schemes: string[]
}
const genes = ref<GeneSummary[]>([])
const gene = ref('TP53')

// ClinVar ground truth for the current gene, keyed "position:mutant". Shown in
// the tooltip so a prediction can be read against the label it should agree
// with - which is where the model is most interesting, since the cases worth
// looking at are the ones where they disagree.
const clinvar = ref(new Map<string, { label: string; stars: number }>())

interface Hover {
  x: number
  y: number
  position: number   // 1-based
  wt: string
  mut: string
  value: number | null
  label: string | null
  stars: number | null
}
const hover = ref<Hover | null>(null)

// Selecting an uncached gene starts a masked-marginal scan on the GPU, which
// runs from about a minute to over half an hour depending on length. That cost
// has to be visible in the picker rather than discovered after clicking, so
// genes are grouped by whether they are ready and the wait is estimated up
// front. LLR mode needs only wt-marginals (one pass per window, sub-second),
// so everything is instant there.
const current = computed(() =>
  genes.value.find((g) => g.symbol === gene.value) ?? null,
)
function isReady(g: GeneSummary): boolean {
  return mode.value === 'llr' || g.saturation_schemes.includes('probability')
}
// Same cost model the precompute script uses: positions x window length,
// calibrated on TP53 (393 residues, 58s).
function estimateSeconds(length: number): number {
  return (length * Math.min(length, 1022)) / ((393 * 393) / 58)
}
function estimateLabel(length: number): string {
  const s = estimateSeconds(length)
  return s < 90 ? `~${Math.round(s)}s` : s < 5400 ? `~${(s / 60).toFixed(0)}m` : `~${(s / 3600).toFixed(1)}h`
}
const readyGenes = computed(() => genes.value.filter(isReady))
const pendingGenes = computed(() => genes.value.filter((g) => !isReady(g)))

const API = 'http://localhost:8000'
const CELL_WIDTH = 3
const CELL_HEIGHT = 14
const RULER_HEIGHT = 16

// LLR colour-scale bounds, chosen from the actual score distribution rather
// than guessed. Across six proteins the pooled log-likelihood ratios run from
// about -17 to +9 with a median of -4.76, and a [-10, 0] clamp would throw away
// 9% of cells - specifically the most deleterious ones, which are the whole
// point. [-15, +3] loses 0.18%.
const MIN_LLR = -15
const MAX_LLR = 3

// --- colour ----------------------------------------------------------------
function lerp(a: number, b: number, t: number): number {
  return Math.round(a + (b - a) * t)
}

/**
 * Pathogenicity probability, diverging around 0.5.
 *
 * Centred rather than a plain white-to-red ramp so that UNCERTAIN cells render
 * pale. Where the model is unsure is exactly where a variant of uncertain
 * significance sits, and that is the most useful thing this picture can show;
 * a sequential ramp would bury it in mid-tone.
 */
function colourProb(raw: number): string {
  // Clamped defensively. An out-of-range input here produces channel values
  // like rgb(-6543, ...), which the canvas rejects silently and then paints
  // with whatever fillStyle was last valid - in practice the default black.
  // That is exactly what happened when a stale server answered a request for
  // probabilities with LLR values, and a whole-canvas blackout is far too
  // quiet a symptom for a data mismatch.
  const p = Math.min(1, Math.max(0, raw))
  if (p >= 0.5) {
    const t = (p - 0.5) / 0.5
    return `rgb(255, ${lerp(255, 0, t)}, ${lerp(255, 38, t)})`
  }
  const t = (0.5 - p) / 0.5
  return `rgb(${lerp(255, 49, t)}, ${lerp(255, 102, t)}, ${lerp(255, 189, t)})`
}

/**
 * Log-likelihood ratio, diverging around zero.
 *
 * Negative (mutation less likely than wild-type = predicted damaging) ramps
 * white -> yellow -> orange -> red -> dark red. Multiple stops rather than one
 * linear fade because the data is heavily skewed negative: with a single ramp
 * almost every cell lands in the same mid-red and the gradation that actually
 * distinguishes "mildly" from "severely" damaging is invisible.
 *
 * Positive (mutation MORE likely than wild-type) ramps white -> blue. These
 * are rare but real, and collapsing them to white hides them entirely.
 */
function colourLLR(value: number): string {
  if (value >= 0) {
    const t = Math.min(value, MAX_LLR) / MAX_LLR
    return `rgb(${lerp(255, 49, t)}, ${lerp(255, 102, t)}, ${lerp(255, 189, t)})`
  }
  const t = Math.min(Math.abs(value), Math.abs(MIN_LLR)) / Math.abs(MIN_LLR)
  const stops: Array<[number, [number, number, number]]> = [
    [0.0, [255, 255, 255]],
    [0.25, [254, 217, 118]],
    [0.5, [253, 141, 60]],
    [0.75, [227, 26, 28]],
    [1.0, [128, 0, 38]],
  ]
  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, c0] = stops[i]
    const [t1, c1] = stops[i + 1]
    if (t <= t1) {
      const local = (t - t0) / (t1 - t0)
      return `rgb(${lerp(c0[0], c1[0], local)}, ${lerp(c0[1], c1[1], local)}, ${lerp(c0[2], c1[2], local)})`
    }
  }
  return 'rgb(128, 0, 38)'
}

// Single dispatch point, so the drawing loop and the legend can never disagree
// about which scale is in force - which is exactly the bug that made every cell
// render pale blue when probabilities were fed to the LLR ramp.
function colourFor(value: number): string {
  return mode.value === 'probability' ? colourProb(value) : colourLLR(value)
}

// --- hover -----------------------------------------------------------------
function onMove(event: MouseEvent) {
  const canvas = cv.value
  if (!canvas || matrix.value.length === 0) return

  // offsetX/offsetY are relative to the canvas itself, so the horizontal
  // scroll of the container needs no correction here.
  const position = Math.floor(event.offsetX / CELL_WIDTH)
  const aaIndex = Math.floor(event.offsetY / CELL_HEIGHT)

  // Below the last amino-acid row is the position ruler, not data.
  if (
    position < 0 ||
    position >= matrix.value.length ||
    aaIndex < 0 ||
    aaIndex >= alphabet.value.length
  ) {
    hover.value = null
    return
  }

  const mut = alphabet.value[aaIndex]
  const wt = sequence.value[position]
  const known = clinvar.value.get(`${position + 1}:${mut}`)
  hover.value = {
    x: event.clientX,
    y: event.clientY,
    position: position + 1,
    wt,
    mut,
    value: matrix.value[position][aaIndex],
    label: known?.label ?? null,
    stars: known?.stars ?? null,
  }
}

function onLeave() {
  hover.value = null
}

const hoverStyle = computed(() => {
  const h = hover.value
  if (!h) return {}
  // Flip to the left of the cursor near the right edge so the tooltip is never
  // clipped by the viewport.
  const flip = h.x > window.innerWidth - 260
  return {
    left: `${flip ? h.x - 250 : h.x + 14}px`,
    top: `${Math.min(h.y + 14, window.innerHeight - 90)}px`,
  }
})

function formatValue(v: number | null): string {
  if (v === null) return 'wild-type'
  return mode.value === 'probability'
    ? `${(v * 100).toFixed(1)}% pathogenic`
    : `LLR ${v.toFixed(2)}`
}

// --- drawing ---------------------------------------------------------------
function draw() {
  const canvas = cv.value
  if (!canvas || matrix.value.length === 0) return

  const length = matrix.value.length
  const nAA = alphabet.value.length

  canvas.width = length * CELL_WIDTH
  canvas.height = nAA * CELL_HEIGHT + RULER_HEIGHT

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  ctx.fillStyle = '#fff'
  ctx.fillRect(0, 0, canvas.width, canvas.height)

  for (let position = 0; position < length; position++) {
    for (let aaIndex = 0; aaIndex < nAA; aaIndex++) {
      const value = matrix.value[position][aaIndex]
      // null = wild-type. The reference, not a prediction, so it gets its own
      // grey rather than whatever the scale maps 0 (or 0.5) to.
      ctx.fillStyle = value === null ? '#c8c8c8' : colourFor(value)
      ctx.fillRect(
        position * CELL_WIDTH,
        aaIndex * CELL_HEIGHT,
        CELL_WIDTH,
        CELL_HEIGHT,
      )
    }
  }

  // Position ruler along the bottom, ticked every 50 residues.
  const rulerTop = nAA * CELL_HEIGHT
  ctx.fillStyle = '#666'
  ctx.font = '10px monospace'
  ctx.textBaseline = 'top'
  for (let position = 0; position < length; position += 50) {
    const x = position * CELL_WIDTH
    ctx.fillRect(x, rulerTop, 1, 4)
    ctx.fillText(String(position + 1), x + 2, rulerTop + 5)
  }
}

// --- data ------------------------------------------------------------------
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

/**
 * Wait for a background saturation job.
 *
 * Probabilities are built on masked-marginal scores, which cost one forward
 * pass per residue, so the first request for a gene comes back 202 with a job
 * handle instead of a matrix. Bounded rather than `while (true)`: a stalled job
 * should surface as an error, not as a tab that spins forever.
 */
async function waitForJob(jobId: string, timeoutMs = 20 * 60 * 1000) {
  const started = Date.now()
  while (Date.now() - started < timeoutMs) {
    const res = await fetch(`${API}/jobs/${jobId}`)
    if (!res.ok) throw new Error(`job poll failed: ${res.status}`)
    const job = await res.json()
    jobProgress.value = job.progress ?? null
    // The API reports 'error', not 'failed'; polling for the wrong string here
    // means a dead job is never noticed.
    if (job.status === 'done') return
    if (job.status === 'error') throw new Error(job.detail || 'job failed')
    await sleep(1500)
  }
  throw new Error('timed out waiting for the saturation job')
}

async function load() {
  loading.value = true
  error.value = null
  jobProgress.value = null
  try {
    const url = `${API}/genes/${gene.value}/saturation?value=${mode.value === 'probability' ? 'probability' : 'llr'}${mode.value === 'llr' ? '&scheme=wt' : ''}`

    // `let`, not `const`: the 202 path reassigns after the job completes.
    let response = await fetch(url)

    if (response.status === 202) {
      const job = await response.json()
      await waitForJob(job.job_id)
      response = await fetch(url)
    }
    if (!response.ok) {
      throw new Error(`API returned ${response.status} ${response.statusText}`)
    }

    const data = await response.json()
    // The server echoes which quantity it produced. Older builds omit it, and
    // silently colouring LLRs with the probability scale is worse than failing.
    if (data.value && data.value !== mode.value) {
      throw new Error(
        `asked for ${mode.value} but the API returned ${data.value} - is it running the current code?`,
      )
    }
    matrix.value = data.matrix
    sequence.value = data.sequence
    // Taken from the response rather than hardcoded: the row order is decided
    // by the server, and a silent mismatch would mislabel every row.
    alphabet.value = data.alphabet

    await loadClinvar()
    draw()
  } catch (e) {
    // Without this the canvas just stays blank and the failure is invisible.
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loading.value = false
    jobProgress.value = null
  }
}

function setMode(next: Mode) {
  if (next === mode.value) return
  mode.value = next
  load()
}

async function loadClinvar() {
  const map = new Map<string, { label: string; stars: number }>()
  try {
    const res = await fetch(`${API}/genes/${gene.value}/variants`)
    if (res.ok) {
      for (const v of await res.json()) {
        map.set(`${v.position}:${v.mut_aa}`, {
          label: v.label,
          stars: v.review_stars,
        })
      }
    }
  } catch {
    // Labels are an overlay, not the point of the view: a failure here should
    // not stop the heat map from rendering.
  }
  clinvar.value = map
}

async function loadGenes() {
  const res = await fetch(`${API}/genes`)
  if (!res.ok) throw new Error(`could not list genes: ${res.status}`)
  genes.value = await res.json()
}

async function selectGene(symbol: string) {
  gene.value = symbol
  await load()
  // The gene just computed is now cached, so refresh the list to move it into
  // the ready group rather than leaving the picker stale.
  await loadGenes().catch(() => {})
}

onMounted(async () => {
  try {
    await loadGenes()
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
    loading.value = false
    return
  }
  await load()
})
</script>

<template>
  <div class="wrap">
    <header>
      <div class="pick">
        <select
          :value="gene"
          :disabled="loading"
          @change="selectGene(($event.target as HTMLSelectElement).value)"
        >
          <optgroup :label="`Ready (${readyGenes.length})`">
            <option v-for="g in readyGenes" :key="g.symbol" :value="g.symbol">
              {{ g.symbol }} &mdash; {{ g.length }} aa, {{ g.n_variants }} variants
            </option>
          </optgroup>
          <optgroup
            v-if="pendingGenes.length"
            :label="`Not yet computed (${pendingGenes.length}) — needs a GPU scan`"
          >
            <option v-for="g in pendingGenes" :key="g.symbol" :value="g.symbol">
              {{ g.symbol }} &mdash; {{ g.length }} aa, {{ estimateLabel(g.length) }} to compute
            </option>
          </optgroup>
        </select>
        <span v-if="current" class="protein">{{ current.protein_name }}</span>
      </div>
      <div class="modes">
        <button :class="{ on: mode === 'probability' }" @click="setMode('probability')">
          Pathogenicity
        </button>
        <button :class="{ on: mode === 'llr' }" @click="setMode('llr')">
          ESM-2 LLR
        </button>
      </div>
    </header>

    <p v-if="current" class="meta">
      {{ mode === 'probability' ? 'Predicted pathogenicity' : 'Evolutionary likelihood' }}
      of every substitution &middot;
      {{ current.length }} residues &middot;
      {{ current.n_pathogenic }} pathogenic / {{ current.n_benign }} benign in ClinVar
      <span v-if="current.split" class="split">held-out: {{ current.split }}</span>
      <span v-if="!isReady(current)" class="warn">
        not cached &mdash; selecting runs a {{ estimateLabel(current.length) }} GPU scan
      </span>
    </p>

    <p v-if="loading" class="status">
      Loading&hellip;
      <span v-if="jobProgress !== null">
        computing masked-marginal scan &mdash; {{ Math.round(jobProgress * 100) }}%
      </span>
    </p>
    <p v-else-if="error" class="status error">
      Could not load: {{ error }}<br />
      <small>Is the API running? <code>uvicorn api.main:app --port 8000</code></small>
    </p>

    <div v-show="!loading && !error" class="heatmap">
      <div class="labels">
        <div v-for="aa in alphabet" :key="aa" class="label">{{ aa }}</div>
      </div>
      <canvas
        ref="cv"
        @mousemove="onMove"
        @mouseleave="onLeave"
      ></canvas>
    </div>

    <div v-if="hover" class="tooltip" :style="hoverStyle">
      <strong>{{ hover.wt }}{{ hover.position }}{{ hover.mut }}</strong>
      <span class="val">{{ formatValue(hover.value) }}</span>
      <span v-if="hover.label" :class="['clinvar', hover.label]">
        ClinVar {{ hover.label }}
        <span class="stars">{{ hover.stars }}&#9733;</span>
      </span>
      <span v-else-if="hover.value !== null" class="clinvar unknown">
        not in ClinVar
      </span>
    </div>

    <div v-if="!loading && !error" class="legend">
      <template v-if="mode === 'probability'">
        <span>benign</span>
        <span class="swatch" :style="{ background: colourProb(0) }"></span>
        <span class="swatch" :style="{ background: colourProb(0.25) }"></span>
        <span class="swatch" :style="{ background: colourProb(0.5) }"></span>
        <span class="swatch" :style="{ background: colourProb(0.75) }"></span>
        <span class="swatch" :style="{ background: colourProb(1) }"></span>
        <span>pathogenic</span>
      </template>
      <template v-else>
        <span>more likely than WT</span>
        <span class="swatch" :style="{ background: colourLLR(3) }"></span>
        <span class="swatch" :style="{ background: colourLLR(0) }"></span>
        <span class="swatch" :style="{ background: colourLLR(-5) }"></span>
        <span class="swatch" :style="{ background: colourLLR(-10) }"></span>
        <span class="swatch" :style="{ background: colourLLR(-15) }"></span>
        <span>damaging</span>
      </template>
      <span class="swatch wt"></span>
      <span>wild-type</span>
    </div>
  </div>
</template>

<style scoped>
/* The heatmap panel pins itself to a light surface instead of inheriting the
   page theme. This is not cosmetic: both colour scales use white/pale for the
   neutral value, so on a dark page the neutral cells become the brightest thing
   on screen and the visual hierarchy inverts - the eye is pulled to exactly the
   cells that carry no signal. */
.wrap {
  font: 13px/1.5 system-ui, sans-serif;
  padding: 16px;
  background: #fff;
  color: #222;
  border-radius: 6px;
}

header {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 12px;
  flex-wrap: wrap;
}

.pick {
  display: flex;
  align-items: baseline;
  gap: 10px;
  min-width: 0;
}

.pick select {
  font: inherit;
  font-size: 13px;
  padding: 4px 8px;
  border: 1px solid #d0d0d0;
  border-radius: 4px;
  background: #fff;
  color: #222;
  max-width: 420px;
}

.protein {
  color: #666;
  font-size: 12px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.meta {
  margin: 0 0 12px;
  color: #666;
  font-size: 12px;
}

.split {
  margin-left: 8px;
  padding: 1px 6px;
  border-radius: 3px;
  background: #eef;
  color: #446;
}

.warn {
  margin-left: 8px;
  padding: 1px 6px;
  border-radius: 3px;
  background: #fff3cd;
  color: #7a5b00;
}

.modes {
  display: flex;
  gap: 4px;
}

.modes button {
  font: inherit;
  font-size: 12px;
  padding: 3px 10px;
  border: 1px solid #d0d0d0;
  background: #f7f7f7;
  color: #444;
  border-radius: 4px;
  cursor: pointer;
}

.modes button.on {
  background: #222;
  border-color: #222;
  color: #fff;
}

.status {
  color: #666;
}

.status.error {
  color: #b00020;
}

.heatmap {
  display: flex;
  align-items: flex-start;
  width: fit-content;
  max-width: 100%;
  overflow-x: auto;
}

.labels {
  flex: 0 0 20px;
}

.label {
  height: 14px;
  line-height: 14px;
  font: 11px/14px monospace;
  color: #333;
  text-align: center;
}

canvas {
  display: block;
  image-rendering: pixelated;
  cursor: crosshair;
}

.tooltip {
  position: fixed;
  z-index: 10;
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 6px 9px;
  background: #1c1c1c;
  color: #f2f2f2;
  border-radius: 4px;
  font-size: 12px;
  line-height: 1.4;
  pointer-events: none;   /* never let the tooltip eat its own mousemove */
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
  white-space: nowrap;
}

.tooltip .val {
  color: #cfcfcf;
}

.tooltip .clinvar {
  font-size: 11px;
  padding: 1px 5px;
  border-radius: 3px;
  align-self: flex-start;
}

.tooltip .clinvar.pathogenic {
  background: #7a1020;
  color: #ffd9df;
}

.tooltip .clinvar.benign {
  background: #103a6b;
  color: #d6e6ff;
}

.tooltip .clinvar.unknown {
  background: #333;
  color: #aaa;
}

.tooltip .stars {
  opacity: 0.7;
}

.legend {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 12px;
  color: #555;
  font-size: 12px;
}

.swatch {
  width: 22px;
  height: 12px;
  border: 1px solid #ddd;
}

.swatch.wt {
  background: #c8c8c8;
  margin-left: 10px;
}
</style>
