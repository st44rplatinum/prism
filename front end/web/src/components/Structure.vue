<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, shallowRef, watch } from 'vue'
import { API, state, syncUrl } from '../state'
import GenePicker from './GenePicker.vue'

declare const $3Dmol: any


const container = ref<HTMLDivElement | null>(null)
// shallowRef, not ref: the 3Dmol viewer holds a large WebGL object graph and
// making it deeply reactive would have Vue walk the whole scene on every touch.
const viewer = shallowRef<any>(null)

const genes = ref<any[]>([])
// Bound to the shared store so the gene survives a tab switch.
const gene = computed({
  get: () => state.gene,
  set: (value: string) => { state.gene = value },
})
const loading = ref(true)
const error = ref<string | null>(null)
const scores = ref<number[]>([])
const plddt = ref<number[]>([])
const sequence = ref('')
const scoreType = ref<'probability' | 'llr'>('probability')
const colourBy = ref<'pathogenicity' | 'plddt'>('pathogenicity')
const picked = ref<{ resi: number; aa: string; score: number; plddt: number } | null>(null)

const current = computed(() => genes.value.find((g) => g.symbol === gene.value) ?? null)

// --- colour -----------------------------------------------------------------
function lerp(a: number, b: number, t: number) {
  return Math.round(a + (b - a) * t)
}
const hex = (r: number, g: number, b: number) =>
  (r << 16) | (g << 8) | b   // 3Dmol wants a packed integer, not a CSS string

/** Pathogenicity, diverging at 0.5 — matches the heat map exactly. */
function pathogenicityColour(p: number): number {
  const v = Math.min(1, Math.max(0, p))
  if (v >= 0.5) {
    const t = (v - 0.5) / 0.5
    return hex(255, lerp(255, 0, t), lerp(255, 38, t))
  }
  const t = (0.5 - v) / 0.5
  return hex(lerp(255, 49, t), lerp(255, 102, t), lerp(255, 189, t))
}

/** LLR fallback: scores run roughly [-15, 0], so rescale before reusing the ramp. */
function llrColour(v: number): number {
  return pathogenicityColour(Math.min(1, Math.max(0, -v / 12)))
}

/**
 * AlphaFold's own confidence bands. Kept as their published colours rather
 * than a continuous ramp, because the bands are the thing people read: below
 * 50 means the region is very likely disordered, which is a categorical claim
 * and not a shade of blue.
 */
function plddtColour(v: number): number {
  if (v >= 90) return hex(0, 83, 214)
  if (v >= 70) return hex(101, 203, 243)
  if (v >= 50) return hex(255, 219, 19)
  return hex(255, 125, 69)
}

function colourFor(resi: number): number {
  if (colourBy.value === 'plddt') return plddtColour(plddt.value[resi - 1] ?? 0)
  const v = scores.value[resi - 1]
  if (v === undefined) return hex(200, 200, 200)
  return scoreType.value === 'probability' ? pathogenicityColour(v) : llrColour(v)
}

// --- data -------------------------------------------------------------------
/** pLDDT rides in the B-factor column of AlphaFold PDB files. */
function parsePlddt(pdb: string, length: number): number[] {
  const out = new Array(length).fill(0)
  for (const line of pdb.split('\n')) {
    if (!line.startsWith('ATOM') || line.slice(12, 16).trim() !== 'CA') continue
    const resi = parseInt(line.slice(22, 26).trim(), 10)
    const b = parseFloat(line.slice(60, 66).trim())
    if (resi >= 1 && resi <= length) out[resi - 1] = b
  }
  return out
}

async function load() {
  loading.value = true
  error.value = null
  picked.value = null
  try {
    const meta = await fetch(`${API}/genes/${gene.value}/structure`)
    if (!meta.ok) {
      const body = await meta.json().catch(() => ({}))
      throw new Error(body.detail || `API returned ${meta.status}`)
    }
    const data = await meta.json()
    scores.value = data.scores
    sequence.value = data.sequence
    scoreType.value = data.score_type

    // Fetched straight from EBI: they send Access-Control-Allow-Origin: *, so
    // routing a 250 KB static file through our API would buy nothing.
    const pdbRes = await fetch(data.pdb_url)
    if (!pdbRes.ok) throw new Error(`AlphaFold returned ${pdbRes.status}`)
    const pdb = await pdbRes.text()
    plddt.value = parsePlddt(pdb, data.length)

    render(pdb)
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loading.value = false
  }
}

function render(pdb: string) {
  if (!container.value) return
  if (typeof $3Dmol === 'undefined') {
    error.value = '3Dmol.js did not load (check the CDN script in index.html)'
    return
  }
  if (!viewer.value) {
    viewer.value = $3Dmol.createViewer(container.value, { backgroundColor: 'white' })
  }
  const v = viewer.value
  v.clear()
  v.addModel(pdb, 'pdb')
  applyStyle()
  v.zoomTo()
  v.render()
}

function applyStyle() {
  const v = viewer.value
  if (!v) return
  v.setStyle({}, { cartoon: { colorfunc: (atom: any) => colourFor(atom.resi) } })
  v.setClickable({}, true, (atom: any) => {
    picked.value = {
      resi: atom.resi,
      aa: sequence.value[atom.resi - 1] ?? '?',
      score: scores.value[atom.resi - 1],
      plddt: plddt.value[atom.resi - 1],
    }
  })
  v.render()
}

function setColourBy(mode: 'pathogenicity' | 'plddt') {
  if (mode === colourBy.value) return
  colourBy.value = mode
  applyStyle()
}

function selectGene(symbol: string) {
  gene.value = symbol
  syncUrl()
}

watch(() => state.gene, load)

onMounted(async () => {
  try {
    const res = await fetch(`${API}/genes`)
    genes.value = await res.json()
  } catch {
    /* the picker is a convenience; the default gene still loads */
  }
  await load()
})

onBeforeUnmount(() => {
  // WebGL contexts are a limited resource and are not garbage collected
  // promptly; switching tabs repeatedly without this eventually loses the
  // context and the viewer silently stops rendering.
  try {
    viewer.value?.clear()
  } catch {
    /* nothing useful to do if the context is already gone */
  }
  viewer.value = null
})
</script>

<template>
  <div class="wrap">
    <header>
      <GenePicker :model-value="state.gene" :genes="genes" :disabled="loading"
                  @select="selectGene" />
      <span v-if="current" class="protein">{{ current.protein_name }}</span>
      <div class="modes">
        <button :class="{ on: colourBy === 'pathogenicity' }" @click="setColourBy('pathogenicity')">
          Pathogenicity
        </button>
        <button :class="{ on: colourBy === 'plddt' }" @click="setColourBy('plddt')">
          AlphaFold pLDDT
        </button>
      </div>
    </header>

    <p v-if="loading" class="status">Loading structure&hellip;</p>
    <p v-else-if="error" class="status error">
      {{ error }}<br />
      <small>Is the API running? <code>uvicorn api.main:app --port 8000</code></small>
    </p>

    <div class="stage">
      <div ref="container" class="viewer"></div>
      <div v-if="picked" class="pick">
        <strong>{{ picked.aa }}{{ picked.resi }}</strong>
        <span v-if="picked.score !== undefined">
          mean predicted pathogenicity
          {{ scoreType === 'probability' ? (picked.score * 100).toFixed(1) + '%' : picked.score.toFixed(2) }}
        </span>
        <span class="dim">pLDDT {{ picked.plddt?.toFixed(0) }}</span>
      </div>
    </div>

    <div class="legend">
      <template v-if="colourBy === 'pathogenicity'">
        <span>tolerated</span>
        <span class="sw" :style="{ background: '#3166bd' }"></span>
        <span class="sw" :style="{ background: '#fff' }"></span>
        <span class="sw" :style="{ background: '#ff0026' }"></span>
        <span>damaging</span>
        <span class="note">
          mean over all 19 substitutions at each residue{{ scoreType === 'llr' ? ' (LLR fallback — no probability matrix cached)' : '' }}
        </span>
      </template>
      <template v-else>
        <span class="sw" :style="{ background: '#0053d6' }"></span><span>very high &ge;90</span>
        <span class="sw" :style="{ background: '#65cbf3' }"></span><span>confident &ge;70</span>
        <span class="sw" :style="{ background: '#ffdb13' }"></span><span>low &ge;50</span>
        <span class="sw" :style="{ background: '#ff7d45' }"></span><span>very low &lt;50</span>
        <span class="note">below 50 usually means intrinsically disordered</span>
      </template>
    </div>
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

header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
  flex-wrap: wrap;
}

select {
  font: inherit;
  font-size: 13px;
  padding: 4px 8px;
  border: 1px solid #d0d0d0;
  border-radius: 4px;
  background: #fff;
  color: #222;
}

.protein {
  color: #666;
  font-size: 12px;
  flex: 1 1 auto;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.modes {
  display: flex;
  gap: 4px;
}

.modes button {
  font: 12px/1.5 system-ui, sans-serif;
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

.status { color: #666; }
.status.error { color: #b00020; }

.stage {
  position: relative;
  width: 100%;
  height: 460px;
  border: 1px solid #e5e5e5;
  border-radius: 4px;
  overflow: hidden;
}

/* 3Dmol measures its container, so the element must have a real size before
   the viewer is created - a zero-height div renders a blank canvas. */
.viewer {
  position: relative;
  width: 100%;
  height: 100%;
}

.pick {
  position: absolute;
  left: 10px;
  bottom: 10px;
  display: flex;
  flex-direction: column;
  padding: 6px 9px;
  background: rgba(28, 28, 28, 0.9);
  color: #f2f2f2;
  border-radius: 4px;
  font-size: 12px;
  line-height: 1.4;
  pointer-events: none;
}

.pick .dim { color: #aaa; }

.legend {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 10px;
  color: #555;
  font-size: 12px;
  flex-wrap: wrap;
}

.sw {
  width: 22px;
  height: 12px;
  border: 1px solid #ddd;
}

.note {
  margin-left: 10px;
  color: #888;
}
</style>
