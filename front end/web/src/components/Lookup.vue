<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { API, state, syncUrl } from '../state'

interface Result {
  variant: string
  wt_aa: string
  position: number
  mut_aa: string
  llr: { wt: number }
  pathogenicity_prob: number | null
  n_context: number | null
  clinvar_label: string | null
  clinvar_stars: number | null
  error: string | null
}

const genes = ref<any[]>([])
const input = ref(state.variant)
const results = ref<Result[]>([])
const modelAvailable = ref(true)
const loading = ref(false)
const error = ref<string | null>(null)

const current = computed(() => genes.value.find((g) => g.symbol === state.gene) ?? null)

function parse(text: string): string[] {
  return text.toUpperCase().split(/[\s,;]+/).filter(Boolean)
}

async function predict() {
  const variants = parse(input.value)
  if (!variants.length) {
    results.value = []
    return
  }
  loading.value = true
  error.value = null

  // Replace rather than push when the query has not changed, so re-running the
  // same variants - or a gene change that re-predicts - leaves one history
  // entry instead of two identical ones.
  const next = input.value.trim()
  const changed = next !== state.variant
  state.variant = next
  syncUrl(!changed)
  try {
    const res = await fetch(`${API}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ gene: state.gene, variants }),
    })
    if (!res.ok) {
      const body = await res.json().catch(() => ({}))
      throw new Error(body.detail || `API returned ${res.status}`)
    }
    const data = await res.json()
    modelAvailable.value = data.model_available
    results.value = data.results
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
    results.value = []
  } finally {
    loading.value = false
  }
}

function selectGene(symbol: string) {
  state.gene = symbol
  syncUrl()
}

// One place a state change causes a re-score, so a picker click and the back
// button behave identically.
watch(() => [state.gene, state.variant], () => {
  if (input.value !== state.variant) input.value = state.variant
  if (input.value) predict()
  else results.value = []
})

/** Same diverging scale as the heat map and the structure view. */
function colour(p: number): string {
  const v = Math.min(1, Math.max(0, p))
  if (v >= 0.5) {
    const t = (v - 0.5) / 0.5
    return `rgb(255,${Math.round(255 - 255 * t)},${Math.round(255 - 217 * t)})`
  }
  const t = (0.5 - v) / 0.5
  return `rgb(${Math.round(255 - 206 * t)},${Math.round(255 - 153 * t)},${Math.round(255 - 66 * t)})`
}

function verdict(p: number | null): string {
  if (p === null) return ''
  if (p >= 0.9) return 'likely pathogenic'
  if (p >= 0.5) return 'leaning pathogenic'
  if (p >= 0.1) return 'leaning benign'
  return 'likely benign'
}

/** Whether the prediction agrees with ClinVar, when ClinVar has an opinion. */
function agreement(r: Result): string | null {
  if (r.pathogenicity_prob === null || !r.clinvar_label) return null
  const predicted = r.pathogenicity_prob >= 0.5 ? 'pathogenic' : 'benign'
  return predicted === r.clinvar_label ? 'agrees' : 'disagrees'
}

onMounted(async () => {
  try {
    const res = await fetch(`${API}/genes`)
    genes.value = await res.json()
  } catch {
    /* the picker is a convenience; lookup still works by typing */
  }
  if (input.value) predict()
})
</script>

<template>
  <div class="wrap">
    <header>
      <select :value="state.gene" :disabled="loading"
              @change="selectGene(($event.target as HTMLSelectElement).value)">
        <option v-for="g in genes" :key="g.symbol" :value="g.symbol">
          {{ g.symbol }} &mdash; {{ g.length }} aa
        </option>
      </select>

      <input v-model="input" class="variants" spellcheck="false"
             placeholder="R175H, P72R &hellip;"
             @keyup.enter="predict" />

      <button class="go" :disabled="loading" @click="predict">
        {{ loading ? 'Scoring…' : 'Predict' }}
      </button>
    </header>

    <p v-if="current" class="sub">{{ current.protein_name }}</p>

    <p v-if="error" class="status error">
      {{ error }}<br />
      <small>Is the API running? <code>uvicorn api.main:app --port 8000</code></small>
    </p>

    <p v-else-if="!modelAvailable" class="status warn">
      No trained head loaded, so only the zero-shot ESM-2 score is shown.
      Build the feature cache with <code>python -m vep.esm.cache</code>.
    </p>

    <table v-if="results.length" class="results">
      <thead>
        <tr>
          <th>Variant</th>
          <th class="num">Pathogenicity</th>
          <th></th>
          <th class="num">ESM-2 LLR</th>
          <th>ClinVar</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="r in results" :key="r.variant">
          <td class="mono">{{ r.variant }}</td>
          <td class="num">
            <template v-if="r.pathogenicity_prob !== null">
              {{ (r.pathogenicity_prob * 100).toFixed(1) }}%
            </template>
            <span v-else class="dim">&mdash;</span>
          </td>
          <td class="barcell">
            <div v-if="r.pathogenicity_prob !== null" class="bar">
              <div class="fill" :style="{ width: `${r.pathogenicity_prob * 100}%`,
                                          background: colour(r.pathogenicity_prob) }"></div>
            </div>
            <span class="verdict">{{ verdict(r.pathogenicity_prob) }}</span>
          </td>
          <td class="num dim">{{ r.llr ? r.llr.wt.toFixed(2) : '—' }}</td>
          <td>
            <span v-if="r.clinvar_label" :class="['tag', r.clinvar_label]">
              {{ r.clinvar_label }}
            </span>
            <span v-else class="dim">not in ClinVar</span>
            <span v-if="r.clinvar_stars" class="dim stars">{{ r.clinvar_stars }}★</span>
          </td>
          <td>
            <span v-if="agreement(r) === 'disagrees'" class="tag disagree">disagrees</span>
            <span v-else-if="r.error" class="tag disagree">{{ r.error }}</span>
          </td>
        </tr>
      </tbody>
    </table>

    <p v-if="results.length" class="note">
      Calibrated probability from ProteinNPT, using
      {{ results[0].n_context ?? 0 }} labelled neighbours per query. The query's own
      label is always masked, and a variant already in ClinVar is excluded from its
      own context. LLR is the zero-shot ESM-2 score for comparison &mdash; more
      negative means evolution tolerates it less.
    </p>

    <p v-if="!results.length && !loading && !error" class="note">
      Enter one or more substitutions for
      <strong>{{ state.gene }}</strong> &mdash; for example
      <code>R175H</code>, or several at once: <code>R175H P72R R273H</code>.
    </p>
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
  gap: 8px;
  flex-wrap: wrap;
}

select, .variants {
  font: inherit;
  font-size: 13px;
  padding: 5px 8px;
  border: 1px solid #d0d0d0;
  border-radius: 4px;
  background: #fff;
  color: #222;
}

.variants {
  flex: 1 1 260px;
  min-width: 180px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}

.go {
  font: 12px/1.5 system-ui, sans-serif;
  padding: 6px 16px;
  border: 1px solid #222;
  background: #222;
  color: #fff;
  border-radius: 4px;
  cursor: pointer;
}

.go:disabled { opacity: 0.5; cursor: default; }

.sub { color: #666; font-size: 12px; margin: 8px 0 0; }
.status { color: #666; }
.status.error { color: #b00020; }
.status.warn { color: #8a6d00; }
.note { color: #666; font-size: 12px; max-width: 70ch; }

table { border-collapse: collapse; font-size: 12px; margin-top: 14px; }
th, td { padding: 6px 14px 6px 0; text-align: left; vertical-align: middle; }
th { font-weight: 600; color: #555; border-bottom: 1px solid #ddd; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.dim { color: #888; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-weight: 600; }

.barcell { min-width: 190px; }
.bar {
  display: inline-block;
  vertical-align: middle;
  width: 110px;
  height: 8px;
  background: #f0f0f0;
  border-radius: 2px;
  overflow: hidden;
}
.fill { height: 100%; }
.verdict { margin-left: 8px; color: #777; font-size: 11px; }

.tag {
  padding: 1px 6px;
  border-radius: 3px;
  font-size: 11px;
  background: #f0f0f0;
  color: #666;
}
.tag.pathogenic { background: #fdeaee; color: #b00020; }
.tag.benign { background: #eaf1fd; color: #1b4fa8; }
.tag.disagree { background: #fff4e5; color: #8a4b00; }
.stars { margin-left: 6px; font-size: 11px; }
</style>
