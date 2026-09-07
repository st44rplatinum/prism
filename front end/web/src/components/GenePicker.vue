<script setup lang="ts">
import { computed, nextTick, ref } from 'vue'

const props = withDefaults(defineProps<{
  modelValue: string
  genes: any[]
  disabled?: boolean
  label?: (g: any) => string
  ready?: (g: any) => boolean
}>(), {
  disabled: false,
  label: (g: any) => `${g.length} aa`,
  ready: () => true,
})

const emit = defineEmits<{ (e: 'select', symbol: string): void }>()

const open = ref(false)
const query = ref('')
const active = ref(0)
const box = ref<HTMLElement | null>(null)
const field = ref<HTMLInputElement | null>(null)

const matches = computed(() => {
  const q = query.value.trim().toLowerCase()
  const hit = (g: any) =>
    !q || g.symbol.toLowerCase().includes(q) ||
    (g.protein_name || '').toLowerCase().includes(q)
  const found = props.genes.filter(hit)
  return [...found].sort((a, b) => {
    const r = Number(props.ready(b)) - Number(props.ready(a))
    if (r) return r
    // Exact prefix matches first, so typing "TP5" surfaces TP53 above TP53BP1.
    const q2 = query.value.trim().toLowerCase()
    const pa = Number(b.symbol.toLowerCase().startsWith(q2)) -
               Number(a.symbol.toLowerCase().startsWith(q2))
    return pa || a.symbol.localeCompare(b.symbol)
  })
})

async function show() {
  if (props.disabled) return
  open.value = true
  query.value = ''
  active.value = Math.max(0, matches.value.findIndex((g) => g.symbol === props.modelValue))
  await nextTick()
  field.value?.focus()
  scrollActive()
}

function hide() {
  open.value = false
}

function choose(symbol: string) {
  hide()
  if (symbol !== props.modelValue) emit('select', symbol)
}

function move(step: number) {
  if (!matches.value.length) return
  active.value = (active.value + step + matches.value.length) % matches.value.length
  scrollActive()
}

function scrollActive() {
  nextTick(() => {
    box.value?.querySelector('.row.active')?.scrollIntoView({ block: 'nearest' })
  })
}

function onBlur(e: FocusEvent) {
  const next = e.relatedTarget as Node | null
  if (!next || !box.value?.contains(next)) hide()
}
</script>

<template>
  <div class="picker" @keydown.escape="hide">
    <button v-if="!open" class="current" :disabled="disabled" @click="show">
      <span class="sym">{{ modelValue }}</span>
      <span class="chev">▾</span>
    </button>

    <div v-else ref="box" class="popup" @focusout="onBlur">
      <input
        ref="field"
        v-model="query"
        class="filter"
        spellcheck="false"
        placeholder="Filter genes…"
        @keydown.down.prevent="move(1)"
        @keydown.up.prevent="move(-1)"
        @keydown.enter.prevent="matches[active] && choose(matches[active].symbol)"
      />
      <div class="list">
        <button
          v-for="(g, i) in matches"
          :key="g.symbol"
          class="row"
          :class="{ active: i === active, current: g.symbol === modelValue }"
          @mousemove="active = i"
          @click="choose(g.symbol)"
        >
          <span class="dot" :class="{ pending: !ready(g) }"></span>
          <span class="sym">{{ g.symbol }}</span>
          <span class="meta">{{ label(g) }}</span>
        </button>
        <p v-if="!matches.length" class="empty">No gene matches “{{ query }}”</p>
      </div>
    </div>
  </div>
</template>

<style scoped>
.picker {
  position: relative;
  font: 13px/1.5 system-ui, sans-serif;
}

.current, .filter {
  font: inherit;
  font-size: 13px;
  padding: 5px 8px;
  border: 1px solid #d0d0d0;
  border-radius: 4px;
  background: #fff;
  color: #222;
}

.current {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 150px;
  cursor: pointer;
  text-align: left;
}

.current:disabled { opacity: 0.5; cursor: default; }
.chev { margin-left: auto; color: #999; font-size: 11px; }
.sym { font-weight: 600; }

.popup {
  position: absolute;
  top: 0;
  left: 0;
  z-index: 20;
  width: 330px;
  background: #fff;
  border: 1px solid #c4c4c4;
  border-radius: 4px;
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.14);
}

.filter {
  width: 100%;
  border: 0;
  border-bottom: 1px solid #eee;
  border-radius: 4px 4px 0 0;
  outline: none;
}

.list { max-height: 300px; overflow-y: auto; }

.row {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 5px 9px;
  border: 0;
  background: none;
  font: inherit;
  font-size: 12px;
  color: #333;
  text-align: left;
  cursor: pointer;
}

.row.active { background: #eef3fb; }
.row.current .sym { color: #0a58ca; }
.meta { margin-left: auto; color: #888; font-size: 11px; }

.dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #7bc47f;
  flex: 0 0 auto;
}
.dot.pending { background: #d9d9d9; }

.empty { padding: 10px; margin: 0; color: #888; font-size: 12px; }
</style>
