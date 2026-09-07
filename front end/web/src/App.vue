<script setup lang="ts">
import { onMounted } from 'vue'
import Heatmap from './components/Heatmap.vue'
import Lookup from './components/Lookup.vue'
import Metrics from './components/Metrics.vue'
import Structure from './components/Structure.vue'
import { state, syncUrl, type View } from './state'

const TABS: { id: View; label: string }[] = [
  { id: 'lookup', label: 'Lookup' },
  { id: 'heatmap', label: 'Heatmap' },
  { id: 'structure', label: 'Structure' },
  { id: 'metrics', label: 'Metrics' },
]

function show(view: View) {
  if (view === state.view) return
  state.view = view
  syncUrl()
}

// Stamp the current state onto the address bar on first load, so a reload or a
// copied link lands in the same place.
onMounted(() => syncUrl(true))
</script>

<template>
  <div class="app">
    <nav>
      <span class="brand">ESM-2 variant effect predictor</span>
      <button v-for="t in TABS" :key="t.id"
              :class="{ on: state.view === t.id }" @click="show(t.id)">
        {{ t.label }}
      </button>
    </nav>

    <!-- v-if, not v-show: the heat map sizes its canvas at draw time, and a
         hidden canvas has no layout, so it must not be mounted while unseen.
         The selected gene lives in the shared store to survive the unmount. -->
    <Lookup v-if="state.view === 'lookup'" />
    <Heatmap v-else-if="state.view === 'heatmap'" />
    <Structure v-else-if="state.view === 'structure'" />
    <Metrics v-else />
  </div>
</template>

<style scoped>
.app {
  padding: 12px;
}

nav {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 10px;
}

.brand {
  font: 600 13px/1.5 system-ui, sans-serif;
  color: #888;
  margin-right: 10px;
}

nav button {
  font: 12px/1.5 system-ui, sans-serif;
  padding: 4px 12px;
  border: 1px solid #d0d0d0;
  background: #f7f7f7;
  color: #444;
  border-radius: 4px;
  cursor: pointer;
}

nav button.on {
  background: #222;
  border-color: #222;
  color: #fff;
}
</style>
