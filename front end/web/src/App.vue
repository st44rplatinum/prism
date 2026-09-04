<script setup lang="ts">
import { ref } from 'vue'
import Heatmap from './components/Heatmap.vue'
import Metrics from './components/Metrics.vue'
import Structure from './components/Structure.vue'

type View = 'heatmap' | 'structure' | 'metrics'
const view = ref<View>('heatmap')
</script>

<template>
  <div class="app">
    <nav>
      <span class="brand">ESM-2 variant effect predictor</span>
      <button :class="{ on: view === 'heatmap' }" @click="view = 'heatmap'">Heatmap</button>
      <button :class="{ on: view === 'structure' }" @click="view = 'structure'">Structure</button>
      <button :class="{ on: view === 'metrics' }" @click="view = 'metrics'">Metrics</button>
    </nav>

    <!-- v-if, not v-show: the heat map sizes its canvas at draw time, and a
         hidden canvas has no layout, so it must not be mounted while unseen. -->
    <Heatmap v-if="view === 'heatmap'" />
    <Structure v-else-if="view === 'structure'" />
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
