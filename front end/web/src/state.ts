import { reactive } from 'vue'

export type View = 'lookup' | 'heatmap' | 'structure' | 'metrics'

export const VIEWS: View[] = ['lookup', 'heatmap', 'structure', 'metrics']

export const API = 'http://localhost:8000'

function fromUrl() {
  const q = new URLSearchParams(location.search)
  const view = q.get('view') as View
  return {
    view: VIEWS.includes(view) ? view : ('heatmap' as View),
    gene: (q.get('gene') || 'TP53').toUpperCase(),
    variant: q.get('variant') || '',
  }
}

// Shared so the selected gene survives a tab switch. Each view unmounts when
// hidden - the heat map sizes its canvas at draw time and cannot be laid out
// while invisible - so component-local state does not survive.
export const state = reactive(fromUrl())

export function syncUrl(replace = false) {
  const q = new URLSearchParams()
  q.set('view', state.view)
  q.set('gene', state.gene)
  if (state.variant) q.set('variant', state.variant)
  const url = `${location.pathname}?${q.toString()}`
  if (replace) history.replaceState(null, '', url)
  else history.pushState(null, '', url)
}

window.addEventListener('popstate', () => Object.assign(state, fromUrl()))
