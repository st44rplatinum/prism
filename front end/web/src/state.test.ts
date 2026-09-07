import { beforeEach, describe, expect, it, vi } from 'vitest'

async function load(search: string) {
  history.replaceState(null, '', `/${search}`)
  // Fresh module per case: the store reads the URL once, at import.
  vi.resetModules()
  return await import('./state')
}

describe('url state', () => {
  beforeEach(() => history.replaceState(null, '', '/'))

  it('defaults when the query string is empty', async () => {
    const { state } = await load('')
    expect(state.view).toBe('heatmap')
    expect(state.gene).toBe('TP53')
    expect(state.variant).toBe('')
  })

  it('reads view, gene and variant', async () => {
    const { state } = await load('?view=lookup&gene=BRCA1&variant=R175H')
    expect(state.view).toBe('lookup')
    expect(state.gene).toBe('BRCA1')
    expect(state.variant).toBe('R175H')
  })

  it('falls back on an unknown view rather than rendering nothing', async () => {
    const { state } = await load('?view=nonsense')
    expect(state.view).toBe('heatmap')
  })

  it('upper-cases the gene so a hand-typed link works', async () => {
    const { state } = await load('?gene=brca1')
    expect(state.gene).toBe('BRCA1')
  })

  it('writes state back to the query string', async () => {
    const { state, syncUrl } = await load('')
    state.view = 'structure'
    state.gene = 'SCN5A'
    syncUrl()
    expect(location.search).toContain('view=structure')
    expect(location.search).toContain('gene=SCN5A')
  })

  it('omits an empty variant', async () => {
    const { state, syncUrl } = await load('?view=lookup&gene=TP53&variant=R175H')
    state.variant = ''
    syncUrl()
    expect(location.search).not.toContain('variant')
  })

  it('replace does not grow history', async () => {
    const { state, syncUrl } = await load('')
    const before = history.length
    state.gene = 'CFTR'
    syncUrl(true)
    expect(history.length).toBe(before)
    expect(location.search).toContain('gene=CFTR')
  })

  it('restores state on popstate', async () => {
    const { state } = await load('?view=lookup&gene=TP53&variant=R175H')
    history.replaceState(null, '', '/?view=metrics&gene=CFTR')
    window.dispatchEvent(new PopStateEvent('popstate'))
    expect(state.view).toBe('metrics')
    expect(state.gene).toBe('CFTR')
    expect(state.variant).toBe('')
  })
})
