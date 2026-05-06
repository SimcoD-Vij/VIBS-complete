import { useRef, useEffect, useState } from 'react'
import ForceGraph2D from 'react-force-graph-2d'

const NODE_COLORS = {
  speaker:   '#534AB7',
  topic:     '#1D9E75',
  claim:     '#D85A30',
  consensus: '#639922',
  shift:     '#BA7517',
}

const EDGE_COLORS = {
  argues:      '#378ADD',
  opposes:     '#D85A30',
  supports:    '#639922',
  introduces:  '#1D9E75',
  agrees_with: '#639922',
  shifts_to:   '#BA7517',
  builds_on:   '#534AB7',
}

export function MindMap({ graphData, onNodeClick }) {
  const fgRef = useRef()
  const containerRef = useRef()
  const [dims, setDims] = useState({ w: 600, h: 440 })
  const [selected, setSelected] = useState(null)

  useEffect(() => {
    if (!containerRef.current) return
    const ro = new ResizeObserver(([e]) => {
      const w = e.contentRect.width
      setDims({ w, h: Math.min(520, Math.max(380, w * 0.65)) })
    })
    ro.observe(containerRef.current)
    return () => ro.disconnect()
  }, [])

  if (!graphData?.nodes?.length) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-600 text-sm">
        No graph data available for this session.
      </div>
    )
  }

  const fgData = {
    nodes: graphData.nodes.map(n => ({ ...n })),
    links: graphData.edges.map(e => ({
      source: e.source, target: e.target,
      label: e.label, strength: e.strength, relation: e.relation,
    })),
  }

  const handleClick = (node) => {
    setSelected(node)
    fgRef.current?.centerAt(node.x, node.y, 600)
    fgRef.current?.zoom(2.4, 600)
    onNodeClick?.(node)
  }

  return (
    <div className="flex gap-4">
      {/* Graph */}
      <div
        ref={containerRef}
        className="flex-1 rounded-xl overflow-hidden border border-gray-800 bg-gray-950"
      >
        <ForceGraph2D
          ref={fgRef}
          graphData={fgData}
          width={dims.w}
          height={dims.h}
          backgroundColor="#030712"
          nodeLabel="label"
          nodeVal={n => (n.weight || 5) * 2.5}
          nodeColor={n => NODE_COLORS[n.type] || '#888780'}
          linkWidth={l => Math.max(0.5, (l.strength || 1) * 0.9)}
          linkColor={l => EDGE_COLORS[l.relation] || '#4B5563'}
          linkDirectionalArrowLength={5}
          linkDirectionalArrowRelPos={1}
          linkLabel="label"
          cooldownTicks={120}
          onNodeClick={handleClick}
          nodeCanvasObjectMode={() => 'after'}
          nodeCanvasObject={(node, ctx, globalScale) => {
            if (globalScale < 0.8) return
            const label = node.label?.length > 20 ? node.label.slice(0, 19) + '…' : node.label
            const fontSize = Math.max(8, 11 / globalScale)
            ctx.font = `${node.type === 'speaker' ? '600' : '400'} ${fontSize}px sans-serif`
            ctx.textAlign = 'center'
            ctx.textBaseline = 'middle'
            ctx.fillStyle = '#f1f5f9'
            ctx.fillText(label || '', node.x, node.y)
          }}
        />
      </div>

      {/* Selected node panel */}
      {selected && (
        <div className="w-56 flex-shrink-0 bg-gray-900 border border-gray-800 rounded-xl p-4">
          <div className="flex items-center gap-2 mb-3">
            <div
              className="w-3 h-3 rounded-full flex-shrink-0"
              style={{ background: NODE_COLORS[selected.type] || '#888' }}
            />
            <span className="text-xs font-bold text-gray-300 uppercase">{selected.type}</span>
            <button
              className="ml-auto text-gray-600 hover:text-gray-300 text-sm"
              onClick={() => setSelected(null)}
            >✕</button>
          </div>
          <h3 className="font-semibold text-white text-sm mb-2">{selected.label}</h3>
          <div className="text-xs text-gray-500 space-y-1">
            <div>Weight: <span className="text-gray-300">{selected.weight}</span></div>
          </div>
        </div>
      )}
    </div>
  )
}
