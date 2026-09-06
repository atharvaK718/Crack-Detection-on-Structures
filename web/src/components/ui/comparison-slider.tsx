import * as React from 'react'
import { cn } from '@/lib/utils'

interface ComparisonSliderProps {
  beforeSrc: string
  afterSrc: string
  beforeAlt?: string
  afterAlt?: string
  className?: string
}

export function ComparisonSlider({ 
  beforeSrc, 
  afterSrc, 
  beforeAlt = 'Before analysis', 
  afterAlt = 'After analysis',
  className 
}: ComparisonSliderProps) {
  const [position, setPosition] = React.useState(50)
  const containerRef = React.useRef<HTMLDivElement>(null)
  
  const handleMove = (e: MouseEvent | TouchEvent) => {
    if (!containerRef.current) return
    
    const rect = containerRef.current.getBoundingClientRect()
    const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX
    const newPosition = Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100))
    setPosition(newPosition)
  }
  
  const handleMouseDown = () => {
    window.addEventListener('mousemove', handleMove)
    window.addEventListener('mouseup', handleMouseUp)
  }
  
  const handleTouchStart = () => {
    window.addEventListener('touchmove', handleMove, { passive: true })
    window.addEventListener('touchend', handleTouchEnd)
  }
  
  const handleMouseUp = () => {
    window.removeEventListener('mousemove', handleMove)
    window.removeEventListener('mouseup', handleMouseUp)
  }
  
  const handleTouchEnd = () => {
    window.removeEventListener('touchmove', handleMove)
    window.removeEventListener('touchend', handleTouchEnd)
  }
  
  return (
    <div 
      ref={containerRef}
      className={cn('relative overflow-hidden rounded-xl bg-surface-900', className)}
      onMouseDown={handleMouseDown}
      onTouchStart={handleTouchStart}
      style={{ 
        aspectRatio: '16 / 9',
        minHeight: 'clamp(200px, 30vw, 400px)',
        width: '100%'
      }}
    >
      <div className="absolute inset-0">
        <img 
          src={beforeSrc} 
          alt={beforeAlt}
          className="w-full h-full object-cover"
          draggable={false}
        />
      </div>
      
      <div 
        className="absolute inset-0 overflow-hidden"
        style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}
      >
        <img 
          src={afterSrc} 
          alt={afterAlt}
          className="w-full h-full object-cover"
          draggable={false}
        />
      </div>
      
      <div
        className="absolute top-0 bottom-0 w-[4px] bg-primary-500 pointer-events-none z-10"
        style={{ left: `${position}%`, transform: 'translateX(-50%)' }}
        aria-hidden="true"
      >
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-primary-500 border-[3px] border-surface-950 shadow-lg" 
          style={{ 
            width: 'clamp(28px, 24px + 2vw, 40px)', 
            height: 'clamp(28px, 24px + 2vw, 40px)' 
          }}
        />
      </div>
      
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-2 text-xs text-surface-400 pointer-events-none flex-wrap justify-center px-2">
        <span className="flex items-center gap-1 whitespace-nowrap">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-primary-500 shrink-0">
            <path d="M21 15V5a2 2 0 0 0-2-2H9a2 2 0 0 0-2 2v10" />
            <polyline points="17 10 12 15 7 10" />
            <line x1="12" y1="15" x2="12" y2="3" />
          </svg>
          Before
        </span>
        <div className="w-2 h-2 rounded-full bg-primary-500 shrink-0" />
        <span className="flex items-center gap-1 whitespace-nowrap">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-primary-500 shrink-0">
            <path d="M3 9V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v10" />
            <polyline points="7 14 12 9 17 14" />
            <line x1="12" y1="9" x2="12" y2="21" />
          </svg>
          After
        </span>
      </div>
    </div>
  )
}