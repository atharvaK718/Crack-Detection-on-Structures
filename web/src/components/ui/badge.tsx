import * as React from 'react'
import { cn } from '@/lib/utils'

export interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'secondary' | 'destructive' | 'outline' | 'success' | 'warning' | 'primary' | 'accent' | 'neutral'
}

const Badge = React.forwardRef<HTMLDivElement, BadgeProps>(
  ({ className, variant = 'default', ...props }, ref) => {
    const variants = {
      default: 'bg-primary-500/20 text-primary-400 border border-primary-500/30',
      secondary: 'bg-surface-700 text-surface-300 border border-surface-600',
      destructive: 'bg-danger-500/20 text-danger-400 border border-danger-500/30',
      outline: 'text-surface-300 border border-surface-600',
      success: 'bg-success-500/20 text-success-400 border border-success-500/30',
      warning: 'bg-warning-500/20 text-warning-400 border border-warning-500/30',
      primary: 'bg-primary-500/20 text-primary-400 border border-primary-500/30',
      accent: 'bg-accent-500/20 text-accent-400 border border-accent-500/30',
      neutral: 'bg-surface-700 text-surface-300 border border-surface-600',
    }
    
    return (
      <div
        ref={ref}
        className={cn(
          'inline-flex items-center rounded-full px-3 py-1 text-xs font-medium',
          variants[variant],
          className
        )}
        {...props}
      />
    )
  }
)
Badge.displayName = 'Badge'

export { Badge }