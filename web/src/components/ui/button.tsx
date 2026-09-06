import * as React from 'react'
import { cn } from '@/lib/utils'

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'default' | 'destructive' | 'danger' | 'outline' | 'secondary' | 'ghost' | 'link'
  size?: 'default' | 'sm' | 'lg' | 'icon'
  asChild?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'default', size = 'default', asChild = false, ...props }, ref) => {
    const Comp = asChild ? 'span' : 'button'
    const baseStyles = 'inline-flex items-center justify-center rounded-xl text-sm font-medium transition-all duration-200 ease-spring focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500 focus-visible:ring-offset-2 focus-visible:ring-offset-surface-950 disabled:pointer-events-none disabled:opacity-50'
    
    const variants = {
      default: 'bg-gradient-to-r from-primary-500 to-primary-600 text-white hover:from-primary-600 hover:to-primary-700 active:from-primary-700 active:to-primary-800 shadow-[0_4px_16px_rgba(20,184,166,0.3)] hover:shadow-[0_8px_24px_rgba(20,184,166,0.4)]',
      danger: 'bg-gradient-to-r from-danger-500 to-danger-600 text-white hover:from-danger-600 hover:to-danger-700 active:from-danger-700 active:to-danger-800 shadow-[0_4px_16px_rgba(239,68,68,0.3)] hover:shadow-[0_8px_24px_rgba(239,68,68,0.4)]',
      destructive: 'bg-gradient-to-r from-danger-500 to-danger-600 text-white hover:from-danger-600 hover:to-danger-700 active:from-danger-700 active:to-danger-800 shadow-[0_4px_16px_rgba(239,68,68,0.3)] hover:shadow-[0_8px_24px_rgba(239,68,68,0.4)]',
      outline: 'bg-transparent text-primary-400 border-2 border-primary-500/50 hover:bg-primary-500/10 hover:border-primary-500 active:bg-primary-500/20',
      secondary: 'bg-surface-800/50 text-surface-100 border border-surface-700 hover:bg-surface-800 hover:border-surface-600 active:bg-surface-900',
      ghost: 'text-surface-400 hover:text-surface-100 hover:bg-surface-800/50 active:bg-surface-800',
      link: 'text-primary-400 underline-offset-4 hover:underline',
    }
    
    const sizes = {
      default: 'h-10 px-4 py-2',
      sm: 'h-9 rounded-lg px-3',
      lg: 'h-11 rounded-xl px-8',
      icon: 'h-10 w-10',
    }
    
    return (
      <Comp
        className={cn(baseStyles, variants[variant], sizes[size], className)}
        ref={ref}
        {...props}
      />
    )
  }
)

Button.displayName = 'Button'

export { Button }