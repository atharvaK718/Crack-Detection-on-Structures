import * as React from 'react'
import { cn } from '@/lib/utils'

export interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {}

const Select = React.forwardRef<HTMLSelectElement, SelectProps>(
  ({ className, children, ...props }, ref) => (
    <select
      className={cn(
        'flex h-10 w-full appearance-none rounded-lg border border-concrete-600 bg-concrete-900/50 px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-crack focus:border-transparent disabled:cursor-not-allowed disabled:opacity-50',
        className
      )}
      ref={ref}
      {...props}
    >
      {children}
    </select>
  )
)
Select.displayName = 'Select'

export { Select }