import * as React from "react"

export interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "default" | "secondary" | "destructive" | "outline"
}

function Badge({ className = "", variant = "default", ...props }: BadgeProps) {
  const variants = {
    default: "bg-primary text-primary-foreground hover:bg-primary/80 bg-purple-600 text-white",
    secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/80 bg-gray-200 text-gray-900",
    destructive: "bg-destructive text-destructive-foreground hover:bg-destructive/80 bg-red-600 text-white",
    outline: "text-foreground border border-gray-300"
  }

  return (
    <div
      className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 ${variants[variant]} ${className}`}
      {...props}
    />
  )
}

export { Badge }
