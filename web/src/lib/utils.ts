import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatNumber(num: number): string {
  if (num >= 1000000) {
    return (num / 1000000).toFixed(1) + 'M'
  }
  if (num >= 1000) {
    return (num / 1000).toFixed(1) + 'K'
  }
  return num.toFixed(1)
}

export function formatPercentage(num: number): string {
  return num.toFixed(2) + '%'
}

export function formatTime(ms: number): string {
  if (ms < 1000) {
    return ms.toFixed(0) + 'ms'
  }
  return (ms / 1000).toFixed(2) + 's'
}

export const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export type AnalysisReport = {
  annotated_image: string
  crack_detected: boolean
  crack_area_percent: number
  crack_regions: number
  inference_time_ms: number
}

export type LiveStats = {
  fps: number
  latency: number
  coverage: number
}

export type CameraDevice = {
  deviceId: string
  label: string
  kind: 'videoinput'
  isPhoneCamera?: boolean
}