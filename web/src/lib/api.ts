import { API_URL, type AnalysisReport } from './utils'

export type { AnalysisReport }

export async function analyzeImage(file: File): Promise<AnalysisReport> {
  const formData = new FormData()
  formData.append('file', file, 'inspection.png')
  
  const response = await fetch(`${API_URL}/api/analyze`, {
    method: 'POST',
    body: formData,
  })
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Analysis failed' }))
    throw new Error(error.detail || 'Analysis failed')
  }
  
  return response.json()
}

export async function checkHealth(): Promise<{ status: string }> {
  const response = await fetch(`${API_URL}/health`)
  if (!response.ok) {
    throw new Error('Backend health check failed')
  }
  return response.json()
}