import { useState, useEffect, useRef, useCallback } from 'react'
import * as tf from '@tensorflow/tfjs'
import type { LiveStats } from '@/lib/utils'

const MODEL_URL = '/tfjs/model.json'
const IMG_SIZE = 256
const DEFAULT_THRESHOLD = 0.3

interface UseLiveInferenceReturn {
  model: tf.LayersModel | null
  isModelLoading: boolean
  modelError: string | null
  stats: LiveStats
  runInference: (video: HTMLVideoElement, canvas: HTMLCanvasElement) => Promise<tf.Tensor | null>
  setThreshold: (threshold: number) => void
  threshold: number
}

export function useLiveInference(): UseLiveInferenceReturn {
  const [model, setModel] = useState<tf.LayersModel | null>(null)
  const [isModelLoading, setIsModelLoading] = useState(true)
  const [modelError, setModelError] = useState<string | null>(null)
  const [threshold, setThreshold] = useState(DEFAULT_THRESHOLD)
  const [stats, setStats] = useState<LiveStats>({ fps: 0, latency: 0, coverage: 0 })
  
  const lastFrameTime = useRef(0)
  const frameCount = useRef(0)
  const inferenceQueue = useRef(false)
  const statsInterval = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    const loadModel = async () => {
      try {
        setIsModelLoading(true)
        await tf.ready()
        const loadedModel = await tf.loadLayersModel(MODEL_URL)
        setModel(loadedModel)
        setModelError(null)
        
        // Warm up
        const dummy = tf.zeros([1, IMG_SIZE, IMG_SIZE, 3])
        await loadedModel.predict(dummy)
        dummy.dispose()
      } catch (err) {
        console.error('Failed to load TF.js model:', err)
        setModelError('Failed to load model. Browser inference unavailable. Use server-side analysis instead.')
      } finally {
        setIsModelLoading(false)
      }
    }
    
    loadModel()
    
    // Stats calculation interval
    statsInterval.current = setInterval(() => {
      const now = performance.now()
      const elapsed = now - lastFrameTime.current
      if (elapsed > 0) {
        setStats(prev => ({
          ...prev,
          fps: Math.round((frameCount.current * 1000) / elapsed),
        }))
      }
      frameCount.current = 0
      lastFrameTime.current = now
    }, 1000)
    
    return () => {
      if (statsInterval.current) {
        clearInterval(statsInterval.current)
      }
    }
  }, [])

  const runInference = useCallback(async (
    video: HTMLVideoElement,
    canvas: HTMLCanvasElement
  ): Promise<tf.Tensor | null> => {
    if (!model || inferenceQueue.current) return null
    
    inferenceQueue.current = true
    const startTime = performance.now()
    
    try {
      // Draw video frame to canvas
      const ctx = canvas.getContext('2d', { willReadFrequently: true })
      if (!ctx) throw new Error('Canvas context unavailable')
      
      canvas.width = IMG_SIZE
      canvas.height = IMG_SIZE
      ctx.drawImage(video, 0, 0, IMG_SIZE, IMG_SIZE)
      
      // Get image data and preprocess
      const imageData = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE)
      const input = tf.tidy(() => {
        const tensor = tf.browser.fromPixels(imageData)
        const resized = tf.image.resizeBilinear(tensor, [IMG_SIZE, IMG_SIZE])
        const normalized = resized.div(255.0)
        return normalized.expandDims(0)
      })
      
      // Run inference
      const prediction = await model.predict(input) as tf.Tensor
      const latency = performance.now() - startTime
      
      // Process output
      const mask = tf.tidy(() => {
        const probs = prediction.squeeze()
        const binary = probs.greater(threshold)
        return binary
      })
      
      // Calculate coverage
      const coverageData = await mask.data()
      let sum = 0
      for (let i = 0; i < coverageData.length; i++) {
        sum += coverageData[i]
      }
      const coverage = (sum / coverageData.length) * 100
      
      // Update stats
      frameCount.current++
      setStats(prev => ({
        ...prev,
        latency: Math.round(latency * 10) / 10,
        coverage: Math.round(coverage * 100) / 100,
      }))
      
      input.dispose()
      prediction.dispose()
      
      return mask
    } catch (err) {
      console.error('Inference error:', err)
      return null
    } finally {
      inferenceQueue.current = false
    }
  }, [model, threshold])

  return {
    model,
    isModelLoading,
    modelError,
    stats,
    runInference,
    setThreshold,
    threshold,
  }
}