import { useState, useEffect, useCallback, useRef } from 'react'
import type { CameraDevice } from '@/lib/utils'

interface UseCameraReturn {
  stream: MediaStream | null
  devices: CameraDevice[]
  selectedDeviceId: string | null
  isLoading: boolean
  error: string | null
  permissionState: 'not-requested' | 'granted' | 'denied'
  startCamera: (deviceId?: string) => Promise<void>
  stopCamera: () => void
  switchCamera: () => Promise<void>
  enumerateDevices: () => Promise<void>
  requestPermission: () => Promise<void>
  setError: (error: string | null) => void
}

function isLikelyPhoneCamera(device: MediaDeviceInfo): boolean {
  const label = device.label.toLowerCase()
  return (
    label.includes('phone') ||
    label.includes('mobile') ||
    label.includes('android') ||
    label.includes('iphone') ||
    label.includes('ipad') ||
    label.includes('usb') ||
    label.includes('external') ||
    label.includes('webcam') && (label.includes('hd') || label.includes('4k') || label.includes('1080'))
  )
}

function getCameraPriority(device: CameraDevice): number {
  const label = device.label.toLowerCase()
  // Priority: phone/external > rear/back > front/user > others
  if (label.includes('phone') || label.includes('mobile') || label.includes('android') || label.includes('iphone')) return 1
  if (label.includes('usb') || label.includes('external')) return 2
  if (label.includes('back') || label.includes('rear') || label.includes('environment')) return 3
  if (label.includes('front') || label.includes('user') || label.includes('face')) return 4
  return 5
}

export function useCamera(): UseCameraReturn {
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [devices, setDevices] = useState<CameraDevice[]>([])
  const [selectedDeviceId, setSelectedDeviceId] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [permissionState, setPermissionState] = useState<'not-requested' | 'granted' | 'denied'>('not-requested')
  const facingModeRef = useRef<'user' | 'environment'>('environment')
  const deviceIndexRef = useRef(0)

  const enumerateDevices = useCallback(async () => {
    try {
      const mediaDevices = await navigator.mediaDevices.enumerateDevices()
      const videoDevices = mediaDevices
        .filter((device): device is MediaDeviceInfo => device.kind === 'videoinput')
        .map((device) => ({
          deviceId: device.deviceId,
          label: device.label || `Camera ${device.deviceId.slice(0, 8)}`,
          kind: 'videoinput' as const,
          isPhoneCamera: isLikelyPhoneCamera(device),
        }))
        .sort((a, b) => getCameraPriority(a) - getCameraPriority(b))
      
      setDevices(videoDevices)
      
      if (videoDevices.length > 0 && !selectedDeviceId) {
        // Prefer phone/external cameras first, then rear cameras
        const preferredDevice = videoDevices[0]
        setSelectedDeviceId(preferredDevice.deviceId)
      }
    } catch (err) {
      console.error('Failed to enumerate devices:', err)
      setError('Failed to access camera devices')
    }
  }, [selectedDeviceId])

  const requestPermission = useCallback(async () => {
    try {
      setError(null)
      // Request permission with a minimal constraint to trigger permission prompt
      const tempStream = await navigator.mediaDevices.getUserMedia({ video: true })
      tempStream.getTracks().forEach(track => track.stop())
      setPermissionState('granted')
      // Re-enumerate to get device labels
      await enumerateDevices()
    } catch (err) {
      console.error('Permission request failed:', err)
      if (err instanceof DOMException && err.name === 'NotAllowedError') {
        setPermissionState('denied')
        setError('Camera permission denied. Please allow camera access in your browser settings.')
      } else {
        setPermissionState('denied')
        setError('Failed to request camera permission')
      }
    }
  }, [enumerateDevices])

  const startCamera = useCallback(async (deviceId?: string) => {
    setIsLoading(true)
    setError(null)
    
    try {
      if (stream) {
        stream.getTracks().forEach(track => track.stop())
      }
      
      const targetDeviceId = deviceId || selectedDeviceId
      const constraints: MediaStreamConstraints = {
        video: {
          deviceId: targetDeviceId ? { exact: targetDeviceId } : undefined,
          facingMode: targetDeviceId ? undefined : facingModeRef.current,
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      }
      
      const newStream = await navigator.mediaDevices.getUserMedia(constraints)
      setStream(newStream)
      setPermissionState('granted')
      
      if (!targetDeviceId) {
        const track = newStream.getVideoTracks()[0]
        const settings = track.getSettings()
        if (settings.deviceId) {
          setSelectedDeviceId(settings.deviceId)
        }
      }
    } catch (err) {
      console.error('Failed to start camera:', err)
      if (err instanceof DOMException) {
        switch (err.name) {
          case 'NotAllowedError':
            setPermissionState('denied')
            setError('Camera permission denied. Please allow camera access in your browser settings.')
            break
          case 'NotFoundError':
            setError('No camera device found. Please connect a camera and try again.')
            break
          case 'NotReadableError':
            setError('Camera is already in use by another application.')
            break
          default:
            setError(`Camera error: ${err.message}`)
        }
      } else {
        setError('Failed to start camera')
      }
      setStream(null)
    } finally {
      setIsLoading(false)
    }
  }, [stream, selectedDeviceId])

  const stopCamera = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop())
      setStream(null)
    }
  }, [stream])

  const switchCamera = useCallback(async () => {
    if (devices.length <= 1) return
    
    deviceIndexRef.current = (deviceIndexRef.current + 1) % devices.length
    const nextDevice = devices[deviceIndexRef.current]
    await startCamera(nextDevice.deviceId)
  }, [devices, startCamera])

  useEffect(() => {
    enumerateDevices()
    
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop())
      }
    }
  }, [enumerateDevices, stream])

  return {
    stream,
    devices,
    selectedDeviceId,
    isLoading,
    error,
    permissionState,
    startCamera,
    stopCamera,
    switchCamera,
    enumerateDevices,
    requestPermission,
    setError,
  }
}