import { Routes, Route, Link, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { useRef, useEffect, useState, useCallback } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { 
  Camera, RotateCcw, CheckCircle, AlertTriangle, Gauge, WifiOff, 
  Download, Upload, Image, Trash2,
  Target, Layers, Activity,
  X as XIcon, Loader2, Monitor, Smartphone,
  Info, ScanLine, ScanSearch, ImageUp, ArrowRight,
  ShieldCheck, Wifi, LockKeyhole, Crosshair, Send, FileUp, Images, FileDown
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { analyzeImage, type AnalysisReport } from '@/lib/api'
import { useCamera } from '@/hooks/useCamera'
import { useLiveInference } from '@/hooks/useLiveInference'

/* ============================================
   LANDING PAGE
   ============================================ */
function LandingPage() {
  return (
    <motion.div 
      initial="hidden"
      animate="visible"
      variants={{ 
        hidden: { opacity: 0 }, 
        visible: { opacity: 1, transition: { staggerChildren: 0.1 } } 
      }}
      className="min-h-screen relative overflow-hidden safe-area"
    >
      {/* Animated background */}
      <div className="fixed inset-0 hero-pattern" />
      <div className="fixed inset-0 grid-pattern" />
      <div className="fixed inset-0 noise-texture" />
      
      {/* Floating orbs - responsive sizes */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <motion.div 
          className="absolute top-1/4 left-1/4 rounded-full bg-primary-500/10 blur-3xl"
          style={{
            width: 'clamp(16rem, 20vw, 24rem)',
            height: 'clamp(16rem, 20vw, 24rem)',
          }}
          animate={{ scale: [1, 1.1, 1], x: [0, 20, 0], y: [0, -20, 0] }}
          transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
        />
        <motion.div 
          className="absolute bottom-1/4 right-1/4 rounded-full bg-accent-500/10 blur-3xl"
          style={{
            width: 'clamp(12rem, 15vw, 18rem)',
            height: 'clamp(12rem, 15vw, 18rem)',
          }}
          animate={{ scale: [1, 1.15, 1], x: [0, -15, 0], y: [0, 15, 0] }}
          transition={{ duration: 15, repeat: Infinity, ease: "linear", delay: 5 }}
        />
        <motion.div 
          className="absolute top-1/2 left-1/2 rounded-full bg-primary-500/5 blur-3xl"
          style={{
            width: 'clamp(10rem, 12vw, 16rem)',
            height: 'clamp(10rem, 12vw, 16rem)',
          }}
          animate={{ scale: [1, 1.2, 1] }}
          transition={{ duration: 25, repeat: Infinity, ease: "linear", delay: 10 }}
        />
      </div>

      {/* Navigation */}
      <header className="relative z-20 bg-surface-950/75 backdrop-blur-xl shadow-[0_10px_30px_rgba(0,0,0,0.18)] sticky top-0 safe-top">
        <div className="container">
          <div className="flex h-16 items-center justify-between">
            <Link to="/" className="flex items-center gap-2" aria-label="Crack Detection on Structures Home">
              <motion.div
                whileHover={{ scale: 1.1, rotate: 5 }}
                transition={{ type: "spring", stiffness: 400, damping: 17 }}
                className="p-2 rounded-xl bg-gradient-to-br from-primary-500 to-primary-600"
              >
                <ScanLine className="h-5 w-5 text-white" />
              </motion.div>
              <span className="font-display font-bold text-xl text-surface-50">Crack Detection on Structures</span>
            </Link>
            <nav className="hidden md:flex items-center gap-1">
              <Link 
                to="/" 
                className="px-4 py-2 rounded-xl text-sm font-medium text-surface-400 hover:text-surface-100 hover:bg-surface-800/50 transition-all duration-200"
              >
                Home
              </Link>
            </nav>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <main className="relative z-10">
        <section className="min-h-[calc(100dvh-4rem)] flex items-center py-6 md:py-10 lg:py-14">
          <div className="container w-full">
            <div className="grid md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] items-start gap-8 md:gap-12 lg:gap-16 xl:gap-20">
              {/* Left Column - Hero Content */}
              <div className="max-w-2xl mx-auto md:mx-0 text-center md:text-left">
                {/* Badge */}
                <motion.div 
                  className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full 
                             glass border border-primary-500/20 mb-5"
                  variants={{ 
                    hidden: { opacity: 0, y: 20 }, 
                    visible: { opacity: 1, y: 0, transition: { duration: 0.6, ease: "easeOut" } } 
                  }}
                >
                  <ScanSearch className="h-4 w-4 text-primary-400" />
                  <span className="text-sm font-medium text-surface-300">Structural Intelligence Platform</span>
                </motion.div>

                {/* Title */}
                <motion.h1 
                  className="hero-title text-balance mb-5"
                  variants={{ 
                    hidden: { opacity: 0, y: 30 }, 
                    visible: { opacity: 1, y: 0, transition: { duration: 0.8, ease: "easeOut" } } 
                  }}
                >
                  Crack Detection
                  <br />
                  <span className="gradient-text">Redefined</span>
                </motion.h1>

                {/* Subtitle */}
                <motion.p 
                  className="text-body-lg text-surface-400 max-w-xl mx-auto md:mx-0 mb-6 leading-relaxed"
                  variants={{ 
                    hidden: { opacity: 0, y: 20 }, 
                    visible: { opacity: 1, y: 0, transition: { duration: 0.8, ease: "easeOut", delay: 0.2 } } 
                  }}
                >
                  U-Net deep learning segmentation for real-time structural crack detection. 
                  Browser-based live scanning with server-grade analysis and contour extraction.
                </motion.p>

                {/* CTA Buttons */}
                <motion.div 
                  className="flex flex-col sm:flex-row items-center md:justify-start gap-3"
                  variants={{ 
                    hidden: { opacity: 0, y: 20 }, 
                    visible: { opacity: 1, y: 0, transition: { duration: 0.8, ease: "easeOut", delay: 0.4 } } 
                  }}
                >
                  <Link to="/live">
                    <Button className="group w-full sm:w-auto gap-3 px-7 py-3.5 text-base rounded-xl" size="lg">
                      <Camera className="h-5 w-5" />
                      Live Camera Scan
                      <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
                    </Button>
                  </Link>
                  <Link to="/analyze">
                    <Button variant="outline" className="group w-full sm:w-auto gap-3 px-7 py-3.5 text-base rounded-xl" size="lg">
                      <ImageUp className="h-5 w-5" />
                      Upload & Analyze
                      <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
                    </Button>
                  </Link>
                </motion.div>

                {/* Trust indicators */}
                <motion.div 
                  className="mt-6 flex flex-wrap items-center justify-center md:justify-start gap-x-5 gap-y-2 text-xs text-surface-500"
                  variants={{ 
                    hidden: { opacity: 0, y: 20 }, 
                    visible: { opacity: 1, y: 0, transition: { duration: 0.8, ease: "easeOut", delay: 0.6 } } 
                  }}
                >
                  <div className="flex items-center gap-2">
                    <Activity className="h-4 w-4 text-primary-500 shrink-0" />
                    <span>Real-time Inference</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <ShieldCheck className="h-4 w-4 text-accent-500 shrink-0" />
                    <span>Server-grade Analysis</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Wifi className="h-4 w-4 text-success-500 shrink-0" />
                    <span>Works Offline</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <LockKeyhole className="h-4 w-4 text-surface-500 shrink-0" />
                    <span>Privacy First</span>
                  </div>
                </motion.div>
              </div>

              {/* Right Column - Workflow Cards */}
              <div className="w-full max-w-3xl mx-auto md:mx-0">
                <motion.div 
                  className="text-center md:text-left mb-5"
                  variants={{ 
                    hidden: { opacity: 0, y: 20 }, 
                    visible: { opacity: 1, y: 0, transition: { duration: 0.6 } } 
                  }}
                >
                  <p className="text-xs uppercase tracking-[0.18em] text-primary-400 mb-2">
                    Inspection workflows
                  </p>
                  <h2 className="text-heading-xl md:text-display-sm font-bold mb-2">
                    Choose your way in.
                  </h2>
                  <p className="text-body-sm text-surface-400">
                    Start with a live scan or inspect an existing image.
                  </p>
                </motion.div>

                <div className="grid sm:grid-cols-2 gap-4 md:gap-5">
                  {/* Live Scan Card */}
                  <motion.article
                    variants={{ 
                      hidden: { opacity: 0, y: 30 }, 
                      visible: { opacity: 1, y: 0, transition: { duration: 0.6, ease: "easeOut" } } 
                    }}
                    className="h-full"
                  >
                    <Link to="/live" className="block h-full">
                      <div className="card-interactive group h-full relative overflow-hidden">
                        {/* Top accent bar */}
                        <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-primary-500 to-accent-500 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                        
                        <div className="relative z-10 space-y-4">
                          <div className="flex items-center gap-3">
                            <div className="p-3 rounded-xl bg-primary-500/10 border border-primary-500/20 group-hover:border-primary-500/40 transition-all duration-300 shrink-0">
                              <ScanLine className="h-6 w-6 text-primary-400" />
                            </div>
                            <div className="min-w-0">
                              <h3 className="text-heading-md font-bold text-surface-50 truncate">Live Camera Scan</h3>
                              <p className="text-body-sm text-surface-500 mt-0.5">Real-time client-side inference</p>
                            </div>
                          </div>

                          <ul className="space-y-4">
                            {[
                              { icon: Camera, text: 'Full-bleed camera feed with live crack overlay', color: 'primary' },
                              { icon: Crosshair, text: 'Live HUD with FPS, latency, and crack coverage', color: 'accent' },
                              { icon: Send, text: 'Capture & send to server for full analysis', color: 'success' },
                            ].map((item, i) => (
                              <motion.li 
                                key={i}
                                className="flex items-start gap-3"
                                variants={{ 
                                  hidden: { opacity: 0, x: -20 }, 
                                  visible: { opacity: 1, x: 0, transition: { delay: 0.1 * i } } 
                                }}
                              >
                                <motion.div
                                  whileHover={{ scale: 1.1, rotate: 3 }}
                                  className={`p-2 rounded-lg bg-${item.color}-500/10 border border-${item.color}-500/20 shrink-0`}
                                >
                                  <item.icon className={`h-5 w-5 text-${item.color}-400`} />
                                </motion.div>
                                <span className="text-body-sm text-surface-300 mt-0.5 leading-relaxed">{item.text}</span>
                              </motion.li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    </Link>
                  </motion.article>

                  {/* Upload & Analyze Card */}
                  <motion.article
                    variants={{ 
                      hidden: { opacity: 0, y: 30 }, 
                      visible: { opacity: 1, y: 0, transition: { duration: 0.6, ease: "easeOut", delay: 0.1 } } 
                    }}
                    className="h-full"
                  >
                    <Link to="/analyze" className="block h-full">
                      <div className="card-interactive group h-full relative overflow-hidden">
                        <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-accent-500 to-success-500 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                        
                        <div className="relative z-10 space-y-4">
                          <div className="flex items-center gap-3">
                            <div className="p-3 rounded-xl bg-accent-500/10 border border-accent-500/20 group-hover:border-accent-500/40 transition-all duration-300 shrink-0">
                              <ImageUp className="h-6 w-6 text-accent-400" />
                            </div>
                            <div className="min-w-0">
                              <h3 className="text-heading-md font-bold text-surface-50 truncate">Upload & Analyze</h3>
                              <p className="text-body-sm text-surface-500 mt-0.5">Server-side full report</p>
                            </div>
                          </div>

                          <ul className="space-y-4">
                            {[
                              { icon: FileUp, text: 'Drag-and-drop or file picker upload', color: 'accent' },
                              { icon: Images, text: 'Before/after comparison slider', color: 'success' },
                              { icon: FileDown, text: 'Download annotated PNG report', color: 'primary' },
                            ].map((item, i) => (
                              <motion.li 
                                key={i}
                                className="flex items-start gap-3"
                                variants={{ 
                                  hidden: { opacity: 0, x: -20 }, 
                                  visible: { opacity: 1, x: 0, transition: { delay: 0.1 * i } } 
                                }}
                              >
                                <motion.div
                                  whileHover={{ scale: 1.1, rotate: 3 }}
                                  className={`p-2 rounded-lg bg-${item.color}-500/10 border border-${item.color}-500/20 shrink-0`}
                                >
                                  <item.icon className={`h-5 w-5 text-${item.color}-400`} />
                                </motion.div>
                                <span className="text-body-sm text-surface-300 mt-0.5 leading-relaxed">{item.text}</span>
                              </motion.li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    </Link>
                  </motion.article>
                </div>
              </div>
            </div>
          </div>
        </section>

      </main>
    </motion.div>
  )
}

/* ============================================
   LIVE CAMERA PAGE
   ============================================ */
function LiveCameraPage() {
  const {
    stream,
    devices,
    selectedDeviceId,
    isLoading,
    error: cameraError,
    permissionState,
    startCamera,
    switchCamera,
    requestPermission,
    setError,
  } = useCamera()
  
  const {
    model,
    isModelLoading,
    modelError,
    stats,
    runInference,
    threshold,
    setThreshold,
  } = useLiveInference()
  
  const videoRef = useRef<HTMLVideoElement>(null)
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<{ current: number | null }>({ current: null })
  const [isCapturing, setIsCapturing] = useState(false)
  const [captureError, setCaptureError] = useState<string | null>(null)
  const [report, setReport] = useState<AnalysisReport | null>(null)
  const [showReport, setShowReport] = useState(false)
  const [permissionDenied, setPermissionDenied] = useState(false)
  const [showPermissionRequest, setShowPermissionRequest] = useState(true)
  const [selectedCameraId, setSelectedCameraId] = useState<string | null>(null)
  const navigate = useNavigate()

  // Handle camera permission denial
  useEffect(() => {
    if (cameraError && cameraError.includes('permission denied')) {
      setPermissionDenied(true)
      setShowPermissionRequest(false)
    }
  }, [cameraError])

  // Auto-request permission on mount
  useEffect(() => {
    if (permissionState === 'not-requested') {
      setShowPermissionRequest(true)
    } else if (permissionState === 'granted' && !stream) {
      setShowPermissionRequest(false)
    }
  }, [permissionState, stream])

  const handlePermissionGranted = async () => {
    await requestPermission()
    if (permissionState === 'granted') {
      setShowPermissionRequest(false)
    }
  }

  const handleCameraSelect = async (deviceId: string) => {
    setSelectedCameraId(deviceId)
    await startCamera(deviceId)
    setShowPermissionRequest(false)
  }

  const handleStartDefaultCamera = async () => {
    await startCamera(selectedDeviceId || undefined)
    setShowPermissionRequest(false)
  }

  // Attach stream to video element and play
  useEffect(() => {
    if (stream && videoRef.current) {
      videoRef.current.srcObject = stream
      videoRef.current.play().catch(err => {
        console.error('Video play failed:', err)
        setError('Failed to start video playback')
      })
    }
  }, [stream])

  // Animation loop for live inference
  useEffect(() => {
    if (!stream || !model || !videoRef.current || !overlayCanvasRef.current) return
    
    const animate = async () => {
      if (videoRef.current?.readyState === 4) {
        await runInference(videoRef.current, overlayCanvasRef.current!)
      }
      animationRef.current.current = requestAnimationFrame(animate)
    }
    
    animate()
    return () => {
      if (animationRef.current.current) {
        cancelAnimationFrame(animationRef.current.current)
      }
    }
  }, [stream, model, runInference])

  // Update overlay canvas size to match video
  useEffect(() => {
    if (!videoRef.current || !overlayCanvasRef.current) return
    
    const resizeCanvas = () => {
      if (videoRef.current && overlayCanvasRef.current) {
        overlayCanvasRef.current.width = videoRef.current.videoWidth
        overlayCanvasRef.current.height = videoRef.current.videoHeight
      }
    }
    
    resizeCanvas()
    const observer = new ResizeObserver(resizeCanvas)
    observer.observe(videoRef.current)
    return () => observer.disconnect()
  }, [stream])

  const handleCapture = async () => {
    if (!videoRef.current) return
    
    // Check if video is ready and has valid dimensions
    if (videoRef.current.readyState < 2 || videoRef.current.videoWidth === 0 || videoRef.current.videoHeight === 0) {
      setCaptureError('Video not ready. Please wait for camera to start.')
      return
    }
    
    setIsCapturing(true)
    setCaptureError(null)
    
    try {
      const canvas = document.createElement('canvas')
      canvas.width = videoRef.current.videoWidth
      canvas.height = videoRef.current.videoHeight
      const ctx = canvas.getContext('2d')!
      ctx.drawImage(videoRef.current, 0, 0)
      
      const blob = await new Promise<Blob | null>((resolve) => {
        canvas.toBlob(resolve, 'image/png')
      })
      
      if (!blob) throw new Error('Failed to create image blob')
      
      const file = new File([blob], 'crack-capture.png', { type: 'image/png' })
      const analysisReport = await analyzeImage(file)
      setReport(analysisReport)
      setShowReport(true)
    } catch (err) {
      setCaptureError(err instanceof Error ? err.message : 'Capture failed')
    } finally {
      setIsCapturing(false)
    }
  }

  const handleRetryCamera = () => {
    setPermissionDenied(false)
    setShowPermissionRequest(true)
  }

  const showDeviceSelector = devices.length > 1

  // Permission Request Screen
  if (showPermissionRequest) {
    return (
      <div className="min-h-screen relative overflow-hidden safe-area">
        <div className="fixed inset-0 hero-pattern" />
        <div className="fixed inset-0 grid-pattern" />
        
        <div className="relative z-10 min-h-screen flex flex-col">
          <header className="relative z-10 bg-surface-950/75 backdrop-blur-xl shadow-[0_10px_30px_rgba(0,0,0,0.18)] sticky top-0 safe-top">
            <div className="container">
              <div className="flex h-16 items-center justify-between gap-4">
                <Link to="/" className="flex items-center gap-2 text-surface-400 hover:text-surface-100 transition-colors flex-shrink-0">
                  <Target className="h-6 w-6 text-primary-400" />
                  <span className="text-xl font-bold text-surface-50 hidden sm:inline">Crack Detection on Structures</span>
                </Link>
                <Link to="/" className="text-sm text-surface-400 hover:text-surface-100 transition-colors px-3 py-1.5 rounded-lg hover:bg-surface-800/50">
                  Home
                </Link>
              </div>
            </div>
          </header>
          
          <div className="relative z-10 flex-1 flex items-center justify-center p-4">
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="max-w-2xl w-full"
            >
              <motion.div 
                className="card text-center mb-8"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <motion.div 
                  className="inline-flex p-3 rounded-2xl bg-primary-500/10 border border-primary-500/20 mb-6"
                  whileHover={{ scale: 1.05, rotate: 3 }}
                  transition={{ type: "spring", stiffness: 400, damping: 17 }}
                >
                  <Camera className="h-10 w-10 text-primary-400" />
                </motion.div>
                <h1 className="text-heading-xl font-bold text-surface-50 mb-2">Camera Access Required</h1>
                <p className="text-body text-surface-400">
                  Crack Detection on Structures needs camera access to perform live crack detection. 
                  Please grant permission to continue.
                </p>
              </motion.div>
              
              {/* Camera Selection */}
              {devices.length > 0 && (
                <motion.div 
                  className="card mb-8"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                >
                  <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
                    <h3 className="text-heading-sm font-semibold text-surface-50">Select Camera</h3>
                    <Badge className="text-xs" variant="secondary">
                      {devices.length} detected
                    </Badge>
                  </div>
                  <div className="space-y-2 max-h-60 overflow-y-auto">
                    {devices.map((device) => (
                      <Button
                        key={device.deviceId}
                        variant={selectedCameraId === device.deviceId ? 'default' : 'secondary'}
                        className="w-full justify-start gap-3 relative overflow-hidden"
                        onClick={() => handleCameraSelect(device.deviceId)}
                        disabled={isLoading}
                      >
                        <div className="flex items-center gap-3 flex-1 relative z-10 flex-wrap">
                          {device.isPhoneCamera && (
                            <Badge className="shrink-0" variant="primary">
                              <Smartphone className="h-3 w-3 mr-1" />
                              Phone/External
                            </Badge>
                          )}
                          {device.label.toLowerCase().includes('back') || device.label.toLowerCase().includes('rear') ? (
                            <Badge className="shrink-0" variant="secondary">
                              <RotateCcw className="h-3 w-3 mr-1" />
                              Rear
                            </Badge>
                          ) : device.label.toLowerCase().includes('front') ? (
                            <Badge className="shrink-0" variant="secondary">
                              <RotateCcw className="h-3 w-3 mr-1" />
                              Front
                            </Badge>
                          ) : (
                            <Badge className="shrink-0" variant="neutral">
                              <Monitor className="h-3 w-3 mr-1" />
                              Built-in
                            </Badge>
                          )}
                          <span className="text-left flex-1 text-body text-surface-300 truncate min-w-0">{device.label}</span>
                        </div>
                        {selectedCameraId === device.deviceId && (
                          <CheckCircle className="h-5 w-5 text-primary-400 shrink-0" />
                        )}
                      </Button>
                    ))}
                  </div>
                </motion.div>
              )}
              
              <motion.div 
                className="flex flex-col gap-3"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
              >
                {devices.length > 0 ? (
                  <>
                    <Button 
                      onClick={handleStartDefaultCamera} 
                      className="w-full group" 
                      size="lg"
                      disabled={isLoading}
                    >
                      {isLoading ? (
                        <>
                          <Loader2 className="h-5 w-5 animate-spin mr-2" />
                          Starting Camera...
                        </>
                      ) : (
                        <>
                          <Camera className="h-5 w-5 mr-2" />
                          {selectedCameraId ? 'Use Selected Camera' : 'Use Default Camera'}
                        </>
                      )}
                    </Button>
                  </>
                ) : (
                  <Button 
                    onClick={handlePermissionGranted} 
                    className="w-full group" 
                    size="lg"
                    disabled={isLoading}
                  >
                    {isLoading ? (
                      <>
                        <Loader2 className="h-5 w-5 animate-spin mr-2" />
                        Requesting Permission...
                      </>
                    ) : (
                      <>
                        <Camera className="h-5 w-5 mr-2" />
                        Grant Camera Permission
                      </>
                    )}
                  </Button>
                )}
                <Button variant="outline" onClick={() => navigate('/analyze')} className="w-full" size="lg">
                  <Upload className="h-5 w-5 mr-2" />
                  Upload Image Instead
                </Button>
              </motion.div>
              
              {cameraError && (
                <motion.p
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-4 text-sm text-danger-400 text-center"
                >
                  {cameraError}
                </motion.p>
              )}
            </motion.div>
          </div>
        </div>
      </div>
    )
  }

  if (permissionDenied) {
    return (
      <div className="min-h-screen relative overflow-hidden flex items-center justify-center p-4 safe-area">
        <div className="fixed inset-0 hero-pattern" />
        <div className="fixed inset-0 grid-pattern" />
        
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="relative z-10 max-w-md w-full text-center"
        >
          <motion.div 
            className="card"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <motion.div 
              className="inline-flex p-3 rounded-2xl bg-danger-500/10 border border-danger-500/20 mb-6"
              whileHover={{ scale: 1.05, rotate: -3 }}
            >
              <AlertTriangle className="h-10 w-10 text-danger-400" />
            </motion.div>
            <h1 className="text-heading-xl font-bold text-surface-50 mb-2">Camera Access Denied</h1>
            <p className="text-body text-surface-400 mb-6">
              Camera permission is required for live scanning. Please enable camera access in your browser settings and try again.
            </p>
            <div className="flex flex-col gap-3">
              <Button onClick={handleRetryCamera} className="w-full" size="lg">
                <RotateCcw className="h-4 w-4 mr-2" />
                Retry Camera Access
              </Button>
              <Button variant="outline" onClick={() => navigate('/analyze')} className="w-full">
                <Upload className="h-4 w-4 mr-2" />
                Upload Image Instead
              </Button>
            </div>
          </motion.div>
        </motion.div>
      </div>
    )
  }

  return (
    <div className="h-dvh relative overflow-hidden bg-surface-950 flex flex-col safe-area">
      {/* Animated background */}
      <div className="fixed inset-0 hero-pattern opacity-50" />
      <div className="fixed inset-0 grid-pattern opacity-30" />
      
      {/* Header */}
      <header className="relative z-10 bg-surface-950/75 backdrop-blur-xl shadow-[0_10px_30px_rgba(0,0,0,0.18)] sticky top-0 safe-top">
        <div className="container">
          <div className="flex h-16 items-center gap-4">
            <Link to="/" className="flex items-center gap-2 text-surface-400 hover:text-surface-100 transition-colors flex-shrink-0">
              <Target className="h-6 w-6 text-primary-400" />
              <span className="text-xl font-bold text-surface-50 hidden sm:inline">Crack Detection on Structures</span>
            </Link>
            <div className="flex items-center gap-2 ml-auto flex-wrap">
              <Link to="/" className="text-sm text-surface-400 hover:text-surface-100 transition-colors px-3 py-1.5 rounded-lg hover:bg-surface-800/50 hidden sm:inline-flex">
                Home
              </Link>
              <Link to="/analyze" className="text-sm text-surface-400 hover:text-surface-100 transition-colors px-3 py-1.5 rounded-lg hover:bg-surface-800/50 hidden sm:inline-flex">
                <Camera className="h-4 w-4 mr-1" />
                Upload & Analyze
              </Link>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative z-10 flex-1 min-h-0 flex flex-col lg:flex-row overflow-hidden">
        {/* Camera View */}
        <section className="flex-1 min-h-0 relative bg-surface-950 lg:rounded-none">
          <div className="relative w-full h-full min-h-0">
            {/* Video Feed */}
            <div className="video-container absolute inset-0">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-contain bg-surface-950"
                style={{ transform: 'scaleX(-1)' }}
              />
              
              {/* Overlay Canvas for Mask */}
              <canvas
                ref={overlayCanvasRef}
                className="canvas-overlay object-contain"
                style={{
                  transform: 'scaleX(-1)',
                  opacity: 0.5,
                }}
              />
            </div>
            
            {/* Loading Overlay */}
            {(isLoading || isModelLoading) && (
              <div className="absolute inset-0 flex items-center justify-center bg-surface-950/95 z-10 p-4">
                <motion.div 
                  className="text-center max-w-sm"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                >
                  <div className="spinner-lg mx-auto mb-4" />
                  <p className="text-body text-surface-400">
                    {isLoading ? 'Starting camera...' : 'Loading AI model...'}
                  </p>
                </motion.div>
              </div>
            )}
            
            {/* Camera Error */}
            {cameraError && !permissionDenied && !showPermissionRequest && (
              <div className="absolute inset-0 flex items-center justify-center bg-surface-950/95 z-10 p-4">
                <motion.div
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="max-w-sm text-center"
                >
                  <div className="p-4 rounded-2xl bg-danger-500/10 border border-danger-500/20 w-fit mx-auto mb-4">
                    <AlertTriangle className="h-10 w-10 text-danger-400" />
                  </div>
                  <h2 className="text-heading-md font-semibold mb-2">Camera Error</h2>
                  <p className="text-body text-surface-400 mb-4">{cameraError}</p>
                  <Button onClick={() => startCamera()} size="lg" className="w-full sm:w-auto">
                    <RotateCcw className="h-4 w-4 mr-2" />
                    Retry
                  </Button>
                </motion.div>
              </div>
            )}
            
            {/* HUD - Responsive positioning */}
            {stream && model && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="absolute bottom-4 left-4 right-4 lg:left-4 lg:right-auto z-20 lg:w-72"
                style={{ 
                  width: 'clamp(200px, 100%, 288px)' 
                }}
              >
                <div className="hud-panel">
                  <div className="flex items-center gap-2 mb-4">
                    <div className="p-2 rounded-lg bg-primary-500/10">
                      <Activity className="h-5 w-5 text-primary-400" />
                    </div>
                    <span className="text-heading-sm font-semibold text-surface-50">Live HUD</span>
                  </div>
                  
                  <div className="grid grid-cols-3 gap-3 mb-4">
                    <div className="hud-metric">
                      <div className="hud-metric-value">{stats.fps}</div>
                      <div className="hud-metric-label">FPS</div>
                    </div>
                    <div className="hud-metric">
                      <div className="hud-metric-value">{stats.latency.toFixed(1)}</div>
                      <div className="hud-metric-label">ms</div>
                    </div>
                    <div className="hud-metric">
                      <div className="hud-metric-value">{stats.coverage.toFixed(2)}%</div>
                      <div className="hud-metric-label">Coverage</div>
                    </div>
                  </div>
                  
                  <Separator className="mb-4" />
                  
                  <div className="flex items-center justify-between text-sm mb-4 flex-wrap gap-2">
                    <span className="text-surface-400">Threshold</span>
                    <div className="flex items-center gap-2 flex-1 min-w-[150px]">
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.05"
                        value={threshold}
                        onChange={(e) => setThreshold(parseFloat(e.target.value))}
                        className="flex-1 accent-primary h-2 rounded-lg appearance-none cursor-pointer min-w-0"
                      />
                      <span className="font-mono text-primary-400 w-10 text-right shrink-0">{threshold.toFixed(2)}</span>
                    </div>
                  </div>
                  
                  {modelError && (
                    <Badge variant="accent" className="w-full justify-center text-wrap">
                      <WifiOff className="h-3 w-3 mr-1 shrink-0" />
                      Browser model unavailable — server fallback
                    </Badge>
                  )}
                </div>
              </motion.div>
            )}
            
            {/* Controls Overlay - Responsive */}
            {stream && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="absolute bottom-4 left-1/2 -translate-x-1/2 flex flex-col items-center gap-4 z-20 lg:bottom-8 px-4"
              >
                <div className="flex items-center gap-3 flex-wrap justify-center">
                  {showDeviceSelector && (
                    <Button 
                      variant="ghost" 
                      size="icon" 
                      onClick={switchCamera} 
                      aria-label="Switch camera"
                      className="h-12 w-12 touch-target"
                    >
                      <RotateCcw className="h-6 w-6" />
                    </Button>
                  )}
                  <Button
                    size="lg"
                    onClick={handleCapture}
                    disabled={isCapturing}
                    className="group gap-3 px-6 py-4 text-base rounded-xl w-full sm:w-auto min-h-[52px] touch-target"
                    style={{ 
                      background: 'linear-gradient(135deg, #14b8a6, #0d9488)',
                      boxShadow: '0 8px 32px rgba(20, 184, 166, 0.4)'
                    }}
                  >
                    {isCapturing ? (
                      <>
                        <Loader2 className="h-5 w-5 animate-spin" />
                        <span>Capturing...</span>
                      </>
                    ) : (
                      <>
                        <Camera className="h-5 w-5" />
                        <span>Capture & Analyze</span>
                      </>
                    )}
                  </Button>
                  {captureError && (
                    <Badge variant="destructive" className="w-full sm:max-w-xs text-wrap">
                      <AlertTriangle className="h-3 w-3 mr-1 shrink-0" />
                      {captureError}
                    </Badge>
                  )}
                </div>
              </motion.div>
            )}
          </div>
        </section>
        
        {/* Sidebar / Report Panel - Responsive: show on tablet and up */}
        <aside className="w-full lg:w-[36rem] xl:w-[42rem] min-h-0 border-t border-surface-800 lg:border-l lg:border-t-0 bg-surface-950/50 flex flex-col hidden md:flex lg:flex">
          <div className="p-4 border-b border-surface-800 flex items-center justify-between">
            <h2 className="text-heading-sm font-semibold flex items-center gap-2 text-surface-50 truncate">
              <Gauge className="h-5 w-5 text-primary-400 shrink-0" />
              Analysis Report
            </h2>
            <Button variant="ghost" size="icon" onClick={() => setShowReport(false)} className="touch-target">
              <XIcon className="h-4 w-4" />
            </Button>
          </div>
          
          <div className="flex-1 overflow-y-auto p-4">
            <AnimatePresence mode="wait">
              {showReport && report ? (
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="space-y-4"
                >
                  <ReportPanel report={report} onClose={() => setShowReport(false)} />
                </motion.div>
              ) : (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="h-full flex flex-col items-center justify-center text-surface-500"
                >
                  <div className="p-4 rounded-2xl bg-primary-500/10 border border-primary-500/20 mb-4">
                    <Camera className="h-12 w-12 text-primary-400" />
                  </div>
                  <p className="text-body text-center px-4">Capture a frame to see the detailed analysis report here.</p>
                  <p className="text-caption text-surface-600 mt-2 text-center">The server runs full U-Net inference with contour extraction.</p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </aside>
      </main>
    </div>
  )
}

/* ============================================
   REPORT PANEL
   ============================================ */
function ReportPanel({ report, onClose }: { report: AnalysisReport; onClose?: () => void }) {
  return (
    <motion.div 
      className="space-y-4"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h3 className="font-semibold text-surface-50">Server Analysis Complete</h3>
          <p className="text-caption text-surface-500">Full U-Net + contour extraction</p>
        </div>
        {onClose && (
          <Button variant="ghost" size="icon" onClick={onClose} className="touch-target">
            <XIcon className="h-4 w-4" />
          </Button>
        )}
      </div>
      
      <Badge 
        variant={report.crack_detected ? 'destructive' : 'success'} 
        className="w-full justify-center px-4 py-2 gap-2 text-wrap"
      >
        {report.crack_detected ? (
          <>
            <AlertTriangle className="h-5 w-5 shrink-0" />
            CRACK DETECTED
          </>
        ) : (
          <>
            <CheckCircle className="h-5 w-5 shrink-0" />
            NO CRACKS FOUND
          </>
        )}
      </Badge>
      
      <div className="grid grid-cols-2 gap-3 md:gap-4">
        <StatCard label="Crack Area" value={`${report.crack_area_percent.toFixed(2)}%`} icon={<Target className="h-4 w-4" />} />
        <StatCard label="Regions" value={report.crack_regions.toString()} icon={<Layers className="h-4 w-4" />} />
        <StatCard label="Inference Time" value={`${report.inference_time_ms.toFixed(1)} ms`} icon={<Activity className="h-4 w-4" />} />
        <StatCard label="Status" value={report.crack_detected ? 'Positive' : 'Negative'} icon={<Info className="h-4 w-4" />} />
      </div>
      
      <div className="relative aspect-square rounded-2xl overflow-hidden border border-surface-700">
        <img
          src={report.annotated_image}
          alt="Annotated crack analysis"
          className="w-full h-full object-contain"
        />
      </div>
      <p className="text-caption text-surface-500 mt-2 text-center">
        Green contours indicate detected crack regions
      </p>
      
      <Button 
        variant="outline" 
        className="w-full gap-2"
        onClick={() => {
          const link = document.createElement('a')
          link.href = report.annotated_image
          link.download = 'crack-analysis.png'
          link.click()
        }}
      >
        <Download className="h-4 w-4" />
        Download Annotated PNG
      </Button>
    </motion.div>
  )
}

function StatCard({ label, value, icon }: { label: string; value: string; icon: React.ReactNode }) {
  return (
    <Card className="stat-card">
      <div className="flex items-center justify-between mb-2">
        <div className="p-2 rounded-lg bg-primary-500/10 text-primary-400">
          {icon}
        </div>
        <div className="text-2xl font-mono font-bold text-primary-400">{value}</div>
      </div>
      <div className="text-caption text-surface-500 uppercase tracking-wide">{label}</div>
    </Card>
  )
}

/* ============================================
   ANALYZE PAGE
   ============================================ */
function AnalyzePage() {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [report, setReport] = useState<AnalysisReport | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [dragActive, setDragActive] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0]
      if (droppedFile.type.startsWith('image/')) {
        setFile(droppedFile)
        setPreviewUrl(URL.createObjectURL(droppedFile))
        setError(null)
        setReport(null)
      } else {
        setError('Please drop an image file (PNG, JPEG, WebP)')
      }
    }
  }, [])

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0]
      if (selectedFile.type.startsWith('image/')) {
        setFile(selectedFile)
        setPreviewUrl(URL.createObjectURL(selectedFile))
        setError(null)
        setReport(null)
      } else {
        setError('Please select an image file (PNG, JPEG, WebP)')
      }
    }
  }

  const handleAnalyze = async () => {
    if (!file) return
    
    setIsAnalyzing(true)
    setError(null)
    
    try {
      const analysisReport = await analyzeImage(file)
      setReport(analysisReport)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleRemove = () => {
    setFile(null)
    setPreviewUrl(null)
    setReport(null)
    setError(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const handleDownload = () => {
    if (!report) return
    const link = document.createElement('a')
    link.href = report.annotated_image
    link.download = 'crack-analysis.png'
    link.click()
  }

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl)
      }
    }
  }, [previewUrl])

  return (
    <div className="h-dvh relative overflow-hidden bg-surface-950 flex flex-col safe-area">
      <div className="fixed inset-0 hero-pattern opacity-50" />
      <div className="fixed inset-0 grid-pattern opacity-30" />
      
      {/* Header */}
      <header className="relative z-10 bg-surface-950/75 backdrop-blur-xl shadow-[0_10px_30px_rgba(0,0,0,0.18)] sticky top-0 safe-top">
        <div className="container">
          <div className="flex h-16 items-center justify-between gap-4">
            <Link to="/" className="flex items-center gap-2 text-surface-400 hover:text-surface-100 transition-colors flex-shrink-0">
              <Target className="h-6 w-6 text-primary-400" />
              <span className="text-xl font-bold text-surface-50 hidden sm:inline">Crack Detection on Structures</span>
            </Link>
            <div className="flex items-center gap-2 flex-wrap">
              <Link to="/" className="text-sm text-surface-400 hover:text-surface-100 transition-colors px-3 py-1.5 rounded-lg hover:bg-surface-800/50 hidden sm:inline-flex">
                Home
              </Link>
              <Link to="/live" className="text-sm text-surface-400 hover:text-surface-100 transition-colors px-3 py-1.5 rounded-lg hover:bg-surface-800/50 hidden sm:inline-flex">
                <Camera className="h-4 w-4 mr-1" />
                Live Scan
              </Link>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="panel-scroll relative z-10 flex-1 min-h-0 max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-4 lg:py-6 overflow-y-auto">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="grid gap-5 md:gap-7 lg:gap-8 lg:grid-cols-2 min-h-full"
        >
          {/* Upload Panel */}
          <section className="min-w-0 flex flex-col">
            <motion.div 
              className={`card drop-zone ${dragActive ? 'active' : ''} w-full`}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <CardHeader className="pb-3">
                <CardTitle className="text-heading-md">Image Input</CardTitle>
                <CardDescription>
                  Drop an image or choose a file for detailed server-side crack analysis
                </CardDescription>
              </CardHeader>
              <CardContent className="pt-0 space-y-4">
                <div className="relative flex items-center justify-center">
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleFileSelect}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                    disabled={isAnalyzing}
                  />
                  
                  {file && previewUrl ? (
                    <motion.div 
                      className="relative w-full"
                      initial={{ opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                    >
                      <div className="relative aspect-square w-full rounded-xl overflow-hidden border border-surface-700/80 bg-surface-900">
                        <img
                          src={previewUrl}
                          alt="Preview"
                          className="w-full h-full object-contain"
                        />
                      </div>
                      <div className="flex flex-col sm:flex-row items-center justify-center gap-3 mt-4">
                        <Button variant="secondary" size="sm" onClick={() => fileInputRef.current?.click()} className="w-full sm:w-auto">
                          <RotateCcw className="h-4 w-4 mr-1" />
                          Change
                        </Button>
                        <Button variant="danger" size="sm" onClick={handleRemove} className="w-full sm:w-auto">
                          <Trash2 className="h-4 w-4 mr-1" />
                          Remove
                        </Button>
                      </div>
                    </motion.div>
                  ) : (
                    <motion.div 
                      className="py-8 px-4 text-center space-y-3"
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                    >
                      <motion.div
                        whileHover={{ scale: 1.05 }}
                        className="p-3 rounded-2xl bg-primary-500/10 border border-primary-500/20 w-fit mx-auto"
                      >
                        <Upload className="h-9 w-9 text-primary-400" />
                      </motion.div>
                      <p className="text-heading-sm font-medium">Drag & drop an image here</p>
                      <p className="text-body-sm text-surface-500">or click to browse</p>
                      <Button variant="outline" onClick={() => fileInputRef.current?.click()} className="w-full sm:w-auto">
                        <Image className="h-4 w-4 mr-2" />
                        Choose File
                      </Button>
                    </motion.div>
                  )}
                </div>

                {error && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="p-4 rounded-xl bg-danger-500/10 border border-danger-500/20 flex items-start gap-3"
                  >
                    <AlertTriangle className="h-5 w-5 text-danger-400 shrink-0 mt-0.5" />
                    <p className="text-danger-300 text-sm">{error}</p>
                  </motion.div>
                )}
              </CardContent>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="mt-5 w-full"
            >
              <Button
                size="lg"
                onClick={handleAnalyze}
                disabled={!file || isAnalyzing}
                className="w-full gap-2"
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 className="h-5 w-5 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Activity className="h-5 w-5" />
                    Run Analysis
                  </>
                )}
              </Button>
            </motion.div>
          </section>

          {/* Results Panel */}
          <section className="min-w-0 flex flex-col">
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="flex-1 w-full"
            >
              <AnimatePresence mode="wait">
                    {report ? (
                      <motion.div
                        key="report"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        className="space-y-6 w-full"
                      >
                        <AnalyzeReportView report={report} onDownload={handleDownload} />
                      </motion.div>
                    ) : (
                      <motion.div
                        key="empty"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="flex-1 flex items-center justify-center"
                      >
                        <motion.div 
                          className="card text-center w-full"
                          whileHover={{ y: -4 }}
                        >
                          <CardContent className="flex flex-col items-center justify-center py-10 px-6 space-y-3">
                            <motion.div
                              whileHover={{ scale: 1.05, rotate: 3 }}
                              className="p-4 rounded-2xl bg-primary-500/10 border border-primary-500/20"
                            >
                              <Image className="h-12 w-12 text-surface-500" />
                            </motion.div>
                            <h3 className="text-heading-sm font-semibold">No Analysis Yet</h3>
                            <p className="text-body-sm text-surface-500 text-center max-w-sm">
                              Upload an image and run analysis to see the comparison slider,
                              statistics, and annotated results here.
                            </p>
                          </CardContent>
                        </motion.div>
                      </motion.div>
                    )}
                  </AnimatePresence>
            </motion.div>
          </section>
        </motion.div>
      </main>
    </div>
  )
}

function AnalyzeReportView({ report, onDownload }: { report: AnalysisReport; onDownload: () => void }) {
  return (
    <motion.div 
      className="flex flex-col space-y-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      {/* Status Badge */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <Badge 
          variant={report.crack_detected ? 'destructive' : 'success'} 
          className="px-4 py-2 gap-2 text-wrap"
        >
          {report.crack_detected ? (
            <>
              <AlertTriangle className="h-4 w-4 shrink-0" />
              CRACK DETECTED
            </>
          ) : (
            <>
              <CheckCircle className="h-4 w-4 shrink-0" />
              NO CRACKS FOUND
            </>
          )}
        </Badge>
      </div>

      {/* Annotated Result */}
      <motion.div 
        className="card flex flex-col"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <CardHeader className="flex flex-row items-center justify-between flex-wrap gap-2">
          <CardTitle className="text-heading-md">Annotated Result</CardTitle>
          <Button variant="outline" size="sm" onClick={onDownload} className="gap-1">
            <Download className="h-3 w-3" />
            Download PNG
          </Button>
        </CardHeader>
        <CardContent className="pt-0 flex-1 flex items-center justify-center">
          <div className="relative aspect-square w-full rounded-xl overflow-hidden border border-surface-700 bg-surface-900">
            <img
              src={report.annotated_image}
              alt="Annotated crack analysis with contours"
              className="w-full h-full object-contain"
            />
          </div>
          <p className="text-caption text-surface-500 mt-2 text-center">
            Green contours indicate detected crack regions
          </p>
        </CardContent>
      </motion.div>

      {/* Statistics */}
      <motion.div 
        className="card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <CardHeader>
          <CardTitle className="text-heading-md">Analysis Statistics</CardTitle>
        </CardHeader>
        <CardContent className="pt-0">
          <div className="grid grid-cols-2 gap-3 md:gap-4">
            <StatItem 
              label="Crack Detected" 
              value={report.crack_detected ? 'Yes' : 'No'} 
              variant={report.crack_detected ? 'destructive' : 'success'}
            />
            <StatItem label="Crack Area" value={`${report.crack_area_percent.toFixed(2)}%`} />
            <StatItem label="Distinct Regions" value={report.crack_regions.toString()} />
            <StatItem label="Inference Time" value={`${report.inference_time_ms.toFixed(1)} ms`} />
          </div>
        </CardContent>
      </motion.div>
    </motion.div>
  )
}

function StatItem({ label, value, variant = 'default' }: { label: string; value: string; variant?: 'default' | 'destructive' | 'success' }) {
  const variants = {
    default: 'bg-surface-800/50 border-surface-700',
    destructive: 'bg-danger-500/10 border-danger-500/20',
    success: 'bg-success-500/10 border-success-500/20',
  }
  
  return (
    <div className={cn('min-w-0 rounded-xl border', variants[variant])} style={{ padding: 'clamp(0.75rem, 0.6rem + 0.75vw, 1rem)' }}>
      <div className="min-w-0 font-mono font-bold text-primary-400 mb-1 break-words" style={{ fontSize: 'clamp(1.125rem, 1rem + 0.75vw, 1.5rem)' }}>{value}</div>
      <div className="text-caption text-surface-500 uppercase tracking-wide leading-tight">{label}</div>
    </div>
  )
}

/* ============================================
   APP ROUTER
   ============================================ */
function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/live" element={<LiveCameraPage />} />
      <Route path="/analyze" element={<AnalyzePage />} />
    </Routes>
  )
}

export default App