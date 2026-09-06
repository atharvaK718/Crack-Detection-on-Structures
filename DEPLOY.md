# Deployment Guide

This guide covers deploying the crack detection web application to production.

## Architecture Overview

- **Frontend**: Next.js 14 (React 18) with TypeScript, Tailwind CSS, Framer Motion
- **Backend**: FastAPI (Python) with TensorFlow/Keras model inference
- **Model**: U-Net for crack segmentation (256×256 input, ~242K params)
- **Live Inference**: TensorFlow.js (client-side) with fallback to server

---

## Local Development

### Prerequisites

- Node.js 20+
- Python 3.11+
- Git

### Backend Setup

```bash
cd Crack-Detection-on-Structures

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Run backend (port 8000)
cd backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

```bash
cd web

# Install dependencies
npm install

# Copy environment file
cp .env.example .env.local

# Run development server (port 3000)
npm run dev
```

### Access

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### HTTPS for Camera Access

`getUserMedia` requires HTTPS or localhost. For local network testing:

```bash
# Option 1: mkcert (recommended)
mkcert -install
mkcert localhost 127.0.0.1 ::1
# Then configure Next.js to use certs

# Option 2: ngrok tunnel
ngrok http 3000
# Use the https ngrok URL
```

---

## TensorFlow.js Model Conversion

For live camera inference in the browser, convert the Keras model:

```bash
cd Crack-Detection-on-Structures

# Activate venv
source .venv/bin/activate

# Install tensorflowjs (may require numpy<2)
pip install "numpy<2" tensorflowjs

# Convert model
python -m tensorflowjs.converter \
  --input_format=keras \
  --quantization_bytes=2 \  # float16 quantization
  model.h5 \
  web/public/tfjs
```

This creates `web/public/tfjs/model.json` and weight shards.

**Note**: The model is ~950KB, so float16 quantization reduces it to ~500KB.

---

## Production Deployment

### Backend: Render (Recommended)

1. **Create a new Web Service on Render**
   - Connect your GitHub repo
   - Root Directory: `Crack-Detection-on-Structures`
   - Build Command: `pip install -r backend/requirements.txt`
   - Start Command: `cd backend && uvicorn app:app --host 0.0.0.0 --port $PORT`
   - Add environment variable: `ALLOWED_ORIGINS=https://your-frontend.vercel.app`

2. **Deploy**
   - Render will install dependencies and start the service
   - Note the service URL (e.g., `https://crack-api.onrender.com`)

### Backend: Railway (Alternative)

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

Set `ALLOWED_ORIGINS` in Railway dashboard.

### Frontend: Vercel (Recommended)

1. **Import project on Vercel**
   - Connect GitHub repo
   - Root Directory: `web`
   - Framework Preset: Next.js
   - Build Command: `npm run build`
   - Output Directory: `.next`

2. **Environment Variables**
   - `NEXT_PUBLIC_API_URL`: Your backend URL (e.g., `https://crack-api.onrender.com`)

3. **Deploy**
   - Vercel auto-deploys on push to main

### Frontend: Netlify (Alternative)

```bash
# Build
npm run build

# Deploy to Netlify
npx netlify deploy --prod --dir=.next
```

Set `NEXT_PUBLIC_API_URL` in Netlify dashboard.

---

## Docker Deployment (Optional)

### Backend Dockerfile

```dockerfile
# backend/Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Frontend Dockerfile

```dockerfile
# web/Dockerfile
FROM node:20-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production

COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

EXPOSE 3000

CMD ["node", "server.js"]
```

Add `output: 'standalone'` to `next.config.js` for standalone output.

---

## Environment Variables

### Backend (.env)

```env
ALLOWED_ORIGINS=https://your-frontend.vercel.app,http://localhost:3000
```

### Frontend (.env.local)

```env
NEXT_PUBLIC_API_URL=https://your-backend.onrender.com
```

---

## Pre-deployment Checklist

- [ ] Model converted to TF.js format in `web/public/tfjs/`
- [ ] Backend health endpoint returns `{"status": "ok"}`
- [ ] CORS origins configured for production frontend URL
- [ ] Frontend builds without errors (`npm run build`)
- [ ] HTTPS enabled for camera access
- [ ] Environment variables set in both platforms
- [ ] Test live camera scan on mobile (rear camera)
- [ ] Test upload & analyze with sample images
- [ ] Verify download report works

---

## Troubleshooting

### Camera not working on mobile
- Ensure HTTPS (not HTTP)
- Check browser permissions
- Try `facingMode: "environment"` for rear camera

### TF.js model not loading
- Verify `web/public/tfjs/model.json` exists
- Check browser console for CORS errors
- Model loads from same origin, so no CORS needed

### Backend model load fails
- Ensure `model.h5` is in repo root
- Check Python dependencies match requirements.txt
- Verify TensorFlow version compatibility

### CORS errors
- Backend `ALLOWED_ORIGINS` must include frontend URL exactly
- No trailing slashes
- Include both `http://localhost:3000` and production URL

---

## Monitoring & Scaling

- **Backend**: Render/Railway auto-scale based on CPU/memory
- **Frontend**: Vercel/Netlify edge network, no scaling needed
- **Model**: Stateless, horizontal scaling works automatically
- **Logs**: Check platform dashboards for errors

---

## Security Notes

- No authentication built-in (add if needed)
- CORS restricted to known origins
- File upload size limited by FastAPI/uvicorn
- Model runs inference only, no training in production
- Keep dependencies updated for security patches