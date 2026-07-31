// frontend/src/app/page.tsx
'use client';

import { useEffect, useState } from 'react';
import axios from 'axios';

export default function Home() {
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading');
  const [message, setMessage] = useState<string>('');
  const [errorDetails, setErrorDetails] = useState<string>('');

  const checkBackendConnection = async () => {
    setStatus('loading');
    setErrorDetails('');
    try {
      // Connects directly to your FastAPI backend running on port 8000
      const response = await axios.get('http://localhost:8000/api/test');
      setMessage(response.data.message || JSON.stringify(response.data));
      setStatus('success');
    } catch (err: any) {
      setStatus('error');
      setErrorDetails(err.message || 'Failed to reach FastAPI server');
    }
  };

  useEffect(() => {
    checkBackendConnection();
  }, []);

  return (
    <main className="min-h-screen bg-slate-900 text-white flex flex-col items-center justify-center p-6">
      <div className="max-w-md w-full bg-slate-800 rounded-xl shadow-lg border border-slate-700 p-8 text-center">
        <h1 className="text-2xl font-bold mb-2">Full-Stack Test Connection</h1>
        <p className="text-slate-400 text-sm mb-6">Next.js Frontend &rarr; FastAPI Backend</p>

        {status === 'loading' && (
          <div className="p-4 bg-slate-700/50 rounded-lg border border-slate-600 animate-pulse">
            <p className="text-amber-400 font-medium">Connecting to backend...</p>
          </div>
        )}

        {status === 'success' && (
          <div className="p-4 bg-emerald-950/60 rounded-lg border border-emerald-600">
            <p className="text-emerald-400 font-semibold mb-1">✓ Connection Successful!</p>
            <p className="text-slate-300 text-sm font-mono">{message}</p>
          </div>
        )}

        {status === 'error' && (
          <div className="p-4 bg-rose-950/60 rounded-lg border border-rose-600">
            <p className="text-rose-400 font-semibold mb-1">Failed to connect to backend</p>
            <p className="text-slate-300 text-xs font-mono mb-3">{errorDetails}</p>
            <button
              onClick={checkBackendConnection}
              className="px-4 py-2 bg-rose-600 hover:bg-rose-500 text-white text-sm font-medium rounded-md transition-colors"
            >
              Retry
            </button>
          </div>
        )}
      </div>
    </main>
  );
}