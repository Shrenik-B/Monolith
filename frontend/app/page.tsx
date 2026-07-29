'use client';
import { useEffect, useState } from 'react';

interface BackendData {
  message: string;
  variable_value: number;
  server_time: string;
  status: string;
  environment: string;
}

export default function Home() {
  const [data, setData] = useState<BackendData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  const fetchBackendData = () => {
    setLoading(true);
    setError(null);
    const apiUrl =
      process.env.NEXT_PUBLIC_API_URL ||
      `${window.location.protocol}//${window.location.hostname}:8000`;

    fetch(`${apiUrl}/api/test`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP error ${res.status}`);
        return res.json();
      })
      .then((result: BackendData) => {
        setData(result);
        setLoading(false);
      })
      .catch(() => {
        setError('Failed to connect to backend');
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchBackendData();
  }, []);

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-8 bg-gray-900 text-white">
      <div className="max-w-md w-full bg-gray-800 rounded-xl shadow-lg border border-gray-700 p-6 space-y-6">
        <div className="border-b border-gray-700 pb-4 text-center">
          <h1 className="text-2xl font-bold text-blue-400">Full-Stack Test Connection</h1>
          <p className="text-xs text-gray-400 mt-1">Next.js Frontend &rarr; FastAPI Backend</p>
        </div>

        {loading && (
          <div className="text-center py-6">
            <p className="text-yellow-400 animate-pulse font-medium">Connecting to backend...</p>
          </div>
        )}

        {error && (
          <div className="bg-red-950/60 border border-red-800 text-red-300 p-4 rounded-lg text-center">
            <p className="font-semibold">{error}</p>
            <p className="text-xs mt-1 text-red-400">Ensure uvicorn server is running on port 8000.</p>
            <button
              onClick={fetchBackendData}
              className="mt-3 px-4 py-1.5 bg-red-800 hover:bg-red-700 text-white rounded text-sm transition"
            >
              Retry
            </button>
          </div>
        )}

        {data && (
          <div className="space-y-4">
            <div className="bg-gray-900/80 p-4 rounded-lg border border-gray-700 space-y-2">
              <div className="flex justify-between items-center text-sm">
                <span className="text-gray-400">Backend Status:</span>
                <span className="px-2 py-0.5 rounded text-xs font-semibold bg-green-900/80 text-green-300 border border-green-700">
                  {data.status.toUpperCase()}
                </span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-gray-400">Backend Message:</span>
                <span className="font-medium text-blue-300">{data.message}</span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-gray-400">Test Variable Value:</span>
                <span className="font-mono text-emerald-400 text-lg font-bold">{data.variable_value}</span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-gray-400">Server Time:</span>
                <span className="font-mono text-gray-300">{data.server_time}</span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-gray-400">Target Host:</span>
                <span className="font-mono text-xs text-purple-300">{data.environment}</span>
              </div>
            </div>

            <button
              onClick={fetchBackendData}
              className="w-full py-2 bg-blue-600 hover:bg-blue-500 font-semibold rounded-lg transition text-sm text-white"
            >
              Refresh Variable from Backend
            </button>
          </div>
        )}
      </div>
    </main>
  );
}
