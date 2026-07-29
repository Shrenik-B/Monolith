'use client';
import { useEffect, useState } from 'react';

export default function Home() {
  const [status, setStatus] = useState<string>('Connecting...');

  useEffect(() => {
    fetch('http://localhost:8000/health')
      .then((res) => res.json())
      .then((data) => setStatus(data.message))
      .catch(() => setStatus('Failed to connect to backend'));
  }, []);

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-24">
      <h1 className="text-3xl font-bold">Monorepo Health Status</h1>
      <p className="mt-4 text-xl text-green-500">{status}</p>
    </main>
  );
}
