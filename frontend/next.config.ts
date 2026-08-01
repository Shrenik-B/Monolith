import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  // Allow dev connections from localhost and your network IP
  allowedDevOrigins: ['localhost', '10.238.51.249'],
};

export default nextConfig;
