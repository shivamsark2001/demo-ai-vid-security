import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Enable experimental features for better file handling
  experimental: {
    serverActions: {
      bodySizeLimit: '100mb',
    },
  },
  // Allow video URLs from Vercel Blob
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: '*.public.blob.vercel-storage.com',
      },
    ],
  },
};

export default nextConfig;
