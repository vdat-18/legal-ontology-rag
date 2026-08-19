/**
 * @type {import('next').NextConfig}
 *
 * `next dev` and `next build` both write to the same output directory, so a production build run
 * while a dev server is up replaces the chunks that server has already loaded and every request
 * then fails with "Cannot find module './331.js'". Setting NEXT_DIST_DIR for the production
 * commands keeps the two from sharing a directory.
 */
const nextConfig = {
  reactStrictMode: true,
  distDir: process.env.NEXT_DIST_DIR ?? ".next",
};

export default nextConfig;
