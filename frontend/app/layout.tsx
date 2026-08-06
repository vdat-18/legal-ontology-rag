import type { Metadata } from "next";
import type { ReactNode } from "react";
import "./globals.css";

export const metadata: Metadata = {
  title: "Hỏi đáp luật giao thông",
  description: "Chatbot luật giao thông Việt Nam trên đồ thị tri thức thời gian",
};

/**
 * Root layout wrapping every page.
 *
 * `suppressHydrationWarning` covers attributes that browser extensions add to `<html>` and
 * `<body>` before React hydrates — Bitdefender writes `bis_register` and a `__processed_*` flag,
 * password managers and translators do the same. The server cannot know about them, so React
 * reports a mismatch the app can neither cause nor fix. It applies to these two elements only,
 * so a genuine mismatch anywhere inside the page is still reported.
 */
export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="vi" suppressHydrationWarning>
      <body suppressHydrationWarning>{children}</body>
    </html>
  );
}
