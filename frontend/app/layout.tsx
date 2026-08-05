import type { Metadata } from "next";
import type { ReactNode } from "react";
import "./globals.css";

export const metadata: Metadata = {
  title: "Hỏi đáp luật giao thông",
  description: "Chatbot luật giao thông Việt Nam trên đồ thị tri thức thời gian",
};

/** Root layout wrapping every page. */
export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="vi">
      <body>{children}</body>
    </html>
  );
}
