import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "📈 Market Regime Intelligence Dashboard | V1",
  description: "Macroeconomic market regime detection, transition forecasting, DTW trajectory similarity search, and AI market analyst engine.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark h-full antialiased">
      <body className="min-h-full flex flex-col bg-slate-950 text-slate-100 selection:bg-cyan-500 selection:text-slate-950 font-sans">
        {children}
      </body>
    </html>
  );
}
