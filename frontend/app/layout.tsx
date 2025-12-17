import "./globals.css";
import React from "react";

export const metadata = {
  title: "SoundFlow AI — focus music that flows.",
  description:
    "AI-powered focus music platform for deep work, study, and relaxation. Personalized sound sessions with ambient mixing."
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <base target="_self" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"
          rel="stylesheet"
        />
        <link
          rel="stylesheet"
          href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css"
        />
      </head>
      <body className="min-h-screen bg-zinc-950 text-white font-sans">{children}</body>
    </html>
  );
}
