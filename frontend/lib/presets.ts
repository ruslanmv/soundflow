export const soundscapeData = [
  { id: 1, title: "Deep Work Session", category: "Deep Work", energy: "Stable", duration: "90 min", imageId: 1, color: "teal" },
  { id: 2, title: "Study Focus", category: "Study", energy: "Soft", duration: "50 min", imageId: 2, color: "blue" },
  { id: 3, title: "Evening Relaxation", category: "Relax", energy: "Calm", duration: "60 min", imageId: 3, color: "purple" },
  { id: 4, title: "Forest Ambience", category: "Nature", energy: "Ambient", duration: "120 min", imageId: 4, color: "green" },
  { id: 5, title: "Flow State Coding", category: "Flow State", energy: "Driving", duration: "45 min", imageId: 5, color: "cyan" },
  { id: 6, title: "Morning Focus", category: "Deep Work", energy: "Stable", duration: "75 min", imageId: 6, color: "teal" },
  { id: 7, title: "Exam Preparation", category: "Study", energy: "Soft", duration: "90 min", imageId: 7, color: "blue" },
  { id: 8, title: "Meditation Session", category: "Relax", energy: "Calm", duration: "30 min", imageId: 8, color: "purple" }
] as const;

export const colorClasses: Record<string, string> = {
  teal: "border-teal-400/30 text-teal-400",
  blue: "border-blue-400/30 text-blue-400",
  purple: "border-purple-400/30 text-purple-400",
  green: "border-green-400/30 text-green-400",
  cyan: "border-cyan-400/30 text-cyan-400"
};
