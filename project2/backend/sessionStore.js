// server/sessionStore.js
import redis from "./redisClient.js";

const SESSION_PREFIX = "session:";

export async function getSession(sessionId) {
  const raw = await redis.get(SESSION_PREFIX + sessionId);
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch (e) {
    console.error("Failed to parse session JSON", e);
    return null;
  }
}

export async function saveSession(sessionId, sessionData) {
  // sessionData 예: { history: [...], lastPrompt: "...", matchedKeywords: [...], ... }
  const raw = JSON.stringify(sessionData);
  // 1일 TTL 예시
  await redis.set(SESSION_PREFIX + sessionId, raw, {
    EX: 60 * 60 * 24,
  });
}
