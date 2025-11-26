// server/redisClient.js
import { createClient } from "redis";

const redis = createClient({
  url: process.env.REDIS_URL || "redis://localhost:6379",
});

redis.on("error", (err) => {
  console.error("Redis Client Error", err);
});

await redis.connect(); // Node 18+라면 top-level await 사용 가능

export default redis;
