"""
Lab 2 — Step 5: Cached Client

Extends HuggingFaceClient with local caching to minimize API calls.
Essential for development on free tier.

The cache directorynano .env setup and key generation are complete.
Complete the three TODOs in the query() method.
"""

import hashlib
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# استيراد الكلاس من الملف الآخر
from hf_client import HuggingFaceClient, get_api_token

load_dotenv()

class CachedHFClient(HuggingFaceClient):
    def __init__(self, token: str, cache_dir: str = ".cache/hf_responses"):
        super().__init__(token)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, model_id: str, payload: dict) -> str:
        """إنشاء مفتاح فريد بناءً على الموديل والطلب."""
        content = json.dumps({"model": model_id, "payload": payload}, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def query(self, model_id: str, payload: dict, use_cache: bool = True) -> dict:
        """الاستعلام مع نظام التخزين المؤقت."""
        cache_key = self._cache_key(model_id, payload)
        cache_file = self.cache_dir / f"{cache_key}.json"

        # TODO 1: Check cache — التحقق من وجود الملف
        if use_cache and cache_file.exists():
            print("[Cache HIT] Using cached response")
            return json.loads(cache_file.read_text(encoding="utf-8"))

        # TODO 2: Make the API call (Cache MISS)
        print("[Cache MISS] Calling API via OpenRouter...")
        result = super().query(model_id, payload)

        # TODO 3: Write result to cache
        # نتأكد أن النتيجة لا تحتوي على أخطاء قبل حفظها
        if result and "error" not in result:
            json_data = json.dumps(result, ensure_ascii=False, indent=4)
            cache_file.write_text(json_data, encoding="utf-8")
            print(f"[Cache Saved] Response stored locally.")

        return result

# --- تشغيل وتجربة الكود ---
if __name__ == "__main__":
    try:
        # تأكد أن HUGGINGFACE_API_TOKEN يحتوي على مفتاح OpenRouter
        client = CachedHFClient(token=get_api_token())

        # استخدام الموديل التلقائي الذي يختار أفضل موديل مجاني متاح
        active_model = "openrouter/auto" 

        prompt_payload = {
            "inputs": "What is retrieval-augmented generation?",
            "parameters": {
                "max_new_tokens": 50,
                "temperature": 0.3,
            },
        }

        print("--- First call ---")
        result1 = client.query(active_model, prompt_payload)
        if "choices" in result1:
            print(f"Result: {result1['choices'][0]['message']['content'][:100]}...")

        print("\n--- Second call (Should be instant) ---")
        result2 = client.query(active_model, prompt_payload)
        if "choices" in result2:
            print(f"Result (from cache): {result2['choices'][0]['message']['content'][:100]}...")
            
    except Exception as e:
        print(f"Error: {e}")