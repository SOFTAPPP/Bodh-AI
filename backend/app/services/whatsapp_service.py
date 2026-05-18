import os
import json
import httpx
import asyncio
from typing import Dict, Optional, List
from app.core.config import Config

class WhatsAppService:
    def __init__(self):
        self.verify_token = Config.META_VERIFY_TOKEN
        self.api_token = Config.META_API_TOKEN
        self.phone_id = Config.WHATSAPP_PHONE_ID
        self.base_url = f"https://graph.facebook.com/v19.0/{self.phone_id}"
        
        # User state tracking
        # Maps phone_number -> {"last_file": str, "history": List[dict]}
        self.user_state_file = os.path.join(Config.DATA_DIR, "whatsapp_state.json")
        self._user_state: Dict[str, Dict] = self._load_state()

    def _load_state(self) -> Dict[str, Dict]:
        if os.path.exists(self.user_state_file):
            try:
                with open(self.user_state_file, "r") as f:
                    content = f.read().strip()
                    if not content:
                        return {}
                    return json.loads(content)
            except json.JSONDecodeError:
                # Handle corrupted file gracefully
                print("--- [WARNING] whatsapp_state.json corrupted, starting fresh ---")
                return {}
            except Exception as e:
                print(f"--- [ERROR] Failed to load whatsapp state: {e} ---")
                return {}
        return {}

    async def _save_state(self):
        try:
            def save():
                with open(self.user_state_file, "w") as f:
                    json.dump(self._user_state, f, indent=2)
            await asyncio.to_thread(save)
        except Exception as e:
            print(f"--- [ERROR] Failed to save whatsapp state: {e} ---")

    def get_user_file(self, phone_number: str) -> Optional[str]:
        return self._user_state.get(phone_number, {}).get("last_file")

    async def set_user_file(self, phone_number: str, filename: str):
        if phone_number not in self._user_state:
            self._user_state[phone_number] = {"history": []}
        self._user_state[phone_number]["last_file"] = filename
        await self._save_state()

    def get_user_history(self, phone_number: str) -> List[dict]:
        return self._user_state.get(phone_number, {}).get("history", [])

    async def add_user_history(self, phone_number: str, role: str, content: str):
        if phone_number not in self._user_state:
            self._user_state[phone_number] = {"history": [], "last_file": None}
        self._user_state[phone_number]["history"].append({"role": role, "content": content})
        # Keep last 10 messages
        self._user_state[phone_number]["history"] = self._user_state[phone_number]["history"][-10:]
        await self._save_state()

    async def send_message(self, to: str, text: str):
        url = f"{self.base_url}/messages"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {"body": text}
        }
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code not in (200, 201):
                    print(f"--- [ERROR] Meta API Error: {response.text} ---")
                return response.json()
            except Exception as e:
                print(f"--- [ERROR] Failed to send WhatsApp message: {e} ---")
                return None

    async def download_media(self, media_id: str, filename: str) -> Optional[str]:
        url = f"https://graph.facebook.com/v19.0/{media_id}"
        headers = {"Authorization": f"Bearer {self.api_token}"}
        
        async with httpx.AsyncClient() as client:
            try:
                # 1. Get media URL
                response = await client.get(url, headers=headers)
                if response.status_code != 200:
                    print(f"--- [ERROR] Failed to get media URL: {response.text} ---")
                    return None
                
                media_url = response.json().get("url")
                if not media_url:
                    return None
                    
                # 2. Download media content
                media_resp = await client.get(media_url, headers=headers)
                if media_resp.status_code != 200:
                    print(f"--- [ERROR] Failed to download media content: {media_resp.text} ---")
                    return None
                    
                # 3. Save to uploads dir
                filepath = os.path.join(Config.UPLOAD_DIR, filename)
                def save_media():
                    with open(filepath, "wb") as f:
                        f.write(media_resp.content)
                await asyncio.to_thread(save_media)
                    
                return filepath
            except Exception as e:
                print(f"--- [ERROR] Failed to download media: {e} ---")
                return None
