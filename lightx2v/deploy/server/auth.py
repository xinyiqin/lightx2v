import os
import time

import aiohttp
import jwt
from fastapi import HTTPException
from loguru import logger

from lightx2v.deploy.common.aliyun import AlibabaCloudClient


class AuthManager:
    def __init__(self):
        # Worker access token
        self.worker_secret_key = os.getenv("WORKER_SECRET_KEY", "worker-secret-key-change-in-production")

        # GitHub OAuth
        self.github_client_id = os.getenv("GITHUB_CLIENT_ID", "")
        self.github_client_secret = os.getenv("GITHUB_CLIENT_SECRET", "")

        # Google OAuth
        self.google_client_id = os.getenv("GOOGLE_CLIENT_ID", "")
        self.google_client_secret = os.getenv("GOOGLE_CLIENT_SECRET", "")
        self.google_redirect_uri = os.getenv("GOOGLE_REDIRECT_URI", "")

        self.jwt_algorithm = os.getenv("JWT_ALGORITHM", "HS256")
        self.jwt_expiration_hours = os.getenv("JWT_EXPIRATION_HOURS", 24)
        self.jwt_secret_key = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")

        # Aliyun SMS
        self.aliyun_client = AlibabaCloudClient()

        logger.info(f"AuthManager: GITHUB_CLIENT_ID: {self.github_client_id}")
        logger.info(f"AuthManager: GITHUB_CLIENT_SECRET: {self.github_client_secret}")
        logger.info(f"AuthManager: GOOGLE_CLIENT_ID: {self.google_client_id}")
        logger.info(f"AuthManager: GOOGLE_CLIENT_SECRET: {self.google_client_secret}")
        logger.info(f"AuthManager: GOOGLE_REDIRECT_URI: {self.google_redirect_uri}")
        logger.info(f"AuthManager: JWT_SECRET_KEY: {self.jwt_secret_key}")
        logger.info(f"AuthManager: WORKER_SECRET_KEY: {self.worker_secret_key}")

    def create_jwt_token(self, data):
        data2 = {
            "user_id": data["user_id"],
            "username": data["username"],
            "email": data["email"],
            "homepage": data["homepage"],
        }
        expire = time.time() + (self.jwt_expiration_hours * 3600)
        data2.update({"exp": expire})
        return jwt.encode(data2, self.jwt_secret_key, algorithm=self.jwt_algorithm)

    async def auth_github(self, code):
        try:
            logger.info(f"GitHub OAuth code: {code}")
            token_url = "https://github.com/login/oauth/access_token"
            token_data = {"client_id": self.github_client_id, "client_secret": self.github_client_secret, "code": code}
            headers = {"Accept": "application/json"}

            proxy = os.getenv("auth_https_proxy", None)
            if proxy:
                logger.info(f"auth_github use proxy: {proxy}")
            async with aiohttp.ClientSession() as session:
                async with session.post(token_url, data=token_data, headers=headers, proxy=proxy) as response:
                    response.raise_for_status()
                    token_info = await response.json()

            if "error" in token_info:
                raise HTTPException(status_code=400, detail=f"GitHub OAuth error: {token_info['error']}")

            access_token = token_info.get("access_token")
            if not access_token:
                raise HTTPException(status_code=400, detail="Failed to get access token")

            user_url = "https://api.github.com/user"
            user_headers = {"Authorization": f"token {access_token}", "Accept": "application/vnd.github.v3+json"}
            async with aiohttp.ClientSession() as session:
                async with session.get(user_url, headers=user_headers, proxy=proxy) as response:
                    response.raise_for_status()
                    user_info = await response.json()

            return {
                "source": "github",
                "id": str(user_info["id"]),
                "username": user_info["login"],
                "email": user_info.get("email", ""),
                "homepage": user_info.get("html_url", ""),
                "avatar_url": user_info.get("avatar_url", ""),
            }

        except aiohttp.ClientError as e:
            logger.error(f"GitHub API request failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to authenticate with GitHub")

        except Exception as e:
            logger.error(f"Authentication error: {e}")
            raise HTTPException(status_code=500, detail="Authentication failed")

    async def auth_google(self, code):
        try:
            logger.info(f"Google OAuth code: {code}")
            token_url = "https://oauth2.googleapis.com/token"
            token_data = {
                "client_id": self.google_client_id,
                "client_secret": self.google_client_secret,
                "code": code,
                "redirect_uri": self.google_redirect_uri,
                "grant_type": "authorization_code",
            }
            headers = {"Content-Type": "application/x-www-form-urlencoded"}

            proxy = os.getenv("auth_https_proxy", None)
            if proxy:
                logger.info(f"auth_google use proxy: {proxy}")
            async with aiohttp.ClientSession() as session:
                async with session.post(token_url, data=token_data, headers=headers, proxy=proxy) as response:
                    response.raise_for_status()
                    token_info = await response.json()

            if "error" in token_info:
                raise HTTPException(status_code=400, detail=f"Google OAuth error: {token_info['error']}")

            access_token = token_info.get("access_token")
            if not access_token:
                raise HTTPException(status_code=400, detail="Failed to get access token")

            # get user info
            user_url = "https://www.googleapis.com/oauth2/v2/userinfo"
            user_headers = {"Authorization": f"Bearer {access_token}"}
            async with aiohttp.ClientSession() as session:
                async with session.get(user_url, headers=user_headers, proxy=proxy) as response:
                    response.raise_for_status()
                    user_info = await response.json()
            return {
                "source": "google",
                "id": str(user_info["id"]),
                "username": user_info.get("name", user_info.get("email", "")),
                "email": user_info.get("email", ""),
                "homepage": user_info.get("link", ""),
                "avatar_url": user_info.get("picture", ""),
            }

        except aiohttp.ClientError as e:
            logger.error(f"Google API request failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to authenticate with Google")

        except Exception as e:
            logger.error(f"Google authentication error: {e}")
            raise HTTPException(status_code=500, detail="Google authentication failed")

    async def send_sms(self, phone_number):
        return await self.aliyun_client.send_sms(phone_number)

    async def check_sms(self, phone_number, verify_code):
        ok = await self.aliyun_client.check_sms(phone_number, verify_code)
        if not ok:
            return None
        return {
            "source": "phone",
            "id": phone_number,
            "username": phone_number,
            "email": "",
            "homepage": "",
            "avatar_url": "",
        }

    def verify_jwt_token(self, token):
        try:
            payload = jwt.decode(token, self.jwt_secret_key, algorithms=[self.jwt_algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except Exception as e:
            logger.error(f"verify_jwt_token error: {e}")
            raise HTTPException(status_code=401, detail="Could not validate credentials")

    def verify_worker_token(self, token):
        return token == self.worker_secret_key
