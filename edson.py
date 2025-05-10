#from asyncio import tasks
import re
import discord
from discord.ext import commands
import os
import time
import sys
import random
import datetime
import io
from PIL import Image, ImageSequence
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import mysql.connector
from openai import OpenAI
from openai import AzureOpenAI
import json
import numpy as np
import pickle
import aiohttp
from bs4 import BeautifulSoup
#import speech_recognition as sr
#from google_auth_oauthlib.flow import InstalledAppFlow
#from googleapiclient.discovery import build

endpoint = ""
model_name = ""
deployment = ""

subscription_key = ""
api_version = ""

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

search_client = OpenAI(api_key="")
intents = discord.Intents.all()
intents.members=True
#intents.typing=True
activity=discord.Streaming(name="向上反映问题请私信我",url="https://www.youtube.com/watch?v=dQw4w9WgXcQ&pp=ygUIcmlja3JvbGw%3D",platform="YouTube")
main_bot=commands.Bot(command_prefix="/",intents=intents,activity=activity)

MYSQL_CONFIG_MC = {
    "host": "",
    "user": "",
    "password": "",
    "database": ""
}

MYSQL_CONFIG = {
    "host": "",
    "user": "",
    "password": "",
    "database": ""
}

# -----------------------------------
# Embedding and Similarity Functions
# -----------------------------------
def get_embedding(text: str) -> list:
    # Add input validation and cleaning
    if not text or not isinstance(text, str):
        print(f"Warning: Invalid input for embedding: {type(text)}")
        return []  # Return empty embedding for invalid input
        
    # Truncate text if it's too long (OpenAI has token limits)
    if len(text) > 8000:
        text = text[:8000]
        
    try:
        response = search_client.embeddings.create(
            input=[text],
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []  # Return empty embedding on error

def cosine_similarity(v1, v2):
    a, b = np.array(v1), np.array(v2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def get_relevant_memories(user_id: str, query: str, top_k: int = 5):
    query_embedding = get_embedding(query)
    connection = mysql.connector.connect(**MYSQL_CONFIG)
    cursor = connection.cursor()
    cursor.execute("SELECT message, role, embedding FROM messages WHERE user_id = %s AND embedding IS NOT NULL", (user_id,))

    memories = []
    for msg, role, blob in cursor.fetchall():
        memory_embedding = pickle.loads(blob)
        sim = cosine_similarity(query_embedding, memory_embedding)
        memories.append((sim, role, msg))

    cursor.close()
    connection.close()
    memories.sort(reverse=True)
    return memories[:top_k]

# -----------------------------------
# Minecraft Knowledge Database Functions
# -----------------------------------
def get_minecraft_connection():
    """Creates and returns a connection to the Minecraft database"""
    try:
        connection = mysql.connector.connect(**MYSQL_CONFIG_MC)
        return connection
    except mysql.connector.Error as err:
        print(f"Error connecting to Minecraft database: {err}")
        return None

def search_minecraft_blocks(search_term=None, limit=10):
    """Search for blocks in the Minecraft database"""
    try:
        connection = get_minecraft_connection()
        if not connection:
            return []
            
        cursor = connection.cursor(dictionary=True)
        
        if search_term:
            # Search by name or displayName
            sql = """
            SELECT * FROM blocks 
            WHERE name LIKE %s OR displayName LIKE %s
            ORDER BY id
            LIMIT %s
            """
            cursor.execute(sql, (f"%{search_term}%", f"%{search_term}%", limit))
        else:
            # Return all blocks up to limit
            sql = "SELECT * FROM blocks ORDER BY id LIMIT %s"
            cursor.execute(sql, (limit,))
            
        results = cursor.fetchall()
        cursor.close()
        connection.close()
        return results
    except mysql.connector.Error as err:
        print(f"Error searching Minecraft blocks: {err}")
        return []

def search_minecraft_items(search_term=None, limit=10):
    """Search for items in the Minecraft database"""
    try:
        connection = get_minecraft_connection()
        if not connection:
            return []
            
        cursor = connection.cursor(dictionary=True)
        
        if search_term:
            # Search by name or displayName
            sql = """
            SELECT * FROM items 
            WHERE name LIKE %s OR displayName LIKE %s
            ORDER BY id
            LIMIT %s
            """
            cursor.execute(sql, (f"%{search_term}%", f"%{search_term}%", limit))
        else:
            # Return all items up to limit
            sql = "SELECT * FROM items ORDER BY id LIMIT %s"
            cursor.execute(sql, (limit,))
            
        results = cursor.fetchall()
        cursor.close()
        connection.close()
        return results
    except mysql.connector.Error as err:
        print(f"Error searching Minecraft items: {err}")
        return []

def search_minecraft_biomes(search_term=None, limit=10):
    """Search for biomes in the Minecraft database"""
    try:
        connection = get_minecraft_connection()
        if not connection:
            return []
            
        cursor = connection.cursor(dictionary=True)
        
        if search_term:
            # Search by name or displayName
            sql = """
            SELECT * FROM biomes 
            WHERE name LIKE %s OR category LIKE %s OR dimension LIKE %s OR displayName LIKE %s
            ORDER BY id
            LIMIT %s
            """
            cursor.execute(sql, (f"%{search_term}%", f"%{search_term}%", f"%{search_term}%", f"%{search_term}%", limit))
        else:
            # Return all biomes up to limit
            sql = "SELECT * FROM biomes ORDER BY id LIMIT %s"
            cursor.execute(sql, (limit,))
            
        results = cursor.fetchall()
        cursor.close()
        connection.close()
        return results
    except mysql.connector.Error as err:
        print(f"Error searching Minecraft biomes: {err}")
        return []

def search_minecraft_entities(search_term=None, limit=10):
    """Search for entities in the Minecraft database"""
    try:
        connection = get_minecraft_connection()
        if not connection:
            return []
            
        cursor = connection.cursor(dictionary=True)
        
        if search_term:
            # Search by name or displayName or type or category
            sql = """
            SELECT * FROM entities 
            WHERE name LIKE %s OR displayName LIKE %s OR type LIKE %s OR category LIKE %s
            ORDER BY id
            LIMIT %s
            """
            cursor.execute(sql, (f"%{search_term}%", f"%{search_term}%", f"%{search_term}%", f"%{search_term}%", limit))
        else:
            # Return all entities up to limit
            sql = "SELECT * FROM entities ORDER BY id LIMIT %s"
            cursor.execute(sql, (limit,))
            
        results = cursor.fetchall()
        cursor.close()
        connection.close()
        return results
    except mysql.connector.Error as err:
        print(f"Error searching Minecraft entities: {err}")
        return []

def get_minecraft_recipes(item_id=None, limit=10):
    """Get recipes from the Minecraft database"""
    try:
        connection = get_minecraft_connection()
        if not connection:
            return []
            
        cursor = connection.cursor(dictionary=True)
        
        if item_id:
            # Get recipes for a specific result item
            sql = """
            SELECT r.*, i.name as result_name, i.displayName as result_display_name
            FROM recipe r
            JOIN items i ON r.result_id = i.id
            WHERE r.result_id = %s
            LIMIT %s
            """
            cursor.execute(sql, (item_id, limit))
        else:
            # Return all recipes up to limit
            sql = """
            SELECT r.*, i.name as result_name, i.displayName as result_display_name
            FROM recipe r
            JOIN items i ON r.result_id = i.id
            LIMIT %s
            """
            cursor.execute(sql, (limit,))
            
        recipes = cursor.fetchall()
        
        # For each recipe, get its ingredients and shape
        for recipe in recipes:
            recipe_id = recipe['id']
            
            # Get shaped recipe pattern if exists
            shape_sql = """
            SELECT rs.row_idx, rs.col_idx, i.name as item_name, i.displayName as item_display_name
            FROM recipe_shape rs
            LEFT JOIN items i ON rs.item_id = i.id
            WHERE rs.recipe_id = %s
            ORDER BY rs.row_idx, rs.col_idx
            """
            cursor.execute(shape_sql, (recipe_id,))
            shape_data = cursor.fetchall()
            
            if shape_data:
                recipe['shape'] = shape_data
            
            # Get recipe ingredients
            ingredients_sql = """
            SELECT ri.ingredient_idx, i.name as item_name, i.displayName as item_display_name
            FROM recipe_ingredient ri
            JOIN items i ON ri.item_id = i.id
            WHERE ri.recipe_id = %s
            ORDER BY ri.ingredient_idx
            """
            cursor.execute(ingredients_sql, (recipe_id,))
            ingredients = cursor.fetchall()
            
            if ingredients:
                recipe['ingredients'] = ingredients
        
        cursor.close()
        connection.close()
        return recipes
    except mysql.connector.Error as err:
        print(f"Error retrieving Minecraft recipes: {err}")
        return []

def get_minecraft_enchantments(search_term=None, limit=10):
    """Search for enchantments in the Minecraft database"""
    try:
        connection = get_minecraft_connection()
        if not connection:
            return []
            
        cursor = connection.cursor(dictionary=True)
        
        if search_term:
            # Search by name, displayName or category
            sql = """
            SELECT * FROM enchantments 
            WHERE name LIKE %s OR displayName LIKE %s OR category LIKE %s
            ORDER BY id
            LIMIT %s
            """
            cursor.execute(sql, (f"%{search_term}%", f"%{search_term}%", f"%{search_term}%", limit))
        else:
            # Return all enchantments up to limit
            sql = "SELECT * FROM enchantments ORDER BY id LIMIT %s"
            cursor.execute(sql, (limit,))
            
        results = cursor.fetchall()
        cursor.close()
        connection.close()
        return results
    except mysql.connector.Error as err:
        print(f"Error searching Minecraft enchantments: {err}")
        return []

def get_minecraft_effects(search_term=None, limit=10):
    """Search for effects/potions in the Minecraft database"""
    try:
        connection = get_minecraft_connection()
        if not connection:
            return []
            
        cursor = connection.cursor(dictionary=True)
        
        if search_term:
            # Search by name, displayName or type
            sql = """
            SELECT * FROM effects 
            WHERE name LIKE %s OR displayName LIKE %s OR type LIKE %s
            ORDER BY id
            LIMIT %s
            """
            cursor.execute(sql, (f"%{search_term}%", f"%{search_term}%", f"%{search_term}%", limit))
        else:
            # Return all effects up to limit
            sql = "SELECT * FROM effects ORDER BY id LIMIT %s"
            cursor.execute(sql, (limit,))
            
        results = cursor.fetchall()
        cursor.close()
        connection.close()
        return results
    except mysql.connector.Error as err:
        print(f"Error searching Minecraft effects: {err}")
        return []

def get_minecraft_foods(search_term=None, limit=10):
    """Search for food items in the Minecraft database"""
    try:
        connection = get_minecraft_connection()
        if not connection:
            return []
            
        cursor = connection.cursor(dictionary=True)
        
        if search_term:
            # Search by name or displayName
            sql = """
            SELECT * FROM foods 
            WHERE name LIKE %s OR displayName LIKE %s
            ORDER BY foodPoints DESC
            LIMIT %s
            """
            cursor.execute(sql, (f"%{search_term}%", f"%{search_term}%", limit))
        else:
            # Return all foods up to limit, ordered by food points (nutrition value)
            sql = "SELECT * FROM foods ORDER BY foodPoints DESC LIMIT %s"
            cursor.execute(sql, (limit,))
            
        results = cursor.fetchall()
        cursor.close()
        connection.close()
        return results
    except mysql.connector.Error as err:
        print(f"Error searching Minecraft foods: {err}")
        return []

def search_minecraft_general(search_term, limit=20):
    """
    Comprehensive search across multiple Minecraft tables
    Returns consolidated results from different categories
    """
    results = {
        "blocks": [],
        "items": [],
        "biomes": [],
        "entities": [],
        "enchantments": [],
        "effects": [],
        "foods": []
    }
    
    # Only search if we have a term
    if search_term and len(search_term.strip()) > 0:
        results["blocks"] = search_minecraft_blocks(search_term, limit)
        results["items"] = search_minecraft_items(search_term, limit)
        results["biomes"] = search_minecraft_biomes(search_term, limit)
        results["entities"] = search_minecraft_entities(search_term, limit)
        results["enchantments"] = get_minecraft_enchantments(search_term, limit)
        results["effects"] = get_minecraft_effects(search_term, limit)
        results["foods"] = get_minecraft_foods(search_term, limit)
    
    # Filter out empty categories
    return {k: v for k, v in results.items() if v}

def get_minecraft_advancements():
    """Get all advancements from the Minecraft database"""
    try:
        connection = get_minecraft_connection()
        if not connection:
            return {}
            
        cursor = connection.cursor(dictionary=True)
        sql = "SELECT * FROM advancements"
        cursor.execute(sql)
        results = cursor.fetchall()
        cursor.close()
        connection.close()
        
        # Convert to a more usable dictionary format
        advancements = {}
        for row in results:
            advancements[row['key']] = row['value']
        
        return advancements
    except mysql.connector.Error as err:
        print(f"Error retrieving Minecraft advancements: {err}")
        return {}

def get_minecraft_materials():
    """Get material categories and their items"""
    try:
        connection = get_minecraft_connection()
        if not connection:
            return {}
            
        cursor = connection.cursor(dictionary=True)
        
        # First get all categories
        sql = "SELECT * FROM material_category ORDER BY name"
        cursor.execute(sql)
        categories = cursor.fetchall()
        
        materials = {}
        
        # For each category, get its items
        for category in categories:
            cat_id = category['id']
            cat_name = category['name']
            
            # Get items in this category
            item_sql = """
            SELECT mi.*, i.name as item_name, i.displayName as item_display_name
            FROM material_item mi
            JOIN items i ON mi.item_code = i.name
            WHERE mi.category_id = %s
            ORDER BY mi.value
            """
            cursor.execute(item_sql, (cat_id,))
            items = cursor.fetchall()
            
            materials[cat_name] = items
        
        cursor.close()
        connection.close()
        return materials
    except mysql.connector.Error as err:
        print(f"Error retrieving Minecraft materials: {err}")
        return {}

# -----------------------------------
# User and Message Storage
# -----------------------------------
def update_user_profile(user):
    canonical = f"{user.name}#{user.discriminator}"
    current_alias = user.display_name
    try:
        connection = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = connection.cursor()
        cursor.execute("SELECT aliases FROM user_profiles WHERE user_id = %s", (str(user.id),))
        row = cursor.fetchone()
        if row is None:
            aliases = [current_alias] if current_alias != user.name else []
            sql = "INSERT INTO user_profiles (user_id, canonical_name, aliases) VALUES (%s, %s, %s)"
            cursor.execute(sql, (str(user.id), canonical, json.dumps(aliases)))
        else:
            stored_aliases = json.loads(row[0]) if row[0] else []
            if current_alias not in stored_aliases and current_alias != user.name:
                stored_aliases.append(current_alias)
                sql = "UPDATE user_profiles SET aliases = %s WHERE user_id = %s"
                cursor.execute(sql, (json.dumps(stored_aliases), str(user.id)))
        connection.commit()
    except mysql.connector.Error as err:
        print(f"Error updating user profile: {err}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def get_user_profile(user_id):
    try:
        connection = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT canonical_name, aliases FROM user_profiles WHERE user_id = %s", (user_id,))
        result = cursor.fetchone()
        if result:
            result['aliases'] = json.loads(result['aliases']) if result['aliases'] else []
            return result
        else:
            return None
    except mysql.connector.Error as err:
        print(f"Error retrieving user profile: {err}")
        return None
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def insert_message(user, role, message, attachments=None, reference_id=None, thread_id=None):
    update_user_profile(user)
    try:
        connection = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = connection.cursor()

        embedding = get_embedding(message)
        embedding_blob = pickle.dumps(embedding)

        # Insert message with thread_id and reference_id support
        sql_msg = "INSERT INTO messages (user_id, role, message, embedding, reference_id, thread_id) VALUES (%s, %s, %s, %s, %s, %s)"
        cursor.execute(sql_msg, (str(user.id), role, message, embedding_blob, reference_id, thread_id))
        message_id = cursor.lastrowid

        # Insert attachments if any
        if attachments:
            sql_att = "INSERT INTO attachments (message_id, type, content) VALUES (%s, %s, %s)"
            for att in attachments:
                if att["type"] == "image_url":
                    cursor.execute(sql_att, (message_id, "image", att["image_url"]["url"]))
                elif att["type"] == "text":
                    cursor.execute(sql_att, (message_id, "text", att["text"]))

        # If this message starts a new thread, create thread entry
        if thread_id and not reference_id:
            try:
                # Check if thread exists
                cursor.execute("SELECT COUNT(*) FROM message_threads WHERE thread_id = %s", (thread_id,))
                if cursor.fetchone()[0] == 0:
                    # Create new thread
                    thread_name = message[:50] + "..." if len(message) > 50 else message
                    cursor.execute("INSERT INTO message_threads (thread_id, thread_name) VALUES (%s, %s)", 
                                 (thread_id, thread_name))
            except mysql.connector.Error as err:
                print(f"Error creating thread entry: {err}")

        connection.commit()
    except mysql.connector.Error as err:
        print(f"Error inserting message: {err}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

    # Optional: Update channel history if channel info is available
    try:
        if hasattr(user, 'guild') and hasattr(user, 'channel'):
            channel_id = str(user.channel.id)
            connection = mysql.connector.connect(**MYSQL_CONFIG)
            cursor = connection.cursor()
            
            # Get current position (count of messages in channel)
            cursor.execute("SELECT COUNT(*) FROM channel_history WHERE channel_id = %s", (channel_id,))
            position = cursor.fetchone()[0] + 1
            
            # Insert into channel history
            cursor.execute(
                "INSERT INTO channel_history (channel_id, message_id, thread_id, position) VALUES (%s, %s, %s, %s)",
                (channel_id, message_id, thread_id, position)
            )
            connection.commit()
            cursor.close()
            connection.close()
    except Exception as e:
        print(f"Error updating channel history: {e}")
        
def get_conversation_context(user_id, limit=50):
    try:
        connection = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT role, message, created_at FROM messages WHERE user_id = %s ORDER BY created_at DESC LIMIT %s", (user_id, limit))
        return cursor.fetchall()
    except mysql.connector.Error as err:
        print(f"Error retrieving messages: {err}")
        return []
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def extract_user_names_from_text(ctx, text):
    user_ids = set()
    name_map = {}
    for member in ctx.guild.members:
        profile = get_user_profile(str(member.id))
        canonical = profile['canonical_name'] if profile else f"{member.name}#{member.discriminator}"
        name_map[canonical.lower()] = member
        name_map[member.display_name.lower()] = member

        if profile and profile.get('aliases'):
            for alias in profile['aliases']:
                name_map[alias.lower()] = member

    for name, member in name_map.items():
        pattern = r'\b' + re.escape(name) + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            if member.id == ctx.author.id:
                continue
            user_ids.add(str(member.id))
    return list(user_ids)


def get_user_alias_map(db_cursor):
    """
    Retrieves all user aliases and canonical names from user_profiles.
    Returns a dict: { user_id: {name: str, aliases: List[str]} }
    """
    db_cursor.execute("SELECT user_id, canonical_name, aliases FROM user_profiles")
    rows = db_cursor.fetchall()
    alias_map = {}
    for row in rows:
        user_id, canonical_name, aliases_json = row
        try:
            aliases = json.loads(aliases_json) if aliases_json else []
        except Exception as e:
            aliases = []
        alias_map[user_id] = {
            "name": canonical_name,
            "aliases": aliases
        }
    return alias_map

# -----------------------------------
# Multi-User Context Builder
# -----------------------------------
async def build_multicontext(ctx, question): 
    # Regular context building code follows
    user_id = str(ctx.author.id)
    profile = get_user_profile(user_id)
    display_name = profile.get('canonical_name', ctx.author.display_name) if profile else ctx.author.display_name

    def format_ts(ts):
        if hasattr(ts, "strftime"):
            return ts.astimezone(datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
        return "unknown time"

    # Get alias map from database
    connection = mysql.connector.connect(**MYSQL_CONFIG)
    cursor = connection.cursor()
    alias_map = get_user_alias_map(cursor)
    connection.close()

    # IMPROVEMENT 1: Enhanced named entity recognition
    def extract_mentioned_user_ids(question, alias_map):
        mentioned_ids = set()
        
        # Match full usernames, IDs, and aliases
        for uid, data in alias_map.items():
            all_names = [data["name"]] + data["aliases"]
            
            # Check for exact matches first (case insensitive)
            for name in all_names:
                # Check for exact matches of the nickname (surrounded by word boundaries)
                if re.search(r'\b' + re.escape(name.lower()) + r'\b', question.lower()):
                    mentioned_ids.add(uid)
                    break
                
                # Check for partial matches of uncommon nicknames (3+ characters)
                if len(name) >= 3 and name.lower() in question.lower():
                    mentioned_ids.add(uid)
                    break
            
            # Also check for mentions in the format "who is X" or "X是谁"
            for name in all_names:
                patterns = [
                    rf"who\s+is\s+{re.escape(name)}",
                    rf"{re.escape(name)}\s*是\s*谁",
                    rf"谁\s*是\s*{re.escape(name)}"
                ]
                for pattern in patterns:
                    if re.search(pattern, question, re.IGNORECASE):
                        mentioned_ids.add(uid)
                        break
        
        return mentioned_ids

    # IMPROVEMENT 2: More aggressive named entity extraction with flexible patterns
    def extract_potential_names(question):
        # Extract potential name entities from the question
        potential_names = []
        
        # Define patterns for different question types
        identity_patterns = [
            # English patterns
            r"who\s+is\s+([a-zA-Z0-9_]+)",                    # who is X
            r"do\s+you\s+know\s+([a-zA-Z0-9_]+)",             # do you know X
            r"tell\s+me\s+about\s+([a-zA-Z0-9_]+)",           # tell me about X
            r"know\s+(?:who|what)\s+([a-zA-Z0-9_]+)\s+is",    # know who/what X is
            r"what\s+(?:do\s+)?you\s+(?:know|think)\s+(?:about|of)\s+([a-zA-Z0-9_]+)",  # what do you know/think about/of X
            r"(?:any|some)(?:thing)?\s+about\s+([a-zA-Z0-9_]+)",  # anything/something about X
            r"information\s+(?:about|on)\s+([a-zA-Z0-9_]+)",  # information about/on X

            # Chinese patterns
            r"([a-zA-Z0-9_]+)\s*是\s*谁",                     # X是谁
            r"谁\s*是\s*([a-zA-Z0-9_]+)",                     # 谁是X
            r"你\s*(?:知道|认识)\s*([a-zA-Z0-9_]+)",           # 你知道/认识X
            r"(?:知道|认识)\s*([a-zA-Z0-9_]+)\s*吗",           # 知道/认识X吗
            r"(?:说一说|聊一聊|告诉我)\s*(?:你\s*)?(?:对\s*)?(?:于\s*)?([a-zA-Z0-9_]+)\s*的", # 说一说你对于X的看法
            r"(?:关于|对于|有关)\s*([a-zA-Z0-9_]+)",           # 关于/对于/有关X
            r"([a-zA-Z0-9_]+)\s*(?:是\s*(?:什么|哪位|怎样))",  # X是什么/哪位/怎样
            r"跟\s*(?:我)?\s*(?:说|聊|讲)(?:一?说|一?聊|一?讲)?\s*(?:有关|关于)?\s*([a-zA-Z0-9_]+)\s*的?(?:事情|信息|资料|内容|消息)?", # 跟我说说有关X的事情
            r"(?:介绍|说明|描述)\s*(?:一?下)?\s*(?:有关|关于)?\s*([a-zA-Z0-9_]+)",  # 介绍一下有关X
            r"(?:查询|查找|搜索)\s*(?:有关|关于)?\s*([a-zA-Z0-9_]+)\s*的?(?:信息|资料)?",  # 查询有关X的信息
            r"(?:给我|帮我|为我)\s*(?:整理|收集|提供)\s*(?:有关|关于)?\s*([a-zA-Z0-9_]+)\s*的?(?:资料|信息|内容)?",  # 给我整理有关X的资料
        ]
        
        # Try all patterns
        for pattern in identity_patterns:
            matches = re.finditer(pattern, question, re.IGNORECASE)
            for match in matches:
                name = match.group(1).strip()
                if name and len(name) >= 2:  # Only consider names with 2+ chars
                    potential_names.append(name)
        
        # Also try to extract standalone names that might be uniquely identifying
        # This helps with questions like "ttp?"
        words = re.findall(r'\b([a-zA-Z0-9_]{3,})\b', question)
        for word in words:
            if word.lower() not in ["who", "what", "when", "where", "why", "how", 
                                  "the", "and", "but", "for", "你", "我", "他", "她",
                                  "是", "的", "了", "吗", "在", "有", "这", "那", "这个", "那个"]:
                potential_names.append(word)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_names = []
        for name in potential_names:
            if name.lower() not in seen:
                seen.add(name.lower())
                unique_names.append(name)
                
        return unique_names

    # First check for directly mentioned users
    mentioned_ids = extract_mentioned_user_ids(question, alias_map)
    
    # Then check for potential name mentions that might not be in our alias_map
    potential_names = extract_potential_names(question)
    
    # IMPROVEMENT 3: Create a user identity lookup map
    user_identity_info = {}
    
    # ENHANCEMENT: Detect if the query is asking about a previous conversation topic
    # This helps with follow-up questions about topics mentioned by other users
    is_reference_to_previous_topic = False
    reference_topic_keywords = []
    
    # Check for reference patterns to previous messages or conversations
    reference_patterns = [
        # English patterns
        r"what(?:'s| is| was)?\s+(?:that|this)\s+(?:about|referring to)",
        r"(?:what|who)(?:'s| is| was)?\s+(?:the|that|this)\s+(?:\w+)?\s*(?:you|they|he|she|we|edson)?\s*(?:talking|speaking|referring|mentioned|discussing|saying|posting)\s+(?:about|to|on)",
        r"(?:tell|explain)\s+(?:me|us)\s+(?:more|again)\s+(?:about|on|regarding)\s+(?:that|this|the|those)",
        r"(?:what|who)(?:'s| is| was)?\s+(?:the|that|this)\s+(?:\w+)?\s*(?:story|topic|discussion|conversation|joke|idea|information|news|update)",
        r"(?:what|who)(?:'s| is| was)?\s+(?:the|that|this)\s+(?:\w+)?\s*(?:mean|implying|suggesting)",
        # Chinese patterns
        r"(?:这|那)(?:个|些)?\s*(?:是什么|指的是|在说什么|什么意思|怎么回事)",
        r"(?:他|她|它|他们|她们|它们|大家|有人)?\s*(?:刚才|之前|前面|刚刚)?\s*(?:在说|在讲|说的|讲的|提到的|聊的|谈论的)\s*(?:是什么|什么|谁|哪个)",
        r"(?:跟|关于|说|讲)(?:一下|说|讲)?\s*(?:这个|那个|之前的|刚才的|前面的|刚刚的)?\s*(?:话题|内容|消息|故事|笑话|新闻|信息)",
        r"(?:你|他|她|它|他们|她们|它们|大家|有人)?\s*(?:刚才|之前|前面|刚刚)?\s*(?:的|说的|讲的|提到的|聊的|谈论的).*(?:是指|指的是|什么意思)"
    ]
    
    # Extract specific topic keywords from the question
    topic_keyword_patterns = [
    # Look for specific topic keywords
    r"(?:about|regarding|concerning|on)\s+(?:the\s+)?(?P<topic>\w+)",
    r"(?:关于|有关|谈谈|说说|讲讲)\s*(?P<topic>\w+)"
]
    # Check if the question is asking about a referenced topic
    for pattern in reference_patterns:
        if re.search(pattern, question.lower()):
            is_reference_to_previous_topic = True
            break

    # If we detect this is a reference to a previous topic, extract potential topic keywords
    topic_keywords = []
    if is_reference_to_previous_topic:
        # Extract any specific topic words mentioned in the query
        for pattern in topic_keyword_patterns:
            matches = re.finditer(pattern, question.lower())
            for match in matches:
                if 'topic' in match.groupdict():
                    topic = match.group('topic')
                    if topic and len(topic) > 2:
                        topic_keywords.append(topic)
        
        # If no specific topics found but question indicates reference,
        # extract potential topic keywords from the question
        if not topic_keywords:
            # Extract nouns and potential topic words
            words = re.findall(r'\b([a-zA-Z\u4e00-\u9fff]{2,})\b', question)
            for word in words:
                if word.lower() not in ["what", "who", "when", "where", "why", "how", 
                                      "tell", "more", "about", "that", "this", "the", 
                                      "你", "我", "他", "她", "是", "的", "了", "吗", 
                                      "在", "有", "这", "那", "这个", "那个"]:
                    topic_keywords.append(word)
    with open("Minecraft_Crafting_Examples.md", 'r', encoding='utf-8') as f:
        crafting_examples = f.read()
    # Build the system context
    system_context = (
        "REMEMBER the following instructions.\n"
        "You are an AI assistant named Edson, created by qqrey.\n"
        "You are the Minecraft expert.\n"
        "You are able to provide accurate and relevant information about Minecraft.\n\n"
        "Messages include metadata formatted as: "
        "[Timestamp] Username (ID: user_id, Role: role): message\n\n"

        "Time Awareness:\n"
        "- When a referenced message appears as '[Referenced Message: yyyy-mm-dd hh:mm:ss]', you MUST use that timestamp as a reference point for time difference calculation.\n"
        "- Treat the latest user message timestamp as the current time.\n"
        "- NEVER guess or estimate the time difference. ALWAYS subtract the referenced timestamp from the current timestamp.\n"
        "- If a message is referenced for timing only, IGNORE its content and ONLY calculate time difference from its timestamp.\n"
        "- You can know the current time using the timestamp of the most recent message.\n"
        "- Use the latest timestamp to compute elapsed time.\n\n"

        "Time Calculation Examples:\n"
        "- To calculate time difference between timestamps, use this systematic approach:\n"
        "  1. Convert both timestamps to total seconds or a comparable format\n"
        "  2. Subtract the earlier timestamp from the later timestamp\n"
        "  3. Convert the difference back to a human-readable format\n\n"
        "- Example 1 (Same day): [2025-04-19 11:53:37] to [2025-04-19 12:13:00]\n"
        "  * Calculate hours: 12 - 11 = 1 hour\n"
        "  * Calculate minutes: (1 hour × 60) + (13 - 53) = 60 - 40 = 20 minutes\n"
        "  * Calculate seconds: 0 - 37 = -37 seconds (borrow 1 minute)\n"
        "  * Final result: 19 minutes and 23 seconds elapsed\n\n"
        "- Example 2 (Different days): [2025-04-19 01:37:00] to [2025-04-20 11:48:22]\n"
        "  * Calculate days: 20 - 19 = 1 day (24 hours)\n"
        "  * Calculate hours: 24 + (11 - 01) = 34 hours\n"
        "  * Calculate minutes: (34 hours × 60) + (48 - 37) = 2040 + 11 = 2051 minutes\n"
        "  * Calculate seconds: 22 - 0 = 22 seconds\n"
        "  * Convert to readable format: 2051 minutes = 34 hours and 11 minutes\n"
        "  * Final result: 1 day, 10 hours, 11 minutes, and 22 seconds elapsed\n\n"
        "- Always verify your calculation by breaking it into clear steps\n"
        "- For timestamps across months or years, account for the exact number of days in each month\n\n"

        "Message Handling:\n"
        "- AVOID saying things like '[Recent @ 2025-04-19 11:31:52] qqrey#0:'. This is support information only.\n"
        "- Do NOT repeat or quote any part of the message log.\n"
        "- Answer ONLY the most recent user message.\n"
        "- If a message is referenced, treat it as context only. Do NOT interpret it as a new user query.\n"
        "- Do NOT output any debug, metadata, or formatting tags in your reply.\n\n"

        "Reasoning and Structure:\n"
        "- Understand and organize responses based on user profile, time context, and available database info.\n"
        "- Correlate and logically organize relevant data for coherent responses.\n"
        "- Craft responses that are clear, accurate, and avoid nonsensical or unrelated language.\n\n"

        "Steps:\n"
        "- Identify the target user and relevant characteristics.\n"
        "- Analyze current timestamp to ensure temporal accuracy.\n"
        "- Retrieve verified info from memory or database if needed.\n"
        "- Structure the data logically and make accurate correlations.\n"
        "- Provide a clear, concise, and sensible response.\n\n"

        "Output Format:\n"
        "- Deliver responses in a coherent paragraph.\n"
        "- Ensure replies are relevant to user, time-aware, and logically connected.\n"
        "- When context is long (3000+ tokens), summarize with bullet points.\n\n"
        "- Make sure the message is clear and easy to read.\n"

        "Markdown Formatting:\n"
        "- Format ALL explanations with proper markdown structure.\n"
        "- EACH section MUST have blank lines before and after it.\n"
        "- NEVER place descriptions on the same line as headings.\n\n"
        "- MUST follow the markdown formatting rules:\n"
        "- Do NOT OVERUSE markdown formatting.\n"
        "- Use **bold** for important points.\n"
        "- Use *italics* for emphasis.\n"
        "- Use `inline code` for code snippets.\n"
        "- Use ```code blocks``` for larger code snippets.\n"
        "- Use [links](https://example.com) for references.\n"
        "- Use bullet points for lists.\n"
        "- Use headings for different sections.\n"
        "- Use line breaks to separate sections.\n\n"
        "Example:\n"
        f"{crafting_examples}\n\n"

        "Notes:\n"
        "- Always assume timestamps are accurate.\n"
        "- Be attentive when user references past messages vaguely.\n"
        "- Avoid language shifts or unnecessary verbosity.\n"
        "- Be concise, coherent, and mindful in your reasoning process.\n\n"

        "Memory Recall with Timestamps:\n"
        "- When asked to retrieve specific past messages related to a topic:\n"
        "  1. Search your memory for semantically relevant past messages.\n"
        "  2. For each match, return the exact timestamp and original message.\n"
        "  3. End your response with a short summary of the content and timing.\n"
        "- Example:\n"
        "  You mentioned Liz in the following messages:\n"
        "  Time: 2025-02-28 17:26:12\n"
        "  Message: I really like her unarchived karaoke.\n"
        "  Time: 2025-02-28 17:26:20\n"
        "  Message: Listening to it while driving in Euro Truck Simulator 2 makes me feel happy.\n"
        "  Time: 2025-02-28 17:26:29\n"
        "  Message: The quiet atmosphere in her streams is exactly what I enjoy—everyone quietly listens to Liz sing.\n"
        "  Summary: You sent three messages about Elizabeth Rose Bloodframe (Liz) around 17:26, expressing your love for her unarchived karaoke and the peaceful vibe of her streams.\n"
        "  Note: The timestamps are in UTC+8.\n"
        "- If you are unsure about the content of a message, do not guess. Just provide the timestamp and the message.\n"
    )

    context_messages = [{"role": "system", "content": system_context}]

        # Add user profile info
    if profile:
        profile_summary = []
        if "canonical_name" in profile:
            profile_summary.append(f"Canonical name: {profile['canonical_name']}")
        if "preferences" in profile:
            profile_summary.append(f"Preferences: {profile['preferences']}")
        if "notes" in profile:
            profile_summary.append(f"Notes: {profile['notes']}")
        if profile_summary:
            context_messages.append({
                "role": "system",
                "content": f"User profile for {display_name}:\n" + "\n".join(profile_summary)
            })

    # NEW FEATURE: Get recent channel messages to add conversation context
    channel_messages = []
    try:
        # Fetch recent messages from the channel (limit to 20 to improve context awareness)
        async for message in ctx.channel.history(limit=20):
            # Skip bot messages and the current message
            if message.author.bot and message.author.id != main_bot.user.id:
                continue
            if message.id == ctx.message.id:
                continue
                
            author_profile = get_user_profile(str(message.author.id))
            author_name = author_profile.get('canonical_name', message.author.display_name) if author_profile else message.author.display_name
            
            channel_messages.append({
                "author_id": str(message.author.id),
                "author_name": author_name,
                "content": message.content,
                "timestamp": message.created_at,
                "formatted_ts": format_ts(message.created_at),
                "is_bot": message.author.bot,
                "bot_id": message.author.id if message.author.bot else None
            })
    except Exception as e:
        print(f"Error fetching channel history: {e}")
        # If we can't get channel history, continue without it

    # Add recent channel messages to context with enhanced structure for better relevance
    if channel_messages:
        # Extract topics from recent conversations
        recent_topics = set()
        
        # First pass: identify potential topics from recent messages
        for msg in channel_messages[:15]:  # Look at last 15 messages
            # Extract meaningful words (ignoring common words)
            content = msg["content"].lower()
            words = re.findall(r'\b([a-zA-Z\u4e00-\u9fff]{2,})\b', content)
            for word in words:
                if word.lower() not in ["what", "who", "when", "where", "why", "how", 
                                       "tell", "more", "about", "that", "this", "the", 
                                       "你", "我", "他", "她", "是", "的", "了", "吗", 
                                       "在", "有", "这", "那", "这个", "那个"] and len(word) > 1:
                    recent_topics.add(word.lower())
        
        # Enhanced channel context message
        channel_context = "Recent channel conversation (newest to oldest):\n"
        for i, msg in enumerate(channel_messages[:15]):  # Limit to 15 messages
            # Mark bot's messages with [BOT] for clearer identification
            author_prefix = "[BOT] " if msg["is_bot"] else ""
            channel_context += f"[{msg['formatted_ts']}] {author_prefix}{msg['author_name']}: {msg['content']}\n"
        
        # ENHANCEMENT: For reference questions, specifically highlight relevant previous messages
        if is_reference_to_previous_topic:
            # Reformat the channel context to emphasize relevant messages
            highlight_context = "SPECIAL ATTENTION - The current question appears to be asking about previously discussed topics. "\
                              "These recent messages might be particularly relevant:\n\n"
            
            # Look for relevant messages based on topic keywords
            relevant_count = 0
            for msg in channel_messages[:15]:
                is_relevant = False
                content_lower = msg["content"].lower()
                
                # If we have specific topic keywords, check if they appear in messages
                if topic_keywords:
                    for keyword in topic_keywords:
                        if keyword.lower() in content_lower:
                            is_relevant = True
                            break
                
                # If no specific keyword identified or not found, do broader matching
                if not is_relevant and recent_topics:
                    # Check which recent topics appear in this message
                    for topic in recent_topics:
                        if topic in content_lower and len(topic) > 2:  # Only consider meaningful topics
                            is_relevant = True
                            break
                
                # If this message is deemed relevant, highlight it
                if is_relevant:
                    author_prefix = "[BOT] " if msg["is_bot"] else ""
                    highlight_context += f"[RELEVANT] [{msg['formatted_ts']}] {author_prefix}{msg['author_name']}: {msg['content']}\n"
                    relevant_count += 1
            
            # Only add highlighted context if we found relevant messages
            if relevant_count > 0:
                context_messages.append({
                    "role": "system",
                    "content": highlight_context
                })
            
        # Always add full recent conversation context
        context_messages.append({
            "role": "system",
            "content": channel_context
        })

    # IMPROVEMENT 4: Add explicit user identity information
    # This creates a clear map between nicknames and user IDs for the AI to reference
    id_map = {}
    for uid, data in alias_map.items():
        name = data["name"]
        aliases = data["aliases"]
        id_map[uid] = {"name": name, "aliases": aliases}
    
    if id_map:
        user_id_info = []
        for uid, data in id_map.items():
            user_id_info.append(f"User ID: {uid}, Name: {data['name']}, Aliases: {', '.join(data['aliases'])}")
        
        # ONLY add this information when asking about user identities
        identity_related = any(term in question.lower() for term in ['who is', 'who are', '是谁', '是谁？'])
        
        if identity_related and user_id_info:
            context_messages.append({
                "role": "system", 
                "content": "User Identity Information:\n" + "\n".join(user_id_info)
            })

    # Add recent messages from the current user
    recent_messages = get_conversation_context(user_id, limit=10)
    if recent_messages:
        context_messages.append({
            "role": "system",
            "content": f"Recent messages from the user ({display_name}):"
        })
        
        for msg in reversed(recent_messages):
            ts = format_ts(msg.get("created_at"))
            role = msg.get("role", "user")
            content = msg.get("message", "")
            context_messages.append({
                "role": role,
                "content": f"[Recent @ {ts}] {display_name}: {content}"
            })

    # ENHANCED FEATURE: Get relevant memories across all users based on query similarity
    # This extends beyond just the current user's memories
    query_embedding = get_embedding(question)
    
    def get_global_relevant_memories(query_embedding, top_k=8):
        connection = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = connection.cursor()
        
        # Get a sample of recent messages from the database across all users
        cursor.execute("SELECT user_id, message, role, embedding, created_at FROM messages WHERE embedding IS NOT NULL ORDER BY created_at DESC LIMIT 200")
        
        memories = []
        user_names = {}  # Cache to store user names
        
        for user_id, msg, role, blob, created_at in cursor.fetchall():
            # Get user display name (with caching to reduce DB queries)
            if user_id not in user_names:
                try:
                    cursor.execute("SELECT canonical_name FROM user_profiles WHERE user_id = %s", (user_id,))
                    name_row = cursor.fetchone()
                    user_names[user_id] = name_row[0] if name_row else f"User-{user_id}"
                except:
                    user_names[user_id] = f"User-{user_id}"
            
            # Calculate similarity with query
            memory_embedding = pickle.loads(blob)
            sim = cosine_similarity(query_embedding, memory_embedding)
            
            # Add to memories if similarity is above threshold (0.65 is a reasonable starting point)
            if sim > 0.65:
                memories.append({
                    "similarity": sim,
                    "user_id": user_id, 
                    "user_name": user_names[user_id],
                    "role": role,
                    "message": msg,
                    "timestamp": format_ts(created_at) if created_at else "unknown time"
                })
        
        cursor.close()
        connection.close()
        
        # Sort by similarity (highest first) and return top k
        memories.sort(key=lambda x: x["similarity"], reverse=True)
        return memories[:top_k]
    
    # Get and add globally relevant memories with enhanced relevance matching
    global_memories = get_global_relevant_memories(query_embedding)
    
    if global_memories:
        # ENHANCEMENT: Group memories by topic similarity for better context organization
        # First, cluster memories by similarity to each other
        grouped_memories = {}
        memory_assigned = set()
        
        # Add the first memory to its own group
        if len(global_memories) > 0:
            first_memory = global_memories[0]
            grouped_memories["group_0"] = [first_memory]
            memory_assigned.add(0)
        
        # For reference questions, prioritize memories that match topic keywords
        if is_reference_to_previous_topic and topic_keywords:
            # Create a special group for topic-relevant memories
            topic_relevant_memories = []
            
            for i, mem in enumerate(global_memories):
                if i in memory_assigned:
                    continue
                
                # Check if memory contains any of our topic keywords
                is_topic_relevant = False
                for keyword in topic_keywords:
                    if keyword.lower() in mem["message"].lower():
                        is_topic_relevant = True
                        break
                
                if is_topic_relevant:
                    topic_relevant_memories.append(mem)
                    memory_assigned.add(i)
            
            # If we found topic-relevant memories, add them as a group
            if topic_relevant_memories:
                grouped_memories["topic_reference"] = topic_relevant_memories
        
        # IMPROVEMENT: Track conversation threads from channel history
        # This helps connect multiple messages from different users in the same conversation
        
        # First, organize channel messages by threads
        conversation_threads = {}
        current_thread_id = "main"
        conversation_threads[current_thread_id] = []
        
        if channel_messages:
            # Identify messages that are replies to other messages to establish threads
            message_reply_map = {}
            
            # Pass 1: Map messages and their replies
            for msg in channel_messages:
                msg_id = msg.get("message_id", "unknown")
                reference_id = msg.get("reference_id")
                if reference_id:
                    if reference_id not in message_reply_map:
                        message_reply_map[reference_id] = []
                    message_reply_map[reference_id].append(msg_id)
            
            # Pass 2: Organize messages into conversation threads
            processed_msgs = set()
            thread_counter = 0
            
            # Helper function to process a message chain
            def process_thread(msg_id, thread_id):
                # Find the message in channel_messages
                for msg in channel_messages:
                    if msg.get("message_id") == msg_id and msg_id not in processed_msgs:
                        processed_msgs.add(msg_id)
                        if thread_id not in conversation_threads:
                            conversation_threads[thread_id] = []
                        conversation_threads[thread_id].append(msg)
                        
                        # Process any replies to this message
                        if msg_id in message_reply_map:
                            for reply_id in message_reply_map[msg_id]:
                                process_thread(reply_id, thread_id)
            
            # Start processing from root messages (messages with no reference)
            for msg in channel_messages:
                msg_id = msg.get("message_id", "unknown")
                reference_id = msg.get("reference_id")
                
                if not reference_id and msg_id not in processed_msgs:
                    thread_id = f"thread_{thread_counter}"
                    thread_counter += 1
                    process_thread(msg_id, thread_id)
            
            # Add any leftover messages to the main thread
            for msg in channel_messages:
                msg_id = msg.get("message_id", "unknown")
                if msg_id not in processed_msgs:
                    conversation_threads["main"].append(msg)
                    processed_msgs.add(msg_id)
        
        # Group remaining memories by similarity
        for i, mem1 in enumerate(global_memories):
            if i in memory_assigned:
                continue
                
            similar_memories = [mem1]
            memory_assigned.add(i)
            
            for j, mem2 in enumerate(global_memories):
                if j in memory_assigned:
                    continue
                
                # Check if messages are similar enough to group
                similarity = cosine_similarity(
                    get_embedding(mem1["message"]), 
                    get_embedding(mem2["message"])
                )
                
                if similarity > 0.85:  # High threshold for grouping
                    similar_memories.append(mem2)
                    memory_assigned.add(j)
            
            # Create a group ID based on main topics
            topic_words = re.findall(r'\b([a-zA-Z\u4e00-\u9fff]{3,})\b', mem1["message"])
            topic_id = "_".join(topic_words[:2]) if topic_words else f"group_{i}"
            grouped_memories[topic_id] = similar_memories
        
        # Format memories by groups
        relevant_context = "Semantically relevant messages from all users:\n\n"
        
        # First add topic-relevant memories if available
        if "topic_reference" in grouped_memories:
            relevant_context += "--- MESSAGES RELEVANT TO YOUR SPECIFIC QUERY ---\n"
            for mem in grouped_memories["topic_reference"]:
                relevant_context += f"[{mem['timestamp']}] {mem['user_name']}: {mem['message']}\n\n"
            
            relevant_context += "--- OTHER RELATED MESSAGES ---\n"
        
        # Then add other memory groups
        for group_id, memories in grouped_memories.items():
            if group_id == "topic_reference":
                continue  # Already added
                
            for mem in memories:
                relevant_context += f"[{mem['timestamp']}] {mem['user_name']} (similarity: {mem['similarity']:.2f}): {mem['message']}\n\n"
            
        # Add conversation thread context for better flow tracking
        if conversation_threads and any(msgs for msgs in conversation_threads.values() if msgs):
            thread_context = "\n--- CONVERSATION THREADS ---\n\n"
            
            # Find the most relevant thread based on query similarity
            thread_relevance = {}
            for thread_id, msgs in conversation_threads.items():
                if not msgs:
                    continue
                    
                # Calculate average similarity for this thread
                thread_text = " ".join([msg["content"] for msg in msgs])
                thread_embedding = get_embedding(thread_text)
                thread_similarity = cosine_similarity(query_embedding, thread_embedding)
                thread_relevance[thread_id] = thread_similarity
            
            # Sort threads by relevance
            sorted_threads = sorted(thread_relevance.items(), key=lambda x: x[1], reverse=True)
            
            for thread_id, relevance in sorted_threads[:3]:  # Show top 3 most relevant threads
                if relevance < 0.65:  # Skip low relevance threads
                    continue
                    
                thread_context += f"--- CONVERSATION THREAD (relevance: {relevance:.2f}) ---\n"
                # Display messages in chronological order
                sorted_msgs = sorted(conversation_threads[thread_id], key=lambda x: x["timestamp"] if isinstance(x["timestamp"], datetime.datetime) else datetime.datetime.min)
                
                for msg in sorted_msgs:
                    author_prefix = "[BOT] " if msg["is_bot"] else ""
                    thread_context += f"[{msg['formatted_ts']}] {author_prefix}{msg['author_name']}: {msg['content']}\n"
                thread_context += "\n"
            
            # Add thread context to overall context
            relevant_context += thread_context
            
        context_messages.append({
            "role": "system",
            "content": relevant_context
        })

    # IMPROVEMENT 5: Add relevant memories with semantic similarity
    relevant_memories = get_relevant_memories(user_id, question, top_k=5)
    for score, role, msg in relevant_memories:
        context_messages.append({
            "role": "system",
            "content": f"The following is a relevant past message from the {role} (similarity score: {score:.2f}): {msg}"
        })

    # IMPROVEMENT 6: Enhanced detection of identity questions
    is_identity_question = False
    target_names = []
    
    # Get potential target names from the question
    potential_names = extract_potential_names(question)
    if potential_names:
        is_identity_question = True
        target_names = potential_names
        
        
    # Further analyze the question to determine identity intent
    identity_indicators = [
        # English
        r"who\s+is", "about", "tell me", "do you know", "recognize", "identify", 
        "familiar with", "heard of", "opinion", "thoughts on", "view",
        # Chinese
        r"是谁", r"谁是", r"认识", r"知道", r"了解", r"听说过",
        r"看法", r"想法", r"意见", r"观点"
    ]
    
    # If no names found but has identity indicators, flag as potential identity question
    if not target_names and any(re.search(pattern, question, re.IGNORECASE) for pattern in identity_indicators):
        is_identity_question = True
    
    # Try to extract any unique looking terms as potential names (3+ alphanumeric)
   
        unique_terms = re.findall(r'\b([a-zA-Z0-9_]{3,})\b', question)
        for term in unique_terms:
            if term.lower() not in ["who", "what", "when", "where", "why", "how", 
                                   "the", "and", "but", "for", "你", "我", "他", "她",
                                   "是", "的", "了", "吗", "在", "有", "这", "那"]:
                target_names.append(term)
    
    # For identity questions, prioritize adding context from matching users
    if is_identity_question and target_names:
        # Find matching users by name or alias
        matching_ids = set()
        matched_names = {}
        
        for target_name in target_names:
            target_name_lower = target_name.lower()
            
            for uid, data in alias_map.items():
                all_names = [data["name"].lower()] + [alias.lower() for alias in data["aliases"]]
                
                # Check for exact matches
                if target_name_lower in all_names:
                    matching_ids.add(uid)
                    matched_names[target_name] = data["name"]
                    break
                    
                # Check for partial matches (when target is 3+ chars)
                if len(target_name) >= 3:
                    for name in all_names:
                        # Match if target is contained within a name or vice versa
                        if target_name_lower in name or name in target_name_lower:
                            matching_ids.add(uid)
                            matched_names[target_name] = data["name"]
                            break
        
        # Add special context notes about the identity matches
        if matching_ids:
            context_messages.append({
                "role": "system",
                "content": f"IMPORTANT USER IDENTIFICATION: The current question is asking about identity information."
            })
            
            for uid in matching_ids:
                data = alias_map.get(uid)
                if data:
                    matched_target = [t for t in target_names if t.lower() in [n.lower() for n in [data["name"]] + data["aliases"]]]
                    matched_target_str = ', '.join(matched_target) if matched_target else 'this user'
                    
                    context_messages.append({
                        "role": "system",
                        "content": f"IMPORTANT: '{matched_target_str}' in the question refers to user ID {uid} with canonical name '{data['name']}' and aliases {data['aliases']}"
                    })

    # Add messages from mentioned users
    for mentioned_id in mentioned_ids:
        mentioned_profile = alias_map.get(mentioned_id)
        mentioned_name = mentioned_profile['name'] if mentioned_profile else mentioned_id
        context_messages.append({
            "role": "system",
            "content": f"The following messages are from the user with ID {mentioned_id} (also known as {mentioned_name}):"
        })
        mentioned_msgs = get_conversation_context(mentioned_id, limit=30)
        for mem in reversed(mentioned_msgs):
            ts = format_ts(mem.get("created_at"))
            role = mem.get("role", "user")
            content = mem.get("message", "")
            context_messages.append({
                "role": role,
                "content": f"[Recent @ {ts}] {mentioned_name}: {content}"
            })

    # IMPROVEMENT 7: Add comprehensive guidance for identity questions
    if is_identity_question:
# Construct guidance based on the specific question
        identity_guidance = "RESPONSE GUIDANCE: "
        
        if target_names:
            names_str = '", "'.join(target_names)
            identity_guidance += f"The user is asking about \"{names_str}\". "
            identity_guidance += "If this entity appears in ANY context messages, you MUST identify them with all available information including:"
            identity_guidance += "\n- Their username/ID"
            identity_guidance += "\n- Any aliases or alternative names"
            identity_guidance += "\n- Information from their past messages"
            identity_guidance += "\n- Their relationship to the current user if known"
        else:
            # Generic identity question without specific target
            identity_guidance += "The user appears to be asking about someone's identity. "
            identity_guidance += "Review ALL context messages carefully for relevant people or entities."
        
        identity_guidance += "\n\nDO NOT respond with 'I don't know' if information about the person exists anywhere in the context."
        
        # Check if the question seems to be about a specific user in a specific way
        user_opinion_patterns = [
            r"(?:what|tell|give).+(?:opinion|thought|view|impression).+about\s+([a-zA-Z0-9_]+)",
            r"(?:说一说|聊一聊|告诉我)\s*(?:你\s*)?(?:对\s*)?(?:于\s*)?([a-zA-Z0-9_]+)\s*的(?:看法|想法|意见)"
        ]
        
        for pattern in user_opinion_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                identity_guidance += "\n\nThis appears to be a request for your opinion about a specific person. "
                identity_guidance += "In addition to identifying who they are, you should provide factual information based on their interaction history."
                break

            context_messages.append({
                "role": "system",
                "content": identity_guidance
        })
        
        # Add additional instruction to cross-reference
        identity_processing = (
            "IDENTITY PROCESSING INSTRUCTIONS:\n"
            "1. When the user asks who someone is, ALWAYS check the entire context for mentions of this person\n"
            "2. Look for messages FROM this person in the context (marked with their name)\n"
            "3. If you find messages from them, use these to help identify who they are\n"
            "4. NEVER say you don't know someone if they appear anywhere in the context\n"
            "5. EXAMINE all messages in the database context, not just recent conversations\n"
        )
                
        context_messages.append({
            "role": "system",
            "content": identity_processing
            })

    # Add current user message
    context_messages.append({
        "role": "user",
        "content": question
    })

    return context_messages

def add_reference_message_with_time(context_messages, ref, user_displayname=None):
    try:
        reply_user = ref.author
        reply_author = f"{user_displayname or reply_user.display_name} ({reply_user.name}#{reply_user.discriminator})"
        reply_text = ref.content.strip()
        reply_time = ref.created_at.astimezone(datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")

        # Add the reference message content
        reference_block = (
            f"[Referenced Message: {reply_time}] {reply_author} said:\n{reply_text}"
            if reply_text else
            f"[Referenced Message: {reply_time}] {reply_author} sent a message with no text."
        )

        # Add guidance to help AI understand the reference is about the image
        if any(attachment.content_type and attachment.content_type.startswith("image/") for attachment in ref.attachments):
            reference_block += "\n(Note: This referenced message contains an image. If the user asks 'who is this' or similar, they are likely asking about the character/person in the image.)"
        
        # Add to context messages
        context_messages.append({
            "role": "user",
            "content": f"{reference_block}\n(Note: This message is referenced only for timing/context. The current user query follows.)"
        })
        
        # Add image attachments from the reference as separate entries
        for attachment in ref.attachments:
            if attachment.content_type and attachment.content_type.startswith("image/"):
                context_messages.append({
                    "role": "system",
                    "content": f"IMPORTANT: The referenced message contains this image. When the user asks 'who is this', '这是谁', or similar identity questions, they are referring to the character or person shown in this image:"
                })
                context_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "image_url",
                        "image_url": {"url": attachment.url}
                    }]
                })
    except Exception as e:
        print(f"Failed to format referenced message: {e}")

# Enhanced message splitting with special handling for lists and numbered items
async def send_message_in_parts(channel, content, context_messages, model="gpt-4.1-nano-2025-04-14"):
    max_discord_length = 1900  # Using 1900 to be safe
    
    # If content is already short enough, just send it
    if len(content) <= max_discord_length:
        await channel.send(content)
        return
    
    # Special list and numbered item detection
    def is_list_item(line):
        # Check for numbered items (in multiple languages)
        if re.match(r'^\s*\d+\.\s+', line):  # "1. item"
            return True
        if re.match(r'^\s*\d+\)\s+', line):  # "1) item" 
            return True
       
        if re.match(r'^\s*\d+、\s*', line):   # Chinese numbered items "1、item"
            return True
        # Check for bullet points
        if re.match(r'^\s*[•\-\*\+]\s+', line):
            return True
        return False
    
    def find_smart_split_point(text, target_length, min_length=800):
        """Find a smart split point that prioritizes keeping list items together."""
        # If the text is shorter than min_length, no need to split
        if len(text) <= min_length:
            return len(text)
        
        # If text is shorter than target, just return its length
        if len(text) <= target_length:
            return len(text)
        
        # First priority: Look for empty lines (paragraph breaks) within range of target
        search_start = min(len(text), target_length)
        search_end = max(0, search_start - 500)  # Look up to 500 chars before target
        
        # Look for double newlines (paragraph breaks)
        for i in range(search_start, search_end, -1):
            if i+1 < len(text) and text[i:i+2] == '\n\n':
                return i + 2  # Return position after the double newline
        
        # Check if the split would break up a numbered list
        lines = text[:target_length+100].split('\n')
        cumulative_length = 0
        last_good_split = min_length
        
        # Track if we're inside a list
        in_list = False
        list_item_block_start = 0
        
        # For each line, decide if it's a good split point
        for i, line in enumerate(lines):
            line_length = len(line) + 1  # +1 for the newline
            
            # Check for list patterns at the beginning of a line
            is_list_start = is_list_item(line)
            
            # If this is a list item and we weren't in a list before, mark the start
            if is_list_start and not in_list:
                list_item_block_start = cumulative_length
                in_list = True
            
            # If we're past min_length and not in a list, this is a potential split point
            if cumulative_length >= min_length and not in_list:
                last_good_split = cumulative_length
            
            # If this line would put us over the limit
            if cumulative_length + line_length > target_length:
                # If we're in the middle of a list, go back to before the list started
                if in_list and last_good_split > list_item_block_start:
                    return last_good_split
                
                # Otherwise use the last known good split point
                return last_good_split if last_good_split > min_length else target_length
            # If this line ends a list item block
            if in_list and i < len(lines)-1 and not is_list_item(lines[i+1]):
                in_list = False
                # After a list is a good split point
                if cumulative_length >= min_length:
                    last_good_split = cumulative_length + line_length
            
            cumulative_length += line_length
            
            # Empty lines make great split points if we're past the minimum
            if line.strip() == '' and cumulative_length >= min_length:
                last_good_split = cumulative_length
        
        # Second priority: Look for single newlines
        for i in range(search_start, search_end, -1):
            if text[i:i+1] == '\n':
                return i + 1  # Return position after the newline
        
        # Third priority: Try to split at sentence endings
        sentence_boundaries = ['.', '!', '?', '。', '！', '？', '…']
        for i in range(search_start, search_end, -1):
            if text[i:i+1] in sentence_boundaries:
                # Make sure we're actually at the end of a sentence (followed by space or newline)
                if i+1 >= len(text) or text[i+1:i+2].isspace():
                    return i + 1  # Include the punctuation
        
        # Last resort: just split at target length
        return min(target_length, len(text))
    
    # Split the content intelligently
    remaining = content
    message_parts = []
    
    # Actually use the find_smart_split_point function to split the content
    while len(remaining) > 0:
        # If what's left is short enough, add it as the last part
        if len(remaining) <= max_discord_length:
            message_parts.append(remaining)
            break
        
        # Find the best split point using our smart function
        split_point = find_smart_split_point(remaining, max_discord_length)
        message_parts.append(remaining[:split_point])
        remaining = remaining[split_point:]
    
    # Send each part
    for i, part in enumerate(message_parts):
        try:
            # Don't add continuation markers - they confuse the output
            await channel.send(part)
        except discord.errors.HTTPException as e:
            if "Must be 2000 or fewer in length" in str(e):
                # Emergency fallback - split in half
                half = len(part) // 2
                await channel.send(part[:half])
                await channel.send(part[half:])
            else:
                await channel.send(f"Error sending message: {e}")
                
def clean_response(response):
    """Clean response text by removing metadata tags and system artifacts"""
    if not response or not isinstance(response, str):
        return response
        
    # Remove timestamp metadata
    response = re.sub(r'\[Recent\s+@\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\]\s+[^:]+:', '', response)
    response = re.sub(r'\[Recent\s+@\s+[^\]]+\]', '', response)
    
    # Remove username/ID formats 
    response = re.sub(r'[a-zA-Z0-9_]+#\d+:', '', response)  # Remove username#0: format
    response = re.sub(r'\(ID:[^)]+\)', '', response)
    response = re.sub(r'\(User ID:[^)]+\)', '', response)
    
    # Remove other system markers
    response = re.sub(r'\[Referenced Message:[^\]]+\]', '', response)
    response = re.sub(r'\(Note: This message is referenced only for timing/context\.[^)]*\)', '', response)
    
    return response


# -----------------------------------
# GIF Processing Function
# -----------------------------------
async def process_gif(gif_url):
    """
    Processes a GIF by extracting distinct frames to reduce redundancy while preserving content.
    
    Args:
        gif_url: URL of the GIF to process
        
   
        
    Returns:
        List of distinct frame URLs that can be passed to edson
    """
    try:
        print(f"Processing GIF from URL: {gif_url}")
        
        # Download the GIF
        async with aiohttp.ClientSession() as session:
            async with session.get(gif_url) as response:
                if response.status != 200:
                    return [{"type": "image_url", "image_url": {"url": gif_url}}]
                    
                gif_data = await response.read()
                
        # Load the GIF using PIL
        image = Image.open(io.BytesIO(gif_data))
        
        # Check if it's actually a GIF with multiple frames
        if not hasattr(image, 'n_frames') or image.n_frames <= 1:
            return [{"type": "image_url", "image_url": {"url": gif_url}}]
            
        print(f"GIF contains {image.n_frames} frames")
        
        # Extract frames
        frames = []
        for frame in ImageSequence.Iterator(image):
            # Convert to RGB to ensure consistent format
            frame_rgb = frame.convert("RGB")
            # Convert PIL image to numpy array for OpenCV processing
            frame_array = np.array(frame_rgb)
            # Convert RGB to BGR (OpenCV format)
            frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            frames.append(frame_bgr)
            
        # Now we need to identify distinct frames (ignoring similar ones)
        distinct_frames = []
        distinct_indices = []
        
        # Add the first frame
        distinct_frames.append(frames[0])
        distinct_indices.append(0)
        
        # Function to check if a frame is significantly different from existing distinct frames
        def is_distinct(frame, threshold=0.85):
            for distinct_frame in distinct_frames:
                # Resize if dimensions differ
                if frame.shape != distinct_frame.shape:
                    frame_resized = cv2.resize(frame, (distinct_frame.shape[1], distinct_frame.shape[0]))
                else:
                    frame_resized = frame
                
                # Convert to grayscale for SSIM comparison
                frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                distinct_frame_gray = cv2.cvtColor(distinct_frame, cv2.COLOR_BGR2GRAY)
                
                # Calculate similarity score
                score, _ = ssim(frame_gray, distinct_frame_gray, full=True)
                
                # If too similar to any existing frame, it's not distinct
                if score > threshold:
                    return False
                    
            return True
        
        # Check remaining frames
        for i, frame in enumerate(frames[1:], 1):
            # Sample frames to reduce computation (e.g., check every 3rd frame)
            if len(frames) > 30 and i % 3 != 0 and i != len(frames) - 1:
                continue
                
            if is_distinct(frame):
                distinct_frames.append(frame)
                distinct_indices.append(i)
                
        print(f"Identified {len(distinct_frames)} distinct frames out of {image.n_frames}")
        
        # Limit the number of frames to prevent token overuse
        max_frames = 8
        if len(distinct_frames) > max_frames:
            # If we have too many frames, take evenly spaced frames
            indices = np.linspace(0, len(distinct_frames) - 1, max_frames, dtype=int)
            distinct_frames = [distinct_frames[i] for i in indices]
            distinct_indices = [distinct_indices[i] for i in indices]
            
        # Convert frames to URLs (for embedding in messages)
        # We'll save them as temporary files and use discord file upload
        frame_data = []
        for i, frame in enumerate(distinct_frames):
            # Convert BGR back to RGB for PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            # Save to memory buffer
            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG")
            buffer.seek(0)
            
            # The frame data will be returned as a buffer that can be used for file uploads
            frame_data.append({
                "buffer": buffer,
                "frame_number": distinct_indices[i],
                "total_frames": image.n_frames
            })
            
        return frame_data
            
    except Exception as e:
        print(f"Error processing GIF: {str(e)}")
        # Return original GIF URL if processing fails
        return [{"type": "image_url", "image_url": {"url": gif_url}}]

# Function to analyze a GIF and process it for edson
async def analyze_gif_for_edson(ctx, gif_url):
    """
    Analyzes a GIF by extracting key frames and sending them to edson for interpretation.
    
    Args:
        ctx: Discord context
        gif_url: URL of the GIF to analyze
    """
    try:
        # Define the dedicated channel for GIF frames
        FRAME_CHANNEL_ID = 1234567890  # Replace with actual channel ID
        frame_channel = main_bot.get_channel(FRAME_CHANNEL_ID)
        
        if not frame_channel:
            print(f"Error: Could not find channel with ID {FRAME_CHANNEL_ID}")
            return [{"type": "image_url", "image_url": {"url": gif_url}}], None, []
        
        # Let the user know we're processing the GIF
        processing_msg = await ctx.send("Processing GIF... (extracting key frames)")
        
        # Extract distinct frames
        frames = await process_gif(gif_url)
        
        if not frames or isinstance(frames[0], dict):
            await processing_msg.edit(content="Could not process GIF frames. Using original GIF instead.")
            return [{"type": "image_url", "image_url": {"url": gif_url}}], None, []
        
        # Upload distinct frames as individual images
        uploaded_frames = []
        frame_messages = []
        
        await processing_msg.edit(content=f"Found {len(frames)} distinct frames. Processing...")
        
        for i, frame_data in enumerate(frames):
            buffer = frame_data["buffer"]
            frame_number = frame_data["frame_number"]
            total_frames = frame_data["total_frames"]
            
            # Upload each frame to the dedicated frame channel
            file = discord.File(buffer, filename=f"frame_{frame_number}.png")
            frame_msg = await frame_channel.send(
                content=f"GIF frame from {ctx.author.name} - Frame {i+1}/{len(frames)} (original position: {frame_number+1}/{total_frames})", 
                file=file
            )
            
            # Get the URL of the uploaded image
            if frame_msg.attachments:
                uploaded_frames.append({
                    "type": "image_url", 
                    "image_url": {"url": frame_msg.attachments[0].url}
                })
                frame_messages.append(frame_msg)
        
        # Add system message to help edson understand the GIF
        gif_context = {
            "role": "system",
            "content": f"This is an analysis of a GIF with {total_frames} total frames. "
                       f"I've extracted {len(frames)} key distinct frames to show the "
                       f"important visual changes in the animation. Please analyze these "
                       f"frames as a sequence representing the GIF's motion and content."
        }
        
        await processing_msg.edit(content=f"✅ Processed GIF into {len(frames)} key frames for analysis")
        
        # Return the uploaded frames, context, and frame messages for cleanup later
        return uploaded_frames, gif_context, frame_messages
        
    except Exception as e:
        print(f"Error analyzing GIF: {str(e)}")
        if 'processing_msg' in locals():
            await processing_msg.edit(content=f"Error processing GIF: {str(e)}")
        # Return original GIF URL if analysis fails
        return [{"type": "image_url", "image_url": {"url": gif_url}}], None, []
    

# -----------------------------------
# Minecraft Web Search Functions
# -----------------------------------
async def search_minecraft_wiki(query: str) -> str:
    """
    Performs a specialized web search focused on Minecraft wiki content.
    This function enhances the Minecraft database with up-to-date information
    from minecraft.wiki.
    
    Args:
        query: The Minecraft-related search query
        
    Returns:
        The search result as formatted text
    """
    try:
        # Optimize query for Minecraft wiki
        minecraft_search_query = f"minecraft wiki {query} site:minecraft.wiki"
        print(f"Performing Minecraft wiki search for: {minecraft_search_query}")
        
        # Set up system prompt specific to Minecraft content
        system_prompt = (
            "You are a Minecraft expert assistant that provides accurate game information. "
            "Search minecraft.wiki for the requested information and provide a comprehensive answer with relevant facts. "
            "Format your response in these sections when applicable:\n"
            "1. Basic Information (what the item/block/mob/mechanic is)\n"
            "2. Obtaining (how to get it, crafting recipes, spawn conditions)\n"
            "3. Usage (what it can be used for, mechanics)\n"
            "4. Technical Details (game mechanics, version history)\n"
            "Include exact crafting recipes with ingredients when relevant. "
            "Provide specific game mechanics details rather than general descriptions. "
            "Only include information that exists in the actual game, not mods or community-created content. "
            "Always cite minecraft.wiki as your primary source."
        )
        
        # API call using the browsing-capable model
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": minecraft_search_query}
            ],
            max_completion_tokens=800,
            temperature=0.7,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            model=deployment
        )
        
        # Extract the answer
        answer = response.choices[0].message.content
        
        # Format the response with proper markdown
        formatted_answer = f"**Minecraft Information: {query}**\n\n{answer}\n\n*Source: minecraft.wiki*"
        return formatted_answer
        
    except Exception as e:
        print(f"Error in Minecraft wiki search: {e}")
        return f"I encountered an error while searching for Minecraft information: {str(e)}. Please try again with different keywords or check our database instead."

# Enhance the process_minecraft_question function to use the new web search capability
async def process_minecraft_question(message, question):
    """
    Process and respond to Minecraft-related questions using the Minecraft database
    and web search.
    
    Args:
        message: The Discord message object
        question: The question text
    """
    print(f"Processing Minecraft question: {question}")
    search_term = None
    
    # Extract specific search terms
    search_patterns = [
        # English patterns
        r"(?:what|how|tell me about|info on|information about|details on|stats for)\s+(?:is|are|does|do)?\s*(?:the|a|an)?\s*([a-zA-Z0-9_\s]+)\s*(?:in minecraft|in mc)?",
        r"(?:how to|how do i|how do you)\s+(?:craft|make|build|get|obtain|find|create)\s+(?:an|a|the)?\s*([a-zA-Z0-9_\s]+)",
        r"(?:where|how)\s+(?:can i|to|do i|do you)\s+(?:find|get|obtain|mine|harvest|collect)\s+(?:a|an|the)?\s*([a-zA-Z0-9_\s]+)",
        r"minecraft\s+(?:item|block|mob|entity|biome|recipe|enchant|effect|food)\s*:\s*([a-zA-Z0-9_\s]+)",
        r"(?:tell me about|what is|what are|who is|where is)\s+([a-zA-Z0-9_\s]+)\s+(?:in minecraft|in mc)",
        # Chinese patterns
        r"(?:如何|怎么|怎样)(?:制作|合成|做|得到|获得|找到)\s*([a-zA-Z0-9_\u4e00-\u9fff\s]+)",
        r"(?:查询|查找|搜索)\s*(?:我的世界|minecraft|mc)?\s*(?:物品|方块|生物|实体|群系|合成|附魔|效果|食物)?\s*(?:：|:)?\s*([a-zA-Z0-9_\u4e00-\u9fff\s]+)",
        r"([a-zA-Z0-9_\u4e00-\u9fff\s]+)\s*(?:是什么|怎么做|在哪里|在哪儿|哪里有)"
    ]
    
    # Check for specific item/block/entity names in the question
    for pattern in search_patterns:
        match = re.search(pattern, question.lower())
        if match:
            search_term = match.group(1).strip()
            break
    
    # If no specific pattern matched, try to extract key terms
    if not search_term:
        # List of common Minecraft terms to ignore in search extraction
        common_terms = {'minecraft', 'mc', 'how', 'what', 'where', 'when', 'why', 'is', 'the', 'a', 'an', 'in', 
                        'to', 'can', 'you', 'tell', 'me', 'about', 'do', 'does', '我的世界', '麦块', '怎么', '如何', 
                        '哪里', '什么', '在哪', '告诉我', '可以', '帮我'}
        
        # Extract potential search terms (words with 3+ chars)
        words = re.findall(r'\b([a-zA-Z0-9_]{3,})\b', question.lower())
        
        # Filter out common terms
        filtered_words = [word for word in words if word not in common_terms]
        
        # Use the most specific term if available
        if filtered_words:
            search_term = filtered_words[0]  # Just use the first term for now
    
    # Default case if we still couldn't extract a search term
    if not search_term or len(search_term) < 3:
        # Instead of just returning a message, use web search for general Minecraft questions
        async with message.channel.typing():
            # Use the entire question for web search since we couldn't extract a specific term
            web_response = await search_minecraft_wiki(question)
        return web_response
    
    # Start typing indicator to show processing
    async with message.channel.typing():
        # Check if this is a request that would benefit from web search
        is_complex_question = any(term in question.lower() for term in 
                             ['detailed', 'explain', 'exactly how', 'mechanics', 'technical', 
                              'version history', 'update', 'changed', 'difference', 'compare',
                              '详细', '解释', '具体怎么', '机制', '技术', '版本历史', '更新', '变化', '区别', '比较'])
        
        if is_complex_question:
            # For complex questions, use the enhanced web search
            response = await search_minecraft_wiki(question)
            return response
            
        # Special case detection
        is_crafting_question = any(term in question.lower() for term in ['craft', 'make', 'build', 'recipe', 'how to', '如何制作', '怎么做', '怎么合成', '合成表'])
        is_location_question = any(term in question.lower() for term in ['find', 'where', 'location', 'biome', 'spawn', '在哪里', '哪里有', '在哪儿', '哪个群系'])
        is_enchantment_question = any(term in question.lower() for term in ['enchant', 'enchanting', 'enchantment', '附魔', '附魔台'])
        is_effect_question = any(term in question.lower() for term in ['potion', 'effect', 'status', 'effects', '药水', '效果', '状态效果'])
        is_food_question = any(term in question.lower() for term in ['food', 'eat', 'hunger', 'nutrition', '食物', '吃', '饥饿'])
        
        # Get comprehensive search results
        results = search_minecraft_general(search_term)
        
        if not results:
            # If database has no results, try web search
            web_response = await search_minecraft_wiki(search_term)
            return web_response
        
        # Prioritize results based on question type
        primary_category = None
        
        if is_crafting_question:
            if "items" in results:
                primary_category = "items"
                # For a crafting question, we need to get the recipes
                for item in results["items"]:
                    item_recipes = get_minecraft_recipes(item["id"])
                    if item_recipes:
                        # Format the recipe information
                        recipe_info = format_minecraft_recipes(item_recipes, item["displayName"])
                        return recipe_info
        elif is_location_question:
            if "biomes" in results:
                primary_category = "biomes"
            elif "entities" in results:
                primary_category = "entities"
        elif is_enchantment_question:
            if "enchantments" in results:
                primary_category = "enchantments"
        elif is_effect_question:
            if "effects" in results:
                primary_category = "effects"
        elif is_food_question:
            if "foods" in results:
                primary_category = "foods"
        
        # If no primary category was determined, choose based on available results
        if not primary_category:
            # Priority order for general queries
            for category in ["items", "blocks", "entities", "biomes", "enchantments", "effects", "foods"]:
                if category in results:
                    primary_category = category
                    break
        
        # Generate response based on the main result type
        if primary_category:
            response = format_minecraft_results(results, primary_category, search_term, question)
            await message.channel.send(response)
        else:
            # Generic response with all results
            response = "Here's what I found about Minecraft "
            categories = list(results.keys())
            for i, category in enumerate(categories):
                if i == len(categories) - 1 and len(categories) > 1:
                    response += f"and {category}"
                else:
                    response += f"{category}" + (", " if i < len(categories) - 2 else "")
            
            response += f" related to '{search_term}':\n\n"
            
            # Limit to 2-3 results per category to avoid huge messages
            for category, items in results.items():
                response += f"**{category.title()}**:\n"
                for i, item in enumerate(items[:3]):  # Limit to 3 items per category
                    display_name = item.get("displayName", item.get("name", "Unknown"))
                    response += f"• {display_name}\n"
                if len(items) > 3:
                    response += f"...and {len(items) - 3} more {category}\n"
                response += "\n"
            
            response += "Would you like more specific information about any of these?"
            await message.channel.send(response)

def format_minecraft_recipes(recipes, item_name):
    """Format recipe information for a Minecraft item"""
    if not recipes:
        return f"I couldn't find any crafting recipes for {item_name}."
    
    response = f"**Crafting Recipes for {item_name}**:\n\n"
    
    for i, recipe in enumerate(recipes[:3]):  # Limit to 3 recipes to avoid huge messages
        recipe_type = recipe.get("type", "Unknown")
        
        if recipe_type == "shaped":
            response += "**Shaped Crafting**:\n"
            
            # Create a visual representation of the crafting grid
            grid = {}
            if "shape" in recipe:
                for cell in recipe["shape"]:
                    row = cell.get("row_idx", 0)
                    col = cell.get("col_idx", 0)
                    item_display_name = cell.get("item_display_name", "Air")
                    if not item_display_name:
                        item_display_name = "Air"
                    
                    if row not in grid:
                        grid[row] = {}
                    grid[row][col] = item_display_name
            
            # Display the crafting grid
            for row in range(3):
                row_text = "| "
                for col in range(3):
                    cell_content = grid.get(row, {}).get(col, "Air")
                    if cell_content == "Air":
                        row_text += "□ | "
                    else:
                        # Truncate long names
                        short_name = cell_content[:5] + "..." if len(cell_content) > 8 else cell_content
                        row_text += f"{short_name} | "
                response += row_text + "\n"
                
            response += f"\nMakes: {recipe.get('count', 1)}× {item_name}\n\n"
            
        elif recipe_type == "shapeless":
            response += "**Shapeless Crafting**:\n"
            if "ingredients" in recipe:
                ingredients = [ing.get("item_display_name", ing.get("item_name", "Unknown")) for ing in recipe["ingredients"]]
                response += "Ingredients: " + ", ".join(ingredients) + "\n"
                response += f"Makes: {recipe.get('count', 1)}× {item_name}\n\n"
                
        else:
            response += f"**{recipe_type.title()} Recipe**:\n"
            if "ingredients" in recipe:
                ingredients = [ing.get("item_display_name", ing.get("item_name", "Unknown")) for ing in recipe["ingredients"]]
                response += "Ingredients: " + ", ".join(ingredients) + "\n"
                response += f"Makes: {recipe.get('count', 1)}× {item_name}\n\n"
    
    if len(recipes) > 3:
        response += f"...and {len(recipes) - 3} more recipe(s)\n"
        
    return response

def format_minecraft_results(results, primary_category, search_term, original_question):
    """Format minecraft search results based on the primary type of results"""
    if not results or primary_category not in results:
        return f"I couldn't find specific information about '{search_term}' in the Minecraft database."
    
    items = results[primary_category]
    if not items:
        return f"I couldn't find specific information about '{search_term}' in the Minecraft {primary_category} database."
    
    # For now, just focus on the top match
    top_item = items[0]
    
    # Basic properties every item should have
    name = top_item.get("displayName", top_item.get("name", "Unknown"))
    
    # Start building response
    response = f"**Information about {name} in Minecraft**\n\n"
    
    # Different formatting based on the category
    if primary_category == "blocks":
        response += f"• **Type**: Block\n"
        if "material" in top_item:
            response += f"• **Material**: {top_item['material']}\n"
        if "hardness" in top_item:
            response += f"• **Hardness**: {top_item['hardness']}\n"
        if "stackSize" in top_item:
            response += f"• **Stack Size**: {top_item['stackSize']}\n"
        if "diggable" in top_item:
            response += f"• **Can be mined**: {'Yes' if top_item['diggable'] else 'No'}\n"
        if "transparent" in top_item:
            response += f"• **Transparent**: {'Yes' if top_item['transparent'] else 'No'}\n"
        if "resistance" in top_item:
            response += f"• **Blast Resistance**: {top_item['resistance']}\n"
        
    elif primary_category == "items":
        response += f"• **Type**: Item\n"
        if "stackSize" in top_item:
            response += f"• **Stack Size**: {top_item['stackSize']}\n"
        
        # For items, also check if we have crafting recipe information
        recipes = get_minecraft_recipes(top_item.get("id"))
        if recipes:
            # If this is specifically a crafting question, go into more detail
            if any(term in original_question.lower() for term in 
                   ['craft', 'make', 'build', 'recipe', 'how to', '如何制作', '怎么做', '怎么合成', '合成表']):
                return format_minecraft_recipes(recipes, name)
            else:
                response += f"• **Craftable**: Yes\n"
        else:
            response += f"• **Craftable**: No\n"
            
    elif primary_category == "entities":
        response += f"• **Type**: Entity\n"
        if "type" in top_item:
            response += f"• **Entity Type**: {top_item['type'].title()}\n"
        if "category" in top_item:
            response += f"• **Category**: {top_item['category'].title()}\n"
        if "width" in top_item and "height" in top_item:
            response += f"• **Size**: {top_item['width']}×{top_item['height']} blocks\n"
        if "maxHealth" in top_item:
            response += f"• **Max Health**: {top_item['maxHealth']} hearts\n"
        
    elif primary_category == "biomes":
        response += f"• **Type**: Biome\n"
        if "category" in top_item:
            response += f"• **Category**: {top_item['category'].title()}\n"
        if "temperature" in top_item:
            response += f"• **Temperature**: {top_item['temperature']}\n"
        if "precipitation" in top_item:
            response += f"• **Precipitation**: {top_item['precipitation'].title()}\n"
        if "dimension" in top_item:
            response += f"• **Dimension**: {top_item['dimension'].title()}\n"
            
    elif primary_category == "enchantments":
        response += f"• **Type**: Enchantment\n"
        if "category" in top_item:
            response += f"• **Category**: {top_item['category'].title()}\n"
        if "maxLevel" in top_item:
            response += f"• **Maximum Level**: {top_item['maxLevel']}\n"
        if "description" in top_item:
            response += f"• **Description**: {top_item['description']}\n"
            
    elif primary_category == "effects":
        response += f"• **Type**: Status Effect\n"
        if "type" in top_item:
            response += f"• **Effect Type**: {top_item['type'].title()}\n"
        if "duration" in top_item:
            response += f"• **Base Duration**: {top_item['duration']} seconds\n"
        if "amplifier" in top_item:
            response += f"• **Base Amplifier**: {top_item['amplifier']}\n"
            
    elif primary_category == "foods":
        response += f"• **Type**: Food Item\n"
        if "foodPoints" in top_item:
            response += f"• **Hunger Points**: {top_item['foodPoints']}\n"
        if "saturation" in top_item:
            response += f"• **Saturation**: {top_item['saturation']}\n"
    
    # Include ID for all items
    if "id" in top_item:
        response += f"• **ID**: {top_item['id']}\n"
    
    # If there are multiple matches for this category
    if len(items) > 1:
        response += f"\nI found {len(items)} matching {primary_category}. Other matches include: "
        response += ", ".join([item.get("displayName", item.get("name", "Unknown")) for item in items[1:4]])
        if len(items) > 4:
            response += f", and {len(items) - 4} more..."
    
    # If we have results in other categories, mention them
    other_categories = [cat for cat in results if cat != primary_category]
    if other_categories:
        response += f"\n\nI also found matches in other categories: "
        response += ", ".join([f"{cat} ({len(results[cat])})" for cat in other_categories])
        response += ". Would you like information about those instead?"
    
    return response


def classify_message(message_content: str , bot_in_recent_convo, minecraft_context) -> str:
    system_msg = {
        "role": "system",
        "content": (
            "You are an intent classifier that determines if a Discord message requires a response from the AI assistant named **Edson**.\n"
            "Be EXTREMELY SELECTIVE — respond 'yes' **only** if the message is clearly and deliberately directed at Edson for minecraft information, minecraft help, or minecraft interaction.\n\n"
            
            "Reply 'yes' **only if** the message:\n"
            "- Directly @mentions Edson (e.g., `<@edson_id>` or name)\n"
            "- Asks a factual or instructional question clearly aimed at Edson\n"
            "- Requests help, advice, explanation, or opinion from Edson\n"
            "- Clearly invites Edson's participation or continuation of a prior exchange\n"
            "- Is a follow-up to an ongoing conversation **with Edson**, not just general chatter\n\n"
            
            "Reply 'no' if:\n"
            "- The message is not explicitly addressing Edson\n"
            "- It appears to be casual or human-to-human conversation\n"
            "- It’s rhetorical, humorous, or expressive without seeking engagement\n"
            "- The user is talking to someone else, including via @mention\n"
            "- It’s a vague or general statement not clearly aimed at a bot\n\n"
            
            "Context awareness:\n"
            + ("- Edson has RECENTLY been active in this conversation" if bot_in_recent_convo else "- Edson has NOT recently been active in this conversation") + "\n"
            + ("- There HAS been recent Minecraft-related discussion in this channel" if minecraft_context else "- There has NOT been recent Minecraft-related discussion in this channel") + "\n\n"
            
            "Judgment notes:\n"
            "- Messages must show **clear intent** to engage Edson.\n"
            "- Do NOT infer intent just from question marks or message tone.\n"
            "- Prefer 'no' if there is any ambiguity.\n\n"
            
            "Examples:\n"
            "`<@edson_id> 能帮我看看这个概率问题吗？` → yes (direct help request)\n"
            "`Edson，这题怎么做？` → yes (name used, question asked)\n"
            "`谁能帮我找个种子？` → no (no mention of Edson, general group ask)\n"
            "`你觉得这个建筑风格好看吗？` → no (unclear who is being addressed)\n"
            "`@User123 有什么常态分布的资料吗？` → no (not addressed to Edson)\n\n"
            "`edson还没睡醒` → no (not directed at Edson)\n"
            "`家里有矿的不用买，毕竟铁矿厂出一点黄金也不要紧的对吧？` → no (not a minecraft question)\n"
            "`我想知道如何制作一个自动化农场。` → yes (direct request for help)\n"
            "`不是这种那么大个的` → no (not directed at Edson)\n"
            "`不是这种的，是那种一块小块的，长方形，有塑料盒子包着的` → no (not directed at Edson)\n\n"
            "`https://tenor.com/view/gold-gif-16760445902809235517` → no (not a minecraft question)\n"
            "`就是不清楚当初给我们三个小孩都买多少` → no (not directed at Edson)\n"
            "`家里是有帮我们买黄金的，以前小时候有给我们看过，好像是我们出生之前买的` → no (not directed at Edson)\n"
            "`发一个关键词：我要举报`` → no (not directed at Edson)\n"
            "Respond ONLY with `'yes'` or `'no'`."
        )
    }
    
    user_msg = {
        "role": "user",
        "content": message_content
    }
    try:
        decision = client.chat.completions.create(
            messages=[system_msg, user_msg],
            max_completion_tokens=5,
            temperature=0.1,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            model=deployment
        )
        reply = decision.choices[0].message.content.lower().strip()
        print(f"AI classification decision: {reply}")
        return reply
    
    except Exception as e:
        print(f"Error in should_respond_to_message: {e}")
        # Be conservative on errors - only respond if already in conversation
        return bot_in_recent_convo
    

async def second_classify_message(ctx, message_content: str) -> str:
    """
    Classifies a message to determine if it should be responded to by Edson.
    
    Args:
        ctx: Discord context
        message_content: The content of the message
    
    Returns:
        'yes' or 'no' based on the classification
    """
    
    # Check recent conversation context
    channel_history = []
    try:
        async for msg in ctx.channel.history(limit=10):
            channel_history.append(msg)
    except Exception:
        channel_history = []

    system_msg = {
        "role": "system",
        "content": (
            "You are an intent classifier that determines if a Discord message requires a response from the AI assistant named **Edson**.\n"
            "Be EXTREMELY SELECTIVE — respond 'yes' **only** if the message is clearly and deliberately directed at Edson for information, interaction, help, advice, opinion, or explanation related to Minecraft.\n\n"
            
            "Reply 'yes' **only if** the message:\n"
            "- Directly @mentions Edson (e.g., `<@edson_id>` or name)\n"
            "- Asks a factual or instructional question clearly aimed at Edson\n"
            "- Requests help, advice, explanation, or opinion from Edson\n"
            "- Clearly invites Edson's participation or continuation of a prior exchange\n"
            "- Is a follow-up to an ongoing conversation **with Edson**, not just general chatter\n\n"
            
            "Reply 'no' if:\n"
            "- The message is not explicitly addressing Edson\n"
            "- It appears to be casual or human-to-human conversation\n"
            "- It’s rhetorical, humorous, or expressive without seeking engagement\n"
            "- The user is talking to someone else, including via @mention\n"
            "- It’s a vague or general statement not clearly aimed at a bot\n\n"
            
            "Judgment notes:\n"
            "- Messages must show **clear intent** to engage Edson.\n"
            "- Do NOT infer intent just from question marks or message tone.\n"
            "- Prefer 'no' if there is any ambiguity.\n\n"
            
            "Examples:\n"
            "`<@edson_id> 能帮我看看这个概率问题吗？` → no (not a direct help request for minecraft)\n"
            "`Edson，这题怎么做？` → no (name used, but question asked is not related to Minecraft)\n"
            "`谁能帮我找个种子？` → no (no mention of Edson, general group ask)\n"
            "`你觉得这个建筑风格好看吗？` → no (unclear who is being addressed)\n"
            "`@User123 有什么常态分布的资料吗？` → no (not addressed to Edson)\n\n"
            "`edson还没睡醒` → no (not directed at Edson)\n"
            "`家里有矿的不用买，毕竟铁矿厂出一点黄金也不要紧的对吧？` → no (not a minecraft question)\n"
            "`我想知道如何制作一个自动化农场。` → yes (direct request for help)\n"
            "`不是这种那么大个的` → no (not directed at Edson)\n"
            "`不是这种的，是那种一块小块的，长方形，有塑料盒子包着的` → no (not directed at Edson)\n\n"
            "`https://tenor.com/view/gold-gif-16760445902809235517` → no (not a minecraft question)\n"
            "`https://tenor.com/view/creeper-minecraft-gif-16760445902809235517` → no (not a minecraft question)\n"
            "`就是不清楚当初给我们三个小孩都买多少` → no (not directed at Edson)\n"
            "`家里是有帮我们买黄金的，以前小时候有给我们看过，好像是我们出生之前买的` → no (not directed at Edson)\n"
            "`发一个关键词：我要举报` → no (not directed at Edson)\n"

            "Respond ONLY with `'yes'` or `'no'`."
        )
    }
    
    user_msg = {
        "role": "user",
        "content": message_content
    }
    try:
        decision = client.chat.completions.create(
            messages=[system_msg, user_msg],
            max_completion_tokens=5,
            temperature=0.1,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            model=deployment
        )
        reply = decision.choices[0].message.content.lower().strip()
        print(f"AI classification decision: {reply}")
        return reply
    
    except Exception as e:
        print(f"Error in should_respond_to_message: {e}")

async def apply_dm_message(ctx, message: str) -> bool:
        # Get the DM user information more explicitly
        dm_user = message.author
        user_id = dm_user.id
        username = dm_user.name
        discriminator = getattr(dm_user, 'discriminator', '0000')

        # Format user info for the admin message
        user_info = f"{username}#{discriminator} (ID: {user_id})"

        # Admin notification channels
        admin_user = await main_bot.fetch_user(1234567890) # Replace with actual admin user ID
        admin_channel = await main_bot.fetch_channel(1234567890) # Replace with actual channel ID

        # Check if this is the first DM from this user by retrieving message history
        is_first_dm = True
        try:
            # Get DM channel history
            async for msg in message.channel.history(limit=10):
                # If we find any previous messages from the bot to this user
                # (excluding the current one), then it's not the first DM
                if msg.id != message.id and msg.author.id == main_bot.user.id:
                    is_first_dm = False
                    break
        except Exception as e:
            print(f"Error checking DM history: {e}")
            # If there's an error checking history, assume it's not the first message
            is_first_dm = False

        # Send first-time announcement if applicable
        if is_first_dm:
            announcement = (
                f"""
                # 📝 通报表格

                ## 报告人信息
                **提交人:** {user_info}

                ## 事件详情
                **通报原因:** 

                **地点:** 

                **时间:** 

                **参与人员:**

                ## 事件描述
                *请详细说明事发经过。包括可能有助于审核此案的所有相关信息:*


                ---

                ## 证明材料
                *请附上任何截图、消息记录或其他相关证据*

                ---

                **声明:** 本人在此确认，以上所提供的所有信息真实准确。我明白，提交虚假报告或试图嫁祸于人将导致对本人的处罚。

                *报告编号: #[自动生成]*
                """
            )
            need_pin = await message.channel.send(announcement)
            await message.channel.send("""请复制以上报表格，填写完毕后发送给我。
                                       我会将其转发给管理员。
                                       请注意，提交虚假报告或试图嫁祸于人将导致对本人的处罚。
                                       请注意，聊天室内的所有消息都将被记录。
                                       """)
            await need_pin.pin(reason="通报表格")

            return
        else:
            context_messages = []  # Initialize context_messages
            attachment_blocks = []
            url = " "
            for attachment in message.attachments:
                url += f"{attachment.url} \n"

            for attachment in message.attachments:
                if attachment.content_type:
                    # Handle all image types including GIFs
                    if attachment.content_type.startswith("image/"):
                        # Special handling for GIFs
                        if attachment.content_type == "image/gif":
                            try:
                                # Process GIF to extract distinct frames
                                print(f"Processing GIF attachment in message: {attachment.url}")
                                ctx = await main_bot.get_context(message)
                                uploaded_frames, gif_context, frame_messages = await analyze_gif_for_edson(ctx, attachment.url)

                                # Add frames to attachment blocks
                                attachment_blocks.extend(uploaded_frames)

                                # Add GIF context if available
                                if gif_context:
                                    context_messages.append(gif_context)

                                continue  # Skip standard image handling for GIFs
                            except Exception as e:
                                print(f"Error processing GIF: {str(e)}, falling back to standard image handling")
                                # Fall back to regular image handling if processing fails

                        # Standard image handling for non-GIFs or if GIF processing failed
                        block = {
                            "type": "image_url", 
                            "image_url": {"url": attachment.url}
                        }
                        attachment_blocks.append(block)
                
                    elif attachment.content_type == "text/plain" or attachment.filename.endswith('.txt'):
                        try:
                            async with aiohttp.ClientSession() as session:
                                async with session.get(attachment.url) as resp:
                                    if resp.status == 200:
                                        text_data = await resp.text()
                                        # Don't limit to 2000 chars for file organization tasks
                                        block = {
                                            "type": "text",
                                            "text": f"Attached text file contents:\n{text_data}"
                                        }
                                        attachment_blocks.append(block)
                                        print(f"Processed text attachment: {attachment.filename}")
                        except Exception as e:
                            await message.channel.send(f"Failed to read attachment: {e}")
                            print(f"Error processing attachment: {e}")
                
            if attachment.content_type.startswith("image/"):
                context_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "image_url",
                        "image_url": {"url": attachment.url}
                    }]
                })

            now_time = time.strftime("%Y%m%d", time.localtime())
            report_number = f"#{now_time}_{str(random.randint(1, 999999)).zfill(6)}"
            system_context = {
                "role": "system",
                "content": (
                    "You need to give a summary of the message/report in Chinese.\n"
                    "No need to identify is the message/report real or not.\n"
                    "Do NOT CHANGE the content of the message/report, But only change the 报告编号: #[自动生成]\n"
                    f"CHANGE the '报告编号: #[自动生成]' to '报告编号: #{report_number}'\n"
                    "the summary should be at the last of the message/report\n"
                )
            }

            user_msg = {
                "role": "user",
                "content": message.content
            }

            response = client.chat.completions.create(
                messages=[system_context, user_msg] + context_messages,
                max_completion_tokens=8000,
                temperature=0.5,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                model=deployment
            )
            answer = response.choices[0].message.content.strip()

        if not message.content:
            await admin_user.send(f"来自***{message.author.name}***:{url}")
            await admin_channel.send(f"来自***{message.author.name}***:{url}")
            await message.channel.send("你的信息已经提交给管理员。请耐心等待处理。")
            return

# Format admin DM notifications with better structure and information
        await admin_user.send(f"""
        # 提交通知

        **来源用户:** {message.author.name} (ID: {message.author.id})
        **提交时间:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        **报告编号:** {report_number}

        ## 原始内容
        {message.content}
        """)

        await admin_channel.send(f"""
        # 新提交通知

        **来源用户:** {message.author.name} (ID: {message.author.id}) 
        **提交时间:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


        {answer}
        """)

        if attachment_blocks:
            await admin_user.send(f"Attachments: {url}")
            await admin_channel.send(f"Attachments: {url}")

        await message.channel.send(f"你的报告编号为{report_number}。")
        await message.channel.send("你的报告已经提交给管理员。请耐心等待处理。")


async def should_respond_to_message(ctx, message_content: str) -> bool:
    try:
        print(f"Message content: {message_content}")
        
        # Ignore empty messages or very short content
        if not message_content or len(message_content.strip()) <= 1:
            return False
            
        # Ignore messages that look like commands for other bots
        if message_content.startswith(('!', '/', '$', '#', '>', '.')) and not message_content.lower().startswith('/edson'):
            print("Appears to be a command for another bot - ignoring")
            return False
            
        # NEW: Check for Minecraft specific patterns - these get high priority
        minecraft_indicators = [
            # English: Game terms
            r'\bminecraft\b', r'\bmc\b', r'\bsurvival mode\b', r'\bcreative mode\b', r'\bvanilla\b',
        
            # English: Blocks and items
            r'\bdiamond\b', r'\bnetherite\b', r'\biron\b', r'\bgold\b', r'\bemerald\b',
            r'\bobsidian\b', r'\bbedrock\b', r'\bread\s*stone\b', r'\bglowstone\b', r'\bend\s*stone\b',
        
            # English: Biomes and dimensions
            r'\bnether\b', r'\bend\b', r'\boverworld\b', r'\bthe\s*end\b', r'\bdesert\b',
            r'\bjungle\b', r'\btaiga\b', r'\bplains\b', r'\bocean\b', r'\bswamp\b', r'\bmountain\b',
        
            # English: Entities and mobs
            r'\bcreeper\b', r'\bzombie\b', r'\bskeleton\b', r'\bspider\b', r'\benderman\b',
            r'\bender\s*dragon\b', r'\bwither\b', r'\bvillager\b', r'\bpig\b', r'\bcow\b', r'\bsheep\b',
        
            # English: Game mechanics
            r'\bcraft\b', r'\bcrafting\b', r'\bsmelt\b', r'\bmine\b', r'\bsmithing\b', r'\bbrewing\b',
            r'\bpot(?:ion)?\b', r'\bspawn\b', r'\bxp\b', r'\blevel\b', r'\bsurvive\b',
        
            # English: Tools and weapons
            r'\bpickaxe\b', r'\bsword\b', r'\baxe\b', r'\bhoe\b', r'\bshovel\b', r'\bbow\b',
        
            # English: Enchantments
            r'\bsharpness\b', r'\befficiency\b', r'\bsilk\s*touch\b', r'\bfortune\b', r'\bmending\b',
        
            # Simplified Chinese: Game terms and actions
            r'我的世界', r'麦块', r'生存模式', r'创造模式', r'附魔', r'附魔台', r'经验(?:值|球)?', r'等级', r'升级',
            r'工作台', r'熔炉', r'酿造台', r'合成台', r'末影之眼', r'末影龙', r'凋零', r'村民',
            r'挖矿', r'采矿', r'种地', r'养动物', r'刷怪', r'砍树', r'打怪', r'建房', r'盖房子',
            r'怎么(?:做|制作|挖|找到|获得|合成|附魔|打|刷)', r'如何(?:做|制作|挖|找到|获得|合成|附魔|打|刷)',
            r'哪里(?:可以|能)?(?:找到|挖到|刷到|合成)', r'怎样(?:合成|制作|获得)',
            r'钻石', r'铁(?:锭|矿)', r'金(?:锭|矿)', r'红石', r'青金石', r'绿宝石', r'下界合金',
            r'下界(?:岩|砖)', r'黑曜石', r'末地石', r'灵魂沙', r'萤石', r'泥土', r'木头', r'石头', r'玻璃', r'沙子',
            r'下界', r'终界', r'主世界', r'地狱', r'终界之地', r'维度',
            r'如何开始.*我的世界', r'新手.*我的世界.*怎么玩',
        
            # Traditional Chinese: Game terms and actions (parallel to Simplified)
            r'我的世界', r'麥塊', r'生存模式', r'創造模式', r'附魔', r'附魔台', r'經驗(?:值|球)?', r'等級', r'升級',
            r'工作台', r'熔爐', r'釀造台', r'合成台', r'終界之眼', r'終界龍', r'凋零', r'村民',
            r'挖礦', r'採礦', r'種地', r'養動物', r'刷怪', r'砍樹', r'打怪', r'建房', r'蓋房子',
            r'怎麼(?:做|製作|挖|找到|獲得|合成|附魔|打|刷)', r'如何(?:做|製作|挖|找到|獲得|合成|附魔|打|刷)',
            r'哪裡(?:可以|能)?(?:找到|挖到|刷到|合成)', r'怎樣(?:合成|製作|獲得)',
            r'鑽石', r'鐵(?:錠|礦)', r'金(?:錠|礦)', r'紅石', r'青金石', r'綠寶石', r'下界合金',
            r'下界(?:岩|磚)', r'黑曜石', r'終界石', r'靈魂沙', r'熒石', r'泥土', r'木頭', r'石頭', r'玻璃', r'沙子',
            r'下界', r'終界', r'主世界', r'地獄', r'終界之地', r'維度',
            r'如何開始.*我的世界', r'新手.*我的世界.*怎麼玩'
        ]
    
        # Check if the message contains any Minecraft indicators
        is_minecraft_related = any(re.search(pattern, message_content.lower()) for pattern in minecraft_indicators)
        
        # If it's clearly Minecraft related, we should respond
        if is_minecraft_related:
            print("Minecraft-related question detected - will respond")
            if await second_classify_message(ctx, message_content) == "yes":
                return True
            else:
                return False
            
        # TIER 1: Direct addressing - highest priority rules that guarantee a response
        direct_address_patterns = [
            message_content.lower().startswith("edson,"),
            message_content.lower().startswith("edson，"),  # Chinese comma
            message_content.lower().startswith("edson "),
            message_content.lower().startswith("edson:"),
            message_content.lower().startswith("edson："),
            message_content.lower() == "edson",
            message_content.lower().endswith(" edson"),
            message_content.lower().endswith("edson?"),
            message_content.lower().endswith("edson？"),
            " edson:" in message_content.lower(),
            " edson：" in message_content.lower(),  # Chinese colon
            re.match(r'^edson[\u4e00-\u9fff]', message_content.lower()) is not None  # Match when edson is followed by Chinese characters
        ]
        
        if any(direct_address_patterns):
            print("Direct addressing detected - will respond")
            if await second_classify_message(ctx, message_content) == "yes":
                return True
            else:
                return False
        
        # TIER 2: Explicit request patterns - high likelihood of being intended for the bot
        request_patterns = [
            "请提供", "帮我", "告诉我", "给我", "我想知道", "我需要", "我要",
            "tell me", "explain", "help me", "give me", "show me",
            "how to", "how do", "what is", "why is", "can you", "could you"
        ]
        
        if any(pattern in message_content.lower() for pattern in request_patterns):
            print(f"Request pattern detected - will respond")
            if await second_classify_message(ctx, message_content) == "yes":
                return True
            else:
                return False
            
        # TIER 3: Context-aware evaluation - check if this is part of an ongoing conversation with the bot
        channel_history = []
        try:
            async for msg in ctx.channel.history(limit=5):
                channel_history.append(msg)
        except Exception as e:
            print(f"Error fetching channel history: {e}")
            channel_history = []
            
        # Check if bot was recently active in conversation
        bot_in_recent_convo = False
        for msg in channel_history:
            if msg.author.id == main_bot.user.id:
                bot_in_recent_convo = True
                break
        
        # NEW: Check if there was recent Minecraft discussion in the channel
        minecraft_context = False
        for msg in channel_history:
            if msg.content and any(re.search(pattern, msg.content.lower()) for pattern in minecraft_indicators):
                minecraft_context = True
                break
                
        # TIER 4: If the bot was recently active or there's Minecraft context, apply more lenient rules
        if bot_in_recent_convo or minecraft_context:
            # Direct questions in active conversations are likely for the bot
            if '?' in message_content or '？' in message_content:
                print("Question in active conversation - will respond")
                if await second_classify_message(ctx, message_content) == "yes":
                    return True
                else:
                    return False
            
            # Conversation continuers when bot is active
            conversation_continuers = [
                'and ', 'so ', 'but ', 'what about', 'how about', 'tell ', 'why ', 'because ',
                'then ', 'actually', 'really', 'yeah,', 'yes,', 'no,', 'sure,', 'okay,', 'thanks',
                'thank you', '那么', '所以', '为什么', '怎么', '真的', '是的', '不是', '好的', '谢谢'
            ]
            
            # Only consider conversation continuers if they start the message
            if any(message_content.lower().startswith(term) for term in conversation_continuers):
                print("Conversation continuation in active conversation - will respond")
                return await (ctx, message_content)
            '''    
            # Very short messages (2-3 words) in active conversations may be directed at the bot
            # But they must have some substance
            # Additional check: If message contains a user mention (e.g., <@...>), it's likely directed at a user, not the bot
            if 1 < len(message_content.split()) <= 3 and len(message_content.strip()) > 3:
                if re.search(r"<@!?[0-9]+>", message_content):
                    print("Short message with user mention - not for bot")
                    return False
                print("Short message in active conversation - will respond")
                return True
                '''
        
        # TIER 5: Analyze message content characteristics more carefully
        
        # Handle direct questions more carefully - questions with clear intent
        # Look for question structures that indicate a direct question needing an answer
        question_patterns = [
            # English question patterns with specific formats
            r"^(?:what|who|where|when|why|how|which|whose|whom)\s+\w+.+\?$",  # Complete "wh" questions ending with ?
            r"^(?:can|could|would|should|will|do|does|did|is|are|was|were)\s+\w+.+\?$",  # Complete auxiliary verb questions
            r"^(?:tell|show|explain|describe|define|clarify)\s+(?:me|us)\s+(?:about|how|what|why|when).+$",  # Direct requests
            
            # Chinese question patterns
            r"^(?:什么|谁|哪里|何时|为什么|如何|怎么|哪个|谁的|何人).+[？?]$",  # Chinese "wh" questions
            r"^(?:能|能够|可以|会|应该|要不要|需要|是|有).+吗[？?]$",  # Chinese yes/no questions with 吗
        ]
        
        for pattern in question_patterns:
            if re.search(pattern, message_content.lower()):
                print("Clear question structure detected - will respond")
                if await second_classify_message(ctx, message_content) == "yes":
                    return True
                else:
                    return False
        
        # TIER 6: More selective complex pattern recognition
        # These patterns should be more specific and have a higher threshold
        complex_intent_patterns = [
            # Specific request formats that strongly indicate intent for AI response
            r"^(?:i|我)(?:'m| am| have|'ve)\s+looking\s+for\s+(?:information|advice|help|assistance)\s+(?:on|about|with|regarding)\s+.+$",
            r"^(?:i|我)(?:'m| am| have|'ve)\s+trying\s+to\s+(?:understand|learn|figure\s+out|solve)\s+.+$",
            r"^(?:i|我)\s+need\s+(?:to\s+know|information\s+on|help\s+with)\s+.+$",
            
            # Chinese equivalents of specific request formats
            r"^(?:我)(?:在|正在|想要|需要)\s*(?:寻找|查询|了解|学习|理解|弄懂)\s+.+$",
            r"^(?:请问|问一下|想问|我想问)\s+.+$"
        ]
        
        for pattern in complex_intent_patterns:
            if re.search(pattern, message_content.lower()):
                print(f"Complex specific intent pattern detected - will respond")
                if await second_classify_message(ctx, message_content) == "yes":
                    return True
                else:
                    return False
        
        # TIER 7: Emotion/opinion needs more careful check - only respond if it appears directed
        # at the bot or seeking engagement rather than just expressing emotion
        directed_emotion_patterns = [
            r"(?:i|我)(?:'m| am)\s+(?:feeling|so|very|really)\s+(?:happy|sad|excited|worried|anxious|tired|bored)",
            r"(?:this|that|it)\s+(?:makes|made)\s+me\s+(?:happy|sad|excited|worried|anxious|tired|bored)",
            r"(?:我)(?:很|非常|真的|好|感到)\s*(?:开心|难过|兴奋|担心|焦虑|累了|无聊)"
        ]
        
        for pattern in directed_emotion_patterns:
            if re.search(pattern, message_content.lower()):
                print("Directed emotion content detected - will respond")
                if await second_classify_message(ctx, message_content) == "yes":
                    return True
                else:
                    return False
                
        # NEW: Special check for specific Minecraft database query format
        # This allows users to make direct database queries about Minecraft items
        minecraft_query_patterns = [
            r"(?:how|what|tell me about)\s+(?:is|are|does|do)\s+([a-zA-Z0-9_\s]+)\s+(?:in minecraft|in mc)",
            r"minecraft\s+(?:item|block|mob|entity|biome|recipe|enchant|effect|food)\s*:\s*([a-zA-Z0-9_\s]+)",
            r"mc\s+(?:item|block|mob|entity|biome|recipe|enchant|effect|food)\s*:\s*([a-zA-Z0-9_\s]+)",
            r"查询\s*(?:我的世界|minecraft|mc)\s*(?:物品|方块|生物|实体|群系|合成|附魔|效果|食物)\s*:\s*([a-zA-Z0-9_\s]+)"
        ]
        
        for pattern in minecraft_query_patterns:
            match = re.search(pattern, message_content.lower())
            if match and match.group(1).strip():
                print(f"Direct Minecraft database query detected: {match.group(1).strip()}")
                if await second_classify_message(ctx, message_content) == "yes":
                    return True
                else:
                    return False
        
        # NEW: Special AI classification specifically for Minecraft content
        # Only do this if the message has some potential Minecraft relevance
        potential_mc_terms = [
            "game", "play", "build", "blocks", "craft", "mining", "farm", 
            "游戏", "玩", "建造", "方块", "合成", "挖矿", "农场"
        ]
        
        has_potential_mc_terms = any(term in message_content.lower() for term in potential_mc_terms)
        
        if has_potential_mc_terms and len(message_content.split()) >= 4:
            system_msg = {
                "role": "system",
                "content": (
                    "You are a classifier that determines if a message is asking about Minecraft. "
                    "Examples of Minecraft-related questions:\n"
                    "- How do I craft a diamond pickaxe?\n"
                    "- What's the best way to find netherite?\n"
                    "- Can you help me understand redstone mechanics?\n"
                    "- Where do endermen spawn?\n"
                    "- What are the best enchantments for a sword?\n"
                    "- 怎么做钻石镐?\n"
                    "- 在哪里可以找到下界合金?\n"
                    "- 红石机制是如何工作的?\n"
                    "\nReply ONLY with 'yes' if the message is clearly about Minecraft, or 'no' if it's not."
                )
            }
            
            user_msg = {
                "role": "user",
                "content": message_content
            }
            try:
                decision = client.chat.completions.create(
                    messages=[system_msg, user_msg],
                    max_completion_tokens=5,
                    temperature=0.1,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    model=deployment
                )
                reply = decision.choices[0].message.content.lower().strip()
                print(f"Minecraft classification decision: {reply}")
                
                if reply == 'yes':
                    print("AI confirmed this is a Minecraft question - will respond")
                    return True
            except Exception as e:
                print(f"Error in Minecraft classification: {e}")

        #TIER 8: Final AI classification for complex edge cases, but with a higher bar for responding
        if len(message_content.split()) >= 5 and len(message_content) > 15:
            try:
                reply = classify_message(message_content, bot_in_recent_convo, minecraft_context)
                print(f"AI classification decision: {reply}")

                # Only respond if AI explicitly says 'yes'
                if reply == 'yes':
                    if await second_classify_message(ctx, message_content) == "yes":
                        print("AI confirmed this is a Minecraft question - will respond")
                        return True
                elif reply == 'no':
                    print("AI classified this as not a Minecraft question - not responding")
                    return False
                # Otherwise, default to not responding
                return False
            except Exception as e:
                print(f"Error in AI classification: {e}")
                # If AI classification fails, fall back to a conservative default
                return bot_in_recent_convo  # Only respond if was already in conversation
        
        # Default to not responding for any messages that didn't match our criteria
        print("No response criteria met - not responding")
        return False
        
    except Exception as e:
        print(f"Error in should_respond_to_message: {e}")
        # Be conservative on errors - only respond if already in conversation
        return bot_in_recent_convo

@main_bot.event
async def on_message(message):
    # Ignore messages from bots or DMs
    if message.author.bot:
        return
    print(f"Message author: {message.author.name}")
    print("message:", message.content)
    if message.channel.type == discord.ChannelType.private:
        await apply_dm_message(message.channel, message)
        print("DM sent to admin user")
        return

    # Extract message content
    question = message.content.strip()
    if not question:
        return

    # Check for Minecraft related questions first with dedicated patterns
    minecraft_indicators = [
        # English: Game terms
        r'\bminecraft\b', r'\bmc\b', r'\bsurvival mode\b', r'\bcreative mode\b', r'\bvanilla\b',
    
        # English: Blocks and items
        r'\bdiamond\b', r'\bnetherite\b', r'\biron\b', r'\bgold\b', r'\bemerald\b',
        r'\bobsidian\b', r'\bbedrock\b', r'\bread\s*stone\b', r'\bglowstone\b', r'\bend\s*stone\b',
    
        # English: Biomes and dimensions
        r'\bnether\b', r'\bend\b', r'\boverworld\b', r'\bthe\s*end\b', r'\bdesert\b',
        r'\bjungle\b', r'\btaiga\b', r'\bplains\b', r'\bocean\b', r'\bswamp\b', r'\bmountain\b',
    
        # English: Entities and mobs
        r'\bcreeper\b', r'\bzombie\b', r'\bskeleton\b', r'\bspider\b', r'\benderman\b',
        r'\bender\s*dragon\b', r'\bwither\b', r'\bvillager\b', r'\bpig\b', r'\bcow\b', r'\bsheep\b',
    
        # English: Game mechanics
        r'\bcraft\b', r'\bcrafting\b', r'\bsmelt\b', r'\bmine\b', r'\bsmithing\b', r'\bbrewing\b',
        r'\bpot(?:ion)?\b', r'\bspawn\b', r'\bxp\b', r'\blevel\b', r'\bsurvive\b',
    
        # English: Tools and weapons
        r'\bpickaxe\b', r'\bsword\b', r'\baxe\b', r'\bhoe\b', r'\bshovel\b', r'\bbow\b',
    
        # English: Enchantments
        r'\bsharpness\b', r'\befficiency\b', r'\bsilk\s*touch\b', r'\bfortune\b', r'\bmending\b',
    
        # Simplified Chinese: Game terms and actions
        r'我的世界', r'麦块', r'生存模式', r'创造模式', r'附魔', r'附魔台', r'经验(?:值|球)?', r'等级', r'升级',
        r'工作台', r'熔炉', r'酿造台', r'合成台', r'末影之眼', r'末影龙', r'凋零', r'村民',
        r'挖矿', r'采矿', r'种地', r'养动物', r'刷怪', r'砍树', r'打怪', r'建房', r'盖房子',
        r'怎么(?:做|制作|挖|找到|获得|合成|附魔|打|刷)', r'如何(?:做|制作|挖|找到|获得|合成|附魔|打|刷)',
        r'哪里(?:可以|能)?(?:找到|挖到|刷到|合成)', r'怎样(?:合成|制作|获得)',
        r'钻石', r'铁(?:锭|矿)', r'金(?:锭|矿)', r'红石', r'青金石', r'绿宝石', r'下界合金',
        r'下界(?:岩|砖)', r'黑曜石', r'末地石', r'灵魂沙', r'萤石', r'泥土', r'木头', r'石头', r'玻璃', r'沙子',
        r'下界', r'终界', r'主世界', r'地狱', r'终界之地', r'维度',
        r'如何开始.*我的世界', r'新手.*我的世界.*怎么玩',
    
        # Traditional Chinese: Game terms and actions (parallel to Simplified)
        r'我的世界', r'麥塊', r'生存模式', r'創造模式', r'附魔', r'附魔台', r'經驗(?:值|球)?', r'等級', r'升級',
        r'工作台', r'熔爐', r'釀造台', r'合成台', r'終界之眼', r'終界龍', r'凋零', r'村民',
        r'挖礦', r'採礦', r'種地', r'養動物', r'刷怪', r'砍樹', r'打怪', r'建房', r'蓋房子',
        r'怎麼(?:做|製作|挖|找到|獲得|合成|附魔|打|刷)', r'如何(?:做|製作|挖|找到|獲得|合成|附魔|打|刷)',
        r'哪裡(?:可以|能)?(?:找到|挖到|刷到|合成)', r'怎樣(?:合成|製作|獲得)',
        r'鑽石', r'鐵(?:錠|礦)', r'金(?:錠|礦)', r'紅石', r'青金石', r'綠寶石', r'下界合金',
        r'下界(?:岩|磚)', r'黑曜石', r'終界石', r'靈魂沙', r'熒石', r'泥土', r'木頭', r'石頭', r'玻璃', r'沙子',
        r'下界', r'終界', r'主世界', r'地獄', r'終界之地', r'維度',
        r'如何開始.*我的世界', r'新手.*我的世界.*怎麼玩'
    ]
    
    # Check if the message contains any Minecraft indicators
    is_minecraft_question = any(re.search(pattern, question.lower()) for pattern in minecraft_indicators)
    
    # Continue with normal processing for non-Minecraft questions
    # Filter: Should the bot respond?
    should_reply = await should_respond_to_message(message, question)
    print(f"Should reply: {should_reply}")

    if not should_reply:
        return

    # Check if this is likely an identity question about a referenced image
    is_identity_question = False
    identity_patterns = [
    # Person identification (Chinese)
    "这是谁", "这个是谁", "这位是谁", "这张图是谁", "图里是谁", "谁是这个", "这人是谁", 
    "这个人是谁", "这位人是谁", "图中的人是谁", "这个角色是谁", "这个动漫人物是谁",

    # Person identification (English)
    "who is this", "who's this", "who is in this", "who is that", "who's that",
    "who is the person", "who is the character", "identify this person", "identify the person",
    "name of this character", "name of this person", "who is shown here",

    # Object/animal identification (Chinese)
    "这是什么", "这个是什么", "这是什么东西", "这物品是什么", "图里是什么", "这动物是什么",
    "这是什么物种", "这是什么动物", "这是什么物品", "这个物体是什么", "这是哪个物种",
    "图中的是什么", "是什么品种", "这是什么型号", "这是什么牌子",

    # Object/animal identification (English)
    "what is this", "what's this", "what is that", "what's that", "what is in this",
    "what is shown here", "identify this", "name of this", "what animal is this", 
    "what breed is this", "what object is this", "what species is this", "what model is this",
    "what brand is this", "what product is this",
    
    # Location identification (both languages)
    "这是哪里", "这是什么地方", "这是哪个地方", "这个地方是哪", "图里是哪个地方",
    "where is this", "what place is this", "what location is this", "identify this place",
    
    # General identification (both languages)
    "can you identify", "能认出", "能辨认", "这是", "what am i looking at",
    "tell me about this", "explain this image", "describe what's in this"
]
    
    if message.reference and any(pattern in question.lower() for pattern in identity_patterns):
        is_identity_question = True
        print(f"Detected identity question about referenced content: {question}")

    # Collect attachments
    attachment_blocks = []
    attachments_for_db = []
    
    for attachment in message.attachments:
        if attachment.content_type:
            # Handle all image types including GIFs
            if attachment.content_type.startswith("image/"):
                # Special handling for GIFs
                if attachment.content_type == "image/gif":
                    try:
                        # Process GIF to extract distinct frames
                        print(f"Processing GIF attachment in message: {attachment.url}")
                        ctx = await main_bot.get_context(message)
                        uploaded_frames, gif_context, frame_messages = await analyze_gif_for_edson(ctx, attachment.url)
                        
                        # Add frames to attachment blocks
                        attachment_blocks.extend(uploaded_frames)
                        
                        # Add GIF context if available
                        if gif_context:
                            context_messages = []  # Initialize context_messages
                            context_messages.append(gif_context)
                            
                        # Store original GIF URL for database
                        attachments_for_db.append({
                            "type": "image_url", 
                            "image_url": {"url": attachment.url}
                        })
                        
                        continue  # Skip standard image handling for GIFs
                    except Exception as e:
                        print(f"Error processing GIF: {str(e)}, falling back to standard image handling")
                        # Fall back to regular image handling if processing fails
                
                # Standard image handling for non-GIFs or if GIF processing failed
                block = {
                    "type": "image_url", 
                    "image_url": {"url": attachment.url}
                }
                attachment_blocks.append(block)
                attachments_for_db.append(block)
                
            elif attachment.content_type == "text/plain" or attachment.filename.endswith('.txt'):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(attachment.url) as resp:
                            if resp.status == 200:
                                text_data = await resp.text()
                                # Don't limit to 2000 chars for file organization tasks
                                block = {
                                    "type": "text",
                                    "text": f"Attached text file contents:\n{text_data}"
                                }
                                attachment_blocks.append(block)
                                attachments_for_db.append(block)
                                print(f"Processed text attachment: {attachment.filename}")
                except Exception as e:
                    await message.channel.send(f"Failed to read attachment: {e}")
                    print(f"Error processing attachment: {e}")
    
    # Save user message to DB
    print("message author: ", message.author)
    print("message content: ", question)
    print("message attachments: ", message.attachments)
    print("attachment blocks: ", attachment_blocks)
    print("attachments for db: ", attachments_for_db)
    insert_message(message.author, "user", question, attachments=attachments_for_db)

    async with message.channel.typing():
        # Build context
        ctx = await main_bot.get_context(message)
        context_messages = await build_multicontext(ctx, question)
    
        # Add user attachments
        if attachment_blocks:
            context_messages.append({
                "role": "user",
                "content": attachment_blocks
            })
    
        # Add replied message reference (if any)
        if message.reference:
            try:
                ref = await message.channel.fetch_message(message.reference.message_id)
                
                # If this is an identity question about an image, add special context
                if is_identity_question:
                    context_messages.append({
                        "role": "system",
                        "content": "IMPORTANT: The user is asking about the identity of someone or something in the referenced image. Focus your response on identifying what's in the image."
                    })
                    
                add_reference_message_with_time(context_messages, ref, message.author.display_name)
    
                for attachment in ref.attachments:
                    if attachment.content_type:
                        # Special handling for GIFs in replied messages
                        if attachment.content_type == "image/gif":
                            try:
                                # Process GIF to extract distinct frames
                                print(f"Processing GIF in replied message: {attachment.url}")
                                uploaded_frames, gif_context, frame_messages = await analyze_gif_for_edson(ctx, attachment.url)
                                
                                # Add frames to attachment blocks
                                for frame in uploaded_frames:
                                    context_messages.append({
                                        "role": "user",
                                        "content": [frame]
                                    })
                                    
                                # Add GIF context if available
                                if gif_context:
                                    context_messages.append(gif_context)
                                    
                                continue  # Skip standard handling for GIFs
                            except Exception as e:
                                print(f"Error processing GIF in reply: {str(e)}, falling back to standard handling")
                        
                        # Standard handling for non-GIFs or if GIF processing failed
                        if attachment.content_type.startswith("image/"):
                            context_messages.append({
                                "role": "user",
                                "content": [{
                                    "type": "image_url",
                                    "image_url": {"url": attachment.url}
                                }]
                            })
                        elif attachment.content_type == "text/plain":
                            async with aiohttp.ClientSession() as session:
                                async with session.get(attachment.url) as resp:
                                    if resp.status == 200:
                                        text_data = await resp.text()
                                        text_data = text_data[:2000]
                                        context_messages.append({
                                            "role": "user",
                                            "content": [{
                                                "type": "text",
                                                "text": f"[Attached reply file]\n{text_data}"
                                            }]
                                        })
            except Exception as e:
                await message.channel.send(f"Failed to read replied message: {e}")
    
        # Ensure the context includes the last response from the assistant
        if message.reference:
            # Make sure the assistant's response is included in the context if this is a reply
            context_messages.append({
                "role": "assistant",
                "content": f"Previous response: {message.reference.resolved.content}"
            })
        
        if is_minecraft_question:
            '''
            print("Minecraft-related question detected - will respond")
            # This is a Minecraft question - use the database functions to answer it
            context_messages = []
    
            context_messages.append({
                        "role": "system",
                        "content": f"REFRENCE: {await process_minecraft_question(message, question)}."
                    })
                    '''
            print("Minecraft-related question detected - will respond")
            response = client.chat.completions.create(
                        messages=context_messages,
                        max_completion_tokens=8000,
                        temperature=0.7,
                        top_p=1.0,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                        model=deployment
                    )
            answer = response.choices[0].message.content.strip()
            print("-------------------------------------------------")
            print(f"minecraft response: {answer}")
    
            
            answer = clean_response(answer)
    
            # Save assistant reply to DB
            insert_message(message.author, "assistant", answer)
    
            print(f"context_messages: {context_messages}")
            print(f"User: {message.author} asked: {question}")
            print(f"Assistant: {answer}")
    
            # Send the answer in parts if it's too long
            if len(answer) > 2000:
                await send_message_in_parts(message.channel, answer, [])
            else:
                await message.channel.send(answer)
    
            return
    
        # Generate response
        try:
            response = client.chat.completions.create(
                        messages=context_messages,
                        max_completion_tokens=8000,
                        temperature=0.7,
                        top_p=1.0,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                        model=deployment
                    )
            answer = response.choices[0].message.content.strip()
            print("-------------------------------------------------")
            print(f"Generated response: {answer}")
    
            
            answer = clean_response(answer)
    
            # Save assistant reply to DB
            insert_message(message.author, "assistant", answer)
    
            print(f"context_messages: {context_messages}")
            print(f"User: {message.author} asked: {question}")
            print(f"Assistant: {answer}")
    
            # Send the answer in parts if it's too long
            if len(answer) > 2000:
                await send_message_in_parts(message.channel, answer, [])
            else:
                await message.channel.send(answer)
                
        except Exception as e:
            await message.channel.send(f"Error during processing: {e}")

if __name__ == '__main__':
    try:
        main_bot.run("", reconnect=True)

    except Exception as ex:
        sys.path.append("error_handle.txt")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        with open("error_handle.txt",mode="r",encoding="utf-8") as file_r:
            rd=file_r.read()
        file_w=open("error_handle.txt", mode="w")
        fn=os.path.realpath(__file__)
        zt=time.strftime("%Y")
        zt1=time.strftime("%m")
        zt2=time.strftime("%d")
        zt3=time.strftime("%H")
        zt4=time.strftime("%M")
        zt5=time.strftime("%S")
        file_w.write(f"{rd}\n\n{zt}y{zt1}M{zt2}d,{zt3}h{zt4}m{zt5}s\nerror_msg:{ex}\nerror_type:{exc_type}\nerror_line:{exc_tb.tb_lineno}\ndoc_name:{fname}\npath:{fn}")
        file_w.close()
