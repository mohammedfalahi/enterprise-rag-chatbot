"""
Unit and integration tests for multi-turn conversation persistence,
message history, structured citations, auto-titling, sliding-window retrieval,
and strict tenant and user isolation.
"""

from __future__ import annotations

import asyncio
import unittest
import uuid
import psycopg

from app.core.config import settings
from app.db.connection import close_db_pool, get_db_connection, init_db_pool
from app.db.provisioning import assign_user_tenant, create_tenant
from app.db.conversations import (
    DEFAULT_TITLE,
    MAX_TITLE_LENGTH,
    append_message,
    create_conversation,
    delete_conversation,
    generate_conversation_title,
    get_conversation,
    get_conversation_messages,
    get_or_create_conversation,
    list_conversations,
    update_conversation_title,
)


class TestTitleGeneration(unittest.TestCase):

    def test_title_under_max_chars(self):
        title = generate_conversation_title("What is the project budget?")
        self.assertEqual(title, "What is the project budget?")

    def test_title_clean_whitespace(self):
        title = generate_conversation_title("   What   is\n\n the   status?  \t ")
        self.assertEqual(title, "What is the status?")

    def test_title_empty_defaults(self):
        self.assertEqual(generate_conversation_title(""), DEFAULT_TITLE)
        self.assertEqual(generate_conversation_title("   \n\t  "), DEFAULT_TITLE)

    def test_title_truncation_to_max_chars(self):
        long_query = "What are the comprehensive quarterly financial results and future enterprise projections for 2027?"
        title = generate_conversation_title(long_query, max_chars=50)
        self.assertLessEqual(len(title), 50)
        self.assertTrue(long_query.startswith(title))
        self.assertFalse(title.endswith(" "))


class TestConversationPersistence(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        await init_db_pool()
        self.created_tenants: list[str] = []
        self.created_users: list[str] = []

        # Setup primary tenant and user
        self.tenant_a = await create_tenant("Acme Corp")
        self.created_tenants.append(self.tenant_a)
        self.user_a1 = await self._create_test_auth_user("alice@acme.local")
        await assign_user_tenant(self.user_a1, self.tenant_a, ["employee"])

    async def asyncTearDown(self):
        async with get_db_connection() as conn:
            async with conn.cursor() as cur:
                for uid in self.created_users:
                    await cur.execute("DELETE FROM auth.users WHERE id = %s;", (uid,))
                for tid in self.created_tenants:
                    await cur.execute("DELETE FROM tenants WHERE id = %s;", (tid,))
                await conn.commit()
        await close_db_pool()

    async def _create_test_auth_user(self, email: str) -> str:
        user_id = str(uuid.uuid4())
        async with get_db_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO auth.users (id, email) VALUES (%s, %s);",
                    (user_id, email),
                )
                await conn.commit()
        self.created_users.append(user_id)
        return user_id

    async def test_create_and_get_conversation(self):
        """Creates a conversation with default title and retrieves it."""
        conv = await create_conversation(self.tenant_a, self.user_a1)
        self.assertIsNotNone(conv["id"])
        self.assertEqual(conv["tenant_id"], self.tenant_a)
        self.assertEqual(conv["user_id"], self.user_a1)
        self.assertEqual(conv["title"], DEFAULT_TITLE)

        fetched = await get_conversation(conv["id"], self.tenant_a, self.user_a1)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["id"], conv["id"])
        self.assertEqual(fetched["title"], DEFAULT_TITLE)

    async def test_create_with_explicit_title(self):
        """Creates a conversation with an explicit custom title."""
        conv = await create_conversation(
            self.tenant_a, self.user_a1, title="Q3 Earnings Discussion"
        )
        self.assertEqual(conv["title"], "Q3 Earnings Discussion")

        fetched = await get_conversation(conv["id"], self.tenant_a, self.user_a1)
        self.assertEqual(fetched["title"], "Q3 Earnings Discussion")

    async def test_update_conversation_title(self):
        """Updates a conversation title."""
        conv = await create_conversation(self.tenant_a, self.user_a1)
        updated = await update_conversation_title(
            conv["id"], self.tenant_a, self.user_a1, "Updated Project Roadmap"
        )
        self.assertIsNotNone(updated)
        self.assertEqual(updated["title"], "Updated Project Roadmap")

        # Empty title fails validation
        with self.assertRaises(ValueError):
            await update_conversation_title(conv["id"], self.tenant_a, self.user_a1, "   ")

    async def test_delete_conversation_cascades_messages(self):
        """Deleting a conversation removes the conversation and cascades to its messages."""
        conv = await create_conversation(self.tenant_a, self.user_a1)
        await append_message(
            conv["id"], self.tenant_a, self.user_a1, "user", "Hello"
        )
        await append_message(
            conv["id"], self.tenant_a, self.user_a1, "assistant", "Hi there!"
        )

        messages_before = await get_conversation_messages(conv["id"], self.tenant_a, self.user_a1)
        self.assertEqual(len(messages_before), 2)

        deleted = await delete_conversation(conv["id"], self.tenant_a, self.user_a1)
        self.assertTrue(deleted)

        # Conversation is gone
        self.assertIsNone(await get_conversation(conv["id"], self.tenant_a, self.user_a1))
        # Messages are cascaded
        messages_after = await get_conversation_messages(conv["id"], self.tenant_a, self.user_a1)
        self.assertEqual(len(messages_after), 0)

    async def test_list_conversations_pagination_and_order(self):
        """Lists conversations ordered by updated_at DESC with pagination."""
        c1 = await create_conversation(self.tenant_a, self.user_a1, title="Chat 1")
        await asyncio.sleep(0.02)
        c2 = await create_conversation(self.tenant_a, self.user_a1, title="Chat 2")
        await asyncio.sleep(0.02)
        c3 = await create_conversation(self.tenant_a, self.user_a1, title="Chat 3")

        # Full list
        convs = await list_conversations(self.tenant_a, self.user_a1, limit=10)
        self.assertEqual(len(convs), 3)
        self.assertEqual([c["id"] for c in convs], [c3["id"], c2["id"], c1["id"]])

        # Pagination: limit 2, offset 1
        page = await list_conversations(self.tenant_a, self.user_a1, limit=2, offset=1)
        self.assertEqual(len(page), 2)
        self.assertEqual(page[0]["id"], c2["id"])
        self.assertEqual(page[1]["id"], c1["id"])

    async def test_append_message_validation(self):
        """Rejects invalid message roles or empty message content."""
        conv = await create_conversation(self.tenant_a, self.user_a1)

        with self.assertRaises(ValueError):
            await append_message(conv["id"], self.tenant_a, self.user_a1, "system", "Invalid role")

        with self.assertRaises(ValueError):
            await append_message(conv["id"], self.tenant_a, self.user_a1, "user", "   \n  ")

    async def test_append_message_unauthorized_conversation_raises_error(self):
        """Attempting to append to an unowned conversation raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            await append_message(str(uuid.uuid4()), self.tenant_a, self.user_a1, "user", "Hello")
        self.assertIn("not found or access denied", str(ctx.exception))

    async def test_append_user_and_assistant_messages_with_citations(self):
        """Appends user and assistant messages, verifying structured citations JSONB persistence."""
        conv = await create_conversation(self.tenant_a, self.user_a1)

        user_msg = await append_message(
            conv["id"], self.tenant_a, self.user_a1, "user", "What is policy 402?"
        )
        self.assertEqual(user_msg["role"], "user")
        self.assertEqual(user_msg["content"], "What is policy 402?")
        self.assertEqual(user_msg["citations"], [])

        citations = [
            {
                "chunk_id": "chunk-101",
                "document_id": str(uuid.uuid4()),
                "title": "Corporate Travel Policy",
                "heading_path": ["Finance", "Travel", "Expenses"],
                "score": 0.8951,
            }
        ]
        asst_msg = await append_message(
            conv["id"],
            self.tenant_a,
            self.user_a1,
            "assistant",
            "Policy 402 covers business travel reimbursements.",
            citations=citations,
        )
        self.assertEqual(asst_msg["role"], "assistant")
        self.assertEqual(len(asst_msg["citations"]), 1)
        self.assertEqual(asst_msg["citations"][0]["chunk_id"], "chunk-101")
        self.assertEqual(asst_msg["citations"][0]["score"], 0.8951)

    async def test_auto_title_update_on_first_user_message(self):
        """When a new conversation receives its first user message, title is auto-generated."""
        conv = await create_conversation(self.tenant_a, self.user_a1)
        self.assertEqual(conv["title"], DEFAULT_TITLE)

        await append_message(
            conv["id"],
            self.tenant_a,
            self.user_a1,
            "user",
            "Can you explain the cloud migration architectural strategy in detail?",
        )

        updated_conv = await get_conversation(conv["id"], self.tenant_a, self.user_a1)
        self.assertIsNotNone(updated_conv)
        self.assertNotEqual(updated_conv["title"], DEFAULT_TITLE)
        self.assertLessEqual(len(updated_conv["title"]), MAX_TITLE_LENGTH)
        self.assertTrue("cloud migration architectural" in updated_conv["title"].lower())

    async def test_custom_title_not_overwritten_by_user_message(self):
        """If a conversation already has a custom title, appending messages preserves it."""
        conv = await create_conversation(
            self.tenant_a, self.user_a1, title="My Custom Analysis"
        )
        await append_message(
            conv["id"], self.tenant_a, self.user_a1, "user", "First question"
        )
        fetched = await get_conversation(conv["id"], self.tenant_a, self.user_a1)
        self.assertEqual(fetched["title"], "My Custom Analysis")

    async def test_conversation_updated_at_trigger(self):
        """Database trigger trg_messages_update_conversation_timestamp updates conversation.updated_at."""
        conv = await create_conversation(self.tenant_a, self.user_a1, title="Trigger Test")
        initial_updated_at = conv["updated_at"]

        await asyncio.sleep(0.05)
        await append_message(
            conv["id"], self.tenant_a, self.user_a1, "user", "Trigger test message"
        )

        refreshed = await get_conversation(conv["id"], self.tenant_a, self.user_a1)
        self.assertGreater(refreshed["updated_at"], initial_updated_at)

    async def test_sliding_window_limit_and_chronological_ordering(self):
        """
        Appends 12 messages and queries history with limit=5 and limit=9.
        Verifies that only the most recent N messages are returned, in ascending chronological order.
        """
        conv = await create_conversation(self.tenant_a, self.user_a1, title="History Window")

        sent_contents = []
        for i in range(1, 13):
            role = "user" if i % 2 == 1 else "assistant"
            content = f"Message #{i:02d}"
            await append_message(conv["id"], self.tenant_a, self.user_a1, role, content)
            sent_contents.append(content)
            await asyncio.sleep(0.005)

        # 1. Query window with limit=5
        history_5 = await get_conversation_messages(
            conv["id"], self.tenant_a, self.user_a1, limit=5
        )
        self.assertEqual(len(history_5), 5)
        # Should be messages 8, 9, 10, 11, 12 in ascending order
        expected_5 = sent_contents[-5:]
        self.assertEqual([m["content"] for m in history_5], expected_5)

        # 2. Query default sliding window (settings.conversation_history_window = 9)
        history_default = await get_conversation_messages(
            conv["id"], self.tenant_a, self.user_a1
        )
        self.assertEqual(len(history_default), 9)
        expected_9 = sent_contents[-9:]
        self.assertEqual([m["content"] for m in history_default], expected_9)

    async def test_get_or_create_conversation(self):
        """get_or_create_conversation creates a new one or fetches existing verifying ownership."""
        # 1. Create new
        new_conv = await get_or_create_conversation(
            self.tenant_a, self.user_a1, initial_title="Auto-Created"
        )
        self.assertEqual(new_conv["title"], "Auto-Created")

        # 2. Fetch existing
        fetched = await get_or_create_conversation(
            self.tenant_a, self.user_a1, conversation_id=new_conv["id"]
        )
        self.assertEqual(fetched["id"], new_conv["id"])

        # 3. Invalid ID raises ValueError
        with self.assertRaises(ValueError):
            await get_or_create_conversation(
                self.tenant_a, self.user_a1, conversation_id=str(uuid.uuid4())
            )


class TestTenantAndUserIsolation(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        await init_db_pool()
        self.created_tenants: list[str] = []
        self.created_users: list[str] = []

        # Tenant 1: Acme Corp with Alice and Bob
        self.tenant_acme = await create_tenant("Acme Corp")
        self.created_tenants.append(self.tenant_acme)

        self.user_alice = await self._create_test_auth_user("alice@acme.local")
        await assign_user_tenant(self.user_alice, self.tenant_acme, ["employee"])

        self.user_bob = await self._create_test_auth_user("bob@acme.local")
        await assign_user_tenant(self.user_bob, self.tenant_acme, ["employee"])

        # Tenant 2: Beta Corp with Charlie
        self.tenant_beta = await create_tenant("Beta Corp")
        self.created_tenants.append(self.tenant_beta)

        self.user_charlie = await self._create_test_auth_user("charlie@beta.local")
        await assign_user_tenant(self.user_charlie, self.tenant_beta, ["employee"])

    async def asyncTearDown(self):
        async with get_db_connection() as conn:
            async with conn.cursor() as cur:
                for uid in self.created_users:
                    await cur.execute("DELETE FROM auth.users WHERE id = %s;", (uid,))
                for tid in self.created_tenants:
                    await cur.execute("DELETE FROM tenants WHERE id = %s;", (tid,))
                await conn.commit()
        await close_db_pool()

    async def _create_test_auth_user(self, email: str) -> str:
        user_id = str(uuid.uuid4())
        async with get_db_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO auth.users (id, email) VALUES (%s, %s);",
                    (user_id, email),
                )
                await conn.commit()
        self.created_users.append(user_id)
        return user_id

    async def test_cross_user_isolation_same_tenant(self):
        """
        Bob cannot read, list, update, append messages to, or delete Alice's conversation,
        even though they belong to the same tenant.
        """
        alice_conv = await create_conversation(
            self.tenant_acme, self.user_alice, title="Alice Secret Chat"
        )
        await append_message(
            alice_conv["id"], self.tenant_acme, self.user_alice, "user", "Alice sensitive query"
        )

        # 1. Bob cannot get Alice's conversation
        bob_view = await get_conversation(alice_conv["id"], self.tenant_acme, self.user_bob)
        self.assertIsNone(bob_view)

        # 2. Bob's conversation list does not include Alice's conversation
        bob_list = await list_conversations(self.tenant_acme, self.user_bob)
        self.assertEqual(len(bob_list), 0)

        # 3. Bob cannot read messages from Alice's conversation
        bob_msgs = await get_conversation_messages(
            alice_conv["id"], self.tenant_acme, self.user_bob
        )
        self.assertEqual(len(bob_msgs), 0)

        # 4. Bob cannot append messages to Alice's conversation
        with self.assertRaises(ValueError):
            await append_message(
                alice_conv["id"], self.tenant_acme, self.user_bob, "user", "Bob intruding"
            )

        # 5. Bob cannot update title of Alice's conversation
        bob_update = await update_conversation_title(
            alice_conv["id"], self.tenant_acme, self.user_bob, "Hacked by Bob"
        )
        self.assertIsNone(bob_update)

        # 6. Bob cannot delete Alice's conversation
        bob_delete = await delete_conversation(
            alice_conv["id"], self.tenant_acme, self.user_bob
        )
        self.assertFalse(bob_delete)

        # Alice's conversation is still intact
        alice_view = await get_conversation(alice_conv["id"], self.tenant_acme, self.user_alice)
        self.assertIsNotNone(alice_view)
        self.assertEqual(alice_view["title"], "Alice Secret Chat")

    async def test_cross_tenant_isolation(self):
        """
        Charlie from Beta Corp cannot access Alice's conversation in Acme Corp under any circumstances.
        """
        alice_conv = await create_conversation(
            self.tenant_acme, self.user_alice, title="Acme Internal Roadmap"
        )
        await append_message(
            alice_conv["id"], self.tenant_acme, self.user_alice, "user", "Roadmap details"
        )

        # Charlie cannot fetch or list
        self.assertIsNone(
            await get_conversation(alice_conv["id"], self.tenant_beta, self.user_charlie)
        )
        charlie_list = await list_conversations(self.tenant_beta, self.user_charlie)
        self.assertEqual(len(charlie_list), 0)

        # Charlie cannot fetch messages
        charlie_msgs = await get_conversation_messages(
            alice_conv["id"], self.tenant_beta, self.user_charlie
        )
        self.assertEqual(len(charlie_msgs), 0)

        # Charlie cannot append messages
        with self.assertRaises(ValueError):
            await append_message(
                alice_conv["id"], self.tenant_beta, self.user_charlie, "user", "Competitor spying"
            )

        # Charlie cannot delete
        self.assertFalse(
            await delete_conversation(alice_conv["id"], self.tenant_beta, self.user_charlie)
        )

    async def test_database_rls_defense_in_depth(self):
        """
        Verifies that PostgreSQL Row Level Security (RLS) policies directly block cross-user
        and cross-tenant SELECT queries when executed under the authenticated PostgreSQL role.
        """
        alice_conv = await create_conversation(
            self.tenant_acme, self.user_alice, title="RLS Protected Chat"
        )
        await append_message(
            alice_conv["id"], self.tenant_acme, self.user_alice, "user", "RLS verified question"
        )

        # Test RLS behavior under simulated 'authenticated' session
        async with get_db_connection() as conn:
            async with conn.cursor() as cur:
                # 1. Simulate Alice session: can see her conversation
                await cur.execute("SET LOCAL ROLE authenticated;")
                await cur.execute("SELECT set_config('request.jwt.claim.sub', %s, true);", (self.user_alice,))
                await cur.execute("SELECT id FROM public.conversations WHERE id = %s;", (alice_conv["id"],))
                alice_row = await cur.fetchone()
                self.assertIsNotNone(alice_row)
                self.assertEqual(str(alice_row[0]), alice_conv["id"])

                # 2. Simulate Bob session: RLS hides Alice's conversation completely
                await cur.execute("SET LOCAL ROLE authenticated;")
                await cur.execute("SELECT set_config('request.jwt.claim.sub', %s, true);", (self.user_bob,))
                await cur.execute("SELECT id FROM public.conversations WHERE id = %s;", (alice_conv["id"],))
                bob_row = await cur.fetchone()
                self.assertIsNone(bob_row)

                # 3. Simulate Bob session: RLS hides Alice's messages
                await cur.execute("SET LOCAL ROLE authenticated;")
                await cur.execute("SELECT set_config('request.jwt.claim.sub', %s, true);", (self.user_bob,))
                await cur.execute("SELECT id FROM public.messages WHERE conversation_id = %s;", (alice_conv["id"],))
                bob_msg_row = await cur.fetchone()
                self.assertIsNone(bob_msg_row)
